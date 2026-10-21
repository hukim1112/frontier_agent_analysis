"""
===============================================================================
[H-01] 5-Stage Context Compactor Pipeline
===============================================================================
Source: frontier-agent-lab/modules/claude_code/compactor.py

Claude Code 아키텍처의 5단계 컨텍스트 압축 엔진:
1. SnipCompactor      : Age 기반 오래된 ToolMessage 1줄 스텁 대체 (wrap_model_call, 비파괴)
2. MicroCompactor     : 대형 출력(>5000 chars) 디스크 스왑 (wrap_model_call, 결정론적 해시 파일명)
3. ContextCollapse    : 연속 탐색 스텝(3+) 그룹 접기 (wrap_model_call, 변경성 도구 자동 제외)
4. AutoCompactor      : 토큰 임계치 초과 시 LLM 요약 + AmnesiaGuard 연동 (before_model, state 교체)
5. ReactiveCompactor  : 413 에러 대응 턴 경계(Turn-Boundary) 기반 슬라이싱 (wrap_model_call, 413 방화벽)

통합 미들웨어: CompactorMiddleware(AgentMiddleware) + create_compactor_middleware()
- before_model / abefore_model : Stage 1~3 적용 후에도 임계치를 넘으면, 원본 state에서 "이전 턴"만
                                 요약해 state["messages"]를 교체 (현재 턴은 그대로 보존, 턴당 최대 1회)
                                 (원문 전체본은 MemoryMiddleware가 transcript.db에 별도 보존)
- wrap_model_call / awrap_model_call : 매 모델 호출마다 Stage 1~3(Snip → Micro → Collapse) 사본 압축 + Stage 5(Reactive 413 방어)

요약·복원 블록은 HumanMessage로 넣고 additional_kwargs["harness_origin"]과 첫 줄 머리말로
"하네스가 만든 메시지"임을 표시합니다. 턴 경계는 app.utils.message_markers 규칙으로 판별합니다.
===============================================================================
"""

import os
import re
import json
import time
import uuid
import asyncio
import hashlib
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import AgentMiddleware

from app.utils.message_markers import (
    AUTO_GENERATED_NOTE,
    COMPACTION_SUMMARY_HEADER,
    ORIGIN_AMNESIA_RESTORE,
    ORIGIN_COMPACTION_SUMMARY,
    is_compaction_artifact,
    is_compaction_summary,
    last_user_turn_start,
    message_origin,
    origin_kwargs,
    user_turn_starts,
)


# =============================================================================
# Blocklist: Tools that MUST NOT be collapsed (write/modify/execute operations)
# =============================================================================
UNSAFE_TO_COLLAPSE_TOOLS = {
    # File write/edit operations (generic names + 프로젝트 실제 도구명)
    "write_file", "create_file", "edit_file", "replace_file_content",
    "multi_replace_file_content", "write_to_file", "patch_file",
    "file_writer", "file_edit", "notebook_edit",
    # Command / code execution
    "run_command", "execute", "exec", "shell", "bash", "terminal",
    "bash_command", "data_query",
    # Artifact generation (analyst: 차트/엑셀/HTML/포맷 변환 산출물 생성)
    "chart_generator", "excel_writer", "html_report", "file_converter",
    # Browser actions with side effects (navigator)
    "interact_page", "browse_web",
    # File delete operations
    "delete_file", "remove", "rm", "rmdir",
    # Plan / task management
    "update_plan", "create_plan", "set_plan", "enter_plan", "exit_plan",
    "task_create", "task_update",
    # Long-term memory writes & sub-agent delegation
    "memory", "invoke_sub_agent",
    # Git operations
    "git_commit", "git_push", "git_merge",
}


# =============================================================================
# 1. SnipCompactor (Age-based Old Tool Output Truncation)
# =============================================================================
class SnipCompactor:
    """Replaces verbose tool results older than age_threshold turns with concise 1-line stubs."""
    def __init__(self, age_threshold: int = 2):
        self.age_threshold = age_threshold

    def compact(self, messages: list) -> tuple[list, bool]:
        # 턴(Turn) 구분: 실제 사용자 메시지만 턴 시작으로 간주 (요약·복원 블록·StopHooks 피드백 제외)
        human_indices = user_turn_starts(messages)
        total_turns = len(human_indices)
        
        # 턴 수가 age_threshold 이하이면 스니핑 불필요
        if total_turns <= self.age_threshold:
            return messages, False

        # 최근 age_threshold개의 턴이 시작되는 인덱스를 cutoff로 지정
        cutoff_msg_idx = human_indices[total_turns - self.age_threshold]

        modified = False
        new_messages = list(messages)

        for idx in range(cutoff_msg_idx):
            msg = new_messages[idx]
            if isinstance(msg, ToolMessage) and len(str(msg.content)) > 80:
                tool_name = getattr(msg, "name", "tool") or "tool"
                snipped_content = f"[Tool result snipped: Executed '{tool_name}' successfully ({len(str(msg.content))} chars)]"
                new_messages[idx] = ToolMessage(content=snipped_content, tool_call_id=msg.tool_call_id, name=tool_name)
                modified = True

        return new_messages, modified


# =============================================================================
# 2. MicroCompactor (Size-based Disk Swap for Large Tool Outputs)
# =============================================================================
class MicroCompactor:
    """Swaps large tool outputs exceeding max_chars to local disk swap files.
    Includes idempotency guard to skip already-swapped stubs."""
    SWAP_MARKER = "microcompacted to disk:"  # Idempotency marker

    def __init__(self, max_chars: int = 5000, swap_dir: str = "./artifacts/swaps"):
        self.max_chars = max_chars
        self.swap_dir = swap_dir
        os.makedirs(self.swap_dir, exist_ok=True)

    def compact(self, messages: list) -> tuple[list, bool]:
        modified = False
        new_messages = []

        for msg in messages:
            content_str = str(msg.content) if msg.content else ""
            # Idempotency guard: skip already-swapped stubs
            if isinstance(msg, ToolMessage) and self.SWAP_MARKER in content_str:
                new_messages.append(msg)
                continue
            if isinstance(msg, ToolMessage) and len(content_str) > self.max_chars:
                tc_id = getattr(msg, "tool_call_id", "") or ""
                digest = hashlib.sha1(
                    f"{tc_id}:{content_str}".encode("utf-8", errors="ignore")
                ).hexdigest()[:12]
                swap_filename = f"swap_{digest}.txt"
                swap_path = os.path.join(self.swap_dir, swap_filename)
                if not os.path.exists(swap_path):
                    with open(swap_path, "w", encoding="utf-8") as f:
                        f.write(content_str)
                
                tool_name = getattr(msg, "name", "tool") or "tool"
                swap_stub = (
                    f"[Output ({len(content_str)} chars) microcompacted to disk: {swap_path}]\n"
                    f"💡 Actionable Hint: If you need specific details from this large result, use "
                    f"file inspection tools (e.g. `read_file` with line slices or `grep_search`) on '{swap_path}' "
                    f"instead of re-running the full query."
                )
                new_messages.append(ToolMessage(content=swap_stub, tool_call_id=msg.tool_call_id, name=tool_name))
                modified = True
            else:
                new_messages.append(msg)

        return new_messages, modified


# =============================================================================
# 3. ContextCollapse (Grouping Intermediate Exploration Work Blocks)
# =============================================================================
class ContextCollapse:
    """Folds 3+ consecutive exploration/research tool calls into 1 collapsed snapshot SystemMessage.
    Uses UNSAFE_TO_COLLAPSE_TOOLS blocklist: groups containing write/execute tools are NOT collapsed.
    Preserves tool_call_ids in collapsed message additional_kwargs for protocol compliance."""
    def __init__(self, min_consecutive: int = 3, swap_dir: str = "./artifacts/swaps",
                 unsafe_tools: set = None):
        self.min_consecutive = min_consecutive
        self.swap_dir = swap_dir
        self.unsafe_tools = unsafe_tools if unsafe_tools is not None else UNSAFE_TO_COLLAPSE_TOOLS
        if self.swap_dir and not os.path.exists(self.swap_dir):
            os.makedirs(self.swap_dir, exist_ok=True)

    def _is_collapsible_candidate(self, msg) -> bool:
        """Check if a message is part of a potentially collapsible tool sequence."""
        if isinstance(msg, ToolMessage):
            return True
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            return True
        return False

    def _group_is_safe(self, group: list) -> bool:
        """Check that NO tools in the group are in the unsafe blocklist."""
        for m in group:
            if isinstance(m, ToolMessage):
                tool_name = getattr(m, "name", None) or ""
                if tool_name in self.unsafe_tools:
                    return False
            elif isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    tc_name = tc.get("name", "")
                    if tc_name in self.unsafe_tools:
                        return False
        return True

    def compact(self, messages: list) -> tuple[list, bool]:
        new_messages = []
        i = 0
        n = len(messages)
        modified = False

        while i < n:
            msg = messages[i]
            if self._is_collapsible_candidate(msg):
                group = []
                while i < n and self._is_collapsible_candidate(messages[i]):
                    group.append(messages[i])
                    i += 1
                
                # Only collapse if group is large enough AND no unsafe tools present
                if len(group) >= self.min_consecutive and self._group_is_safe(group):
                    tool_names = set()
                    tool_call_ids = []
                    raw_lines = []
                    for m in group:
                        if isinstance(m, ToolMessage) and getattr(m, "name", None):
                            tool_names.add(m.name)
                            tool_call_ids.append(m.tool_call_id)
                            raw_lines.append(f"[ToolResult:{m.name}] {m.content}")
                        elif isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                            for tc in m.tool_calls:
                                tool_names.add(tc.get("name", "tool"))
                                if tc.get("id"):
                                    tool_call_ids.append(tc["id"])
                            raw_lines.append(f"[AIThought] {m.content}")
                    
                    tools_str = ", ".join(sorted(tool_names)) if tool_names else "exploration tools"
                    
                    # Save raw snapshot to disk for 100% restoration if needed by agent
                    snap_full_path = None
                    if self.swap_dir:
                        joined_raw = "\n\n".join(raw_lines)
                        digest = hashlib.sha1(
                            joined_raw.encode("utf-8", errors="ignore")
                        ).hexdigest()[:12]
                        snap_filename = f"collapse_snap_{digest}.txt"
                        snap_full_path = os.path.join(self.swap_dir, snap_filename)
                        try:
                            if not os.path.exists(snap_full_path):
                                with open(snap_full_path, "w", encoding="utf-8") as sf:
                                    sf.write(joined_raw)
                        except Exception:
                            snap_full_path = None

                    hint_str = ""
                    if snap_full_path:
                        hint_str = (
                            f"\n💡 Actionable Hint: Full intermediate research logs are preserved at '{snap_full_path}'. "
                            f"If you need to recall past investigation details, use `read_file(path='{snap_full_path}')` or `grep_search`."
                        )

                    collapse_msg = SystemMessage(
                        content=(
                            f"[Context Collapsed: {len(group)} research steps ({tools_str}) performed in workspace]"
                            + hint_str
                        ),
                        additional_kwargs={"collapsed_tool_call_ids": tool_call_ids},
                    )
                    new_messages.append(collapse_msg)
                    modified = True
                else:
                    new_messages.extend(group)
            else:
                new_messages.append(msg)
                i += 1

        return new_messages, modified


# =============================================================================
# 4. AutoCompactor (Proactive Token Threshold Summary + Amnesia Guard)
# =============================================================================
def build_compaction_summary_message(summary_text: str) -> HumanMessage:
    """이전 대화 요약을 HumanMessage로 만듭니다.

    모델 제공사별로 대화 중간의 SystemMessage 처리가 달라서(오류 / 시스템 지침으로 이동) 요약은
    대화 메시지로 넣습니다. 사용자의 새 요청으로 오인하지 않도록 머리말과 출처 표식을 붙입니다.
    """
    return HumanMessage(
        content=f"{COMPACTION_SUMMARY_HEADER}\n{AUTO_GENERATED_NOTE}\n\n{summary_text}",
        additional_kwargs=origin_kwargs(ORIGIN_COMPACTION_SUMMARY),
    )


class AutoCompactor:
    """Proactively summarizes history when token count exceeds threshold.
    Preserves only original L1-L5 system prompts at the top.
    Intermediate SystemMessages (Collapse artifacts, etc.) are included in the summary."""

    # Prefixes used by pipeline-generated SystemMessages (not original L1-L5)
    MAX_CONSECUTIVE_FAILURES = 3

    # Prefixes used by pipeline-generated SystemMessages (not original L1-L5)
    _PIPELINE_PREFIXES = (
        "[Context Collapsed:",
        "[Compaction Amnesia Guard:",
        "[Reactive Compact",
        "Previous Conversation Summary:",
    )

    def __init__(self, llm, threshold_tokens: int = 8000, amnesia_guard=None,
                 max_consecutive_failures: int = 3):
        self.llm = llm
        self.threshold_tokens = threshold_tokens
        self.amnesia_guard = amnesia_guard
        self.max_consecutive_failures = max_consecutive_failures
        self.consecutive_failures = 0

    @property
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (tripped)."""
        return self.consecutive_failures >= self.max_consecutive_failures

    def reset_circuit_breaker(self):
        """Manually reset circuit breaker failure counter."""
        self.consecutive_failures = 0

    def get_token_count(self, messages: list) -> int:
        try:
            return self.llm.get_num_tokens_from_messages(messages)
        except Exception:
            return sum(len(str(m.content)) for m in messages) // 4

    def _is_original_system_msg(self, msg) -> bool:
        """Check if a SystemMessage is an original L1-L5 system prompt (not a pipeline artifact)."""
        if not isinstance(msg, SystemMessage):
            return False
        content = str(msg.content)
        for prefix in self._PIPELINE_PREFIXES:
            if content.startswith(prefix):
                return False
        return True

    @staticmethod
    def has_completed_turns(messages: list) -> bool:
        """현재 사용자 턴 앞에 요약할 이전 턴(압축 산출물이 아닌 메시지)이 있는지 확인합니다."""
        dialogue = [m for m in messages if not isinstance(m, SystemMessage)]
        idx = last_user_turn_start(dialogue)
        if not idx:
            return False
        return any(not is_compaction_artifact(m) for m in dialogue[:idx])

    @staticmethod
    def _split_dialogue_for_compaction(dialogue_messages: list) -> tuple[list, list]:
        """대화 메시지를 압축 대상(과거 턴)과 보존 대상(현재 진행 중인 마지막 턴/페어)으로 안전하게 분리합니다."""
        # 턴 경계는 실제 사용자 메시지만 (요약·복원 블록·StopHooks 피드백 HumanMessage는 경계가 아님)
        last_human_idx = last_user_turn_start(dialogue_messages)

        # 마지막 유저 턴 이전에 과거 대화가 존재하면, 마지막 유저 턴부터 끝까지(진행 중인 tool loop 포함) 전부 보존
        if last_human_idx is not None and last_human_idx > 0:
            return dialogue_messages[:last_human_idx], dialogue_messages[last_human_idx:]

        # 단일 턴 내 메시지인 경우: 마지막 메시지가 ToolMessage이면 직전 AIMessage(tool_calls)까지 함께 보존
        if isinstance(dialogue_messages[-1], ToolMessage):
            idx = len(dialogue_messages) - 1
            while idx > 0 and isinstance(dialogue_messages[idx], ToolMessage):
                idx -= 1
            if idx > 0 and isinstance(dialogue_messages[idx], AIMessage):
                return dialogue_messages[:idx], dialogue_messages[idx:]

        return dialogue_messages[:-1], [dialogue_messages[-1]]

    @staticmethod
    def _build_summary_prompt(to_compact: list, max_chars_per_message: int = 4000) -> str:
        history_text = ""
        for m in to_compact:
            origin = message_origin(m)
            if origin == ORIGIN_COMPACTION_SUMMARY:
                role = "PREVIOUS_SUMMARY"
            elif origin == ORIGIN_AMNESIA_RESTORE:
                role = "RESTORED_CONTEXT"
            elif isinstance(m, SystemMessage):
                role = "SYSTEM_NOTE"
            elif m.type == "ai":
                role = "AI"
            elif m.type == "human":
                role = "User"
            else:
                role = m.type.upper()
            content_str = str(m.content or "")
            # 원본 state를 요약하므로 대형 도구 출력이 요약 프롬프트를 키우지 않도록 메시지당 길이를 제한
            if len(content_str) > max_chars_per_message:
                omitted = len(content_str) - max_chars_per_message
                content_str = content_str[:max_chars_per_message] + f" ... [{omitted} chars omitted]"
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                tc_names = ", ".join(tc.get("name", "tool") for tc in m.tool_calls if isinstance(tc, dict))
                if tc_names:
                    content_str = f"{content_str} [ToolCalls: {tc_names}]".strip()
            history_text += f"{role}: {content_str}\n"

        return (
            "You are a context compaction engine. Please summarize the following conversation history.\n"
            "Structure your summary into 4 clear sections:\n"
            "1. Primary Goal & Intent\n"
            "2. Key Decisions & Architecture\n"
            "3. Code Modifications & Workspace Delta\n"
            "4. Current Task & Next Action\n\n"
            f"Conversation History:\n{history_text}"
        )

    def _finalize_compacted_messages(
        self,
        original_system_messages: list,
        summary_text: str,
        tail_messages: list,
    ) -> list:
        summary_msg = build_compaction_summary_message(summary_text)

        recovery_attachments = []
        if self.amnesia_guard:
            for att in self.amnesia_guard.create_recovery_attachments():
                ak = dict(getattr(att, "additional_kwargs", None) or {})
                for key, value in origin_kwargs(ORIGIN_AMNESIA_RESTORE).items():
                    ak.setdefault(key, value)
                att.additional_kwargs = ak
                recovery_attachments.append(att)

        return original_system_messages + [summary_msg] + recovery_attachments + list(tail_messages)

    def _plan_compaction(self, messages: list, force: bool) -> Optional[tuple[list, list, list]]:
        """압축이 필요하면 (원본 시스템 메시지, 압축 대상, 보존 대상)을, 아니면 None을 반환합니다."""
        # 1. Circuit breaker check: halt repeated failing compactions
        if self.is_circuit_open:
            print(f"⚠️ [AutoCompactor Circuit Breaker ACTIVE] {self.consecutive_failures} consecutive failures. Skipping compaction to prevent infinite loop.")
            return None

        tokens = self.get_token_count(messages)
        if not force and tokens <= self.threshold_tokens:
            return None

        # Separate only original L1-L5 system messages; everything else goes into dialogue
        original_system_messages = [m for m in messages if self._is_original_system_msg(m)]
        dialogue_messages = [m for m in messages if not self._is_original_system_msg(m)]

        if len(dialogue_messages) <= 1:
            return None

        to_compact, tail_messages = self._split_dialogue_for_compaction(dialogue_messages)
        if not to_compact:
            return None
        return original_system_messages, to_compact, tail_messages

    def compact_if_needed(self, messages: list, force: bool = False) -> tuple[list, bool]:
        compacted, was_compacted, _ = self.compact_with_details(messages, force=force)
        return compacted, was_compacted

    async def acompact_if_needed(self, messages: list, force: bool = False) -> tuple[list, bool]:
        compacted, was_compacted, _ = await self.acompact_with_details(messages, force=force)
        return compacted, was_compacted

    def compact_with_details(self, messages: list, force: bool = False) -> tuple[list, bool, list]:
        """compact_if_needed와 같지만 요약된 원본 메시지 목록도 함께 반환합니다 (압축 이벤트 기록용)."""
        plan = self._plan_compaction(messages, force)
        if plan is None:
            return messages, False, []
        original_system_messages, to_compact, tail_messages = plan

        summary_prompt = self._build_summary_prompt(to_compact)

        try:
            from app.utils.message_utils import normalize_content
            summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            summary_text = normalize_content(getattr(summary_response, "content", summary_response))
            # Compaction succeeded -> reset circuit breaker
            self.consecutive_failures = 0
        except Exception as e:
            # Compaction failed -> increment circuit breaker
            self.consecutive_failures += 1
            print(f"⚠️ [AutoCompactor Failure] LLM summary invocation failed ({self.consecutive_failures}/{self.max_consecutive_failures}): {e}")
            if self.is_circuit_open:
                print(f"🚨 [AutoCompactor Circuit Tripped] Maximum consecutive failures ({self.max_consecutive_failures}) reached. Circuit opened.")
            summary_text = f"Error generating summary: {e}. Raw message count: {len(to_compact)}"

        compacted = self._finalize_compacted_messages(
            original_system_messages, summary_text, tail_messages
        )
        return compacted, True, to_compact

    async def acompact_with_details(self, messages: list, force: bool = False) -> tuple[list, bool, list]:
        plan = self._plan_compaction(messages, force)
        if plan is None:
            return messages, False, []
        original_system_messages, to_compact, tail_messages = plan

        summary_prompt = self._build_summary_prompt(to_compact)

        try:
            from app.utils.message_utils import normalize_content
            if hasattr(self.llm, "ainvoke"):
                summary_response = await self.llm.ainvoke([HumanMessage(content=summary_prompt)])
            else:
                summary_response = await asyncio.to_thread(
                    self.llm.invoke, [HumanMessage(content=summary_prompt)]
                )
            summary_text = normalize_content(getattr(summary_response, "content", summary_response))
            self.consecutive_failures = 0
        except Exception as e:
            self.consecutive_failures += 1
            print(f"⚠️ [AutoCompactor Failure] LLM summary invocation failed ({self.consecutive_failures}/{self.max_consecutive_failures}): {e}")
            if self.is_circuit_open:
                print(f"🚨 [AutoCompactor Circuit Tripped] Maximum consecutive failures ({self.max_consecutive_failures}) reached. Circuit opened.")
            summary_text = f"Error generating summary: {e}. Raw message count: {len(to_compact)}"

        compacted = self._finalize_compacted_messages(
            original_system_messages, summary_text, tail_messages
        )
        return compacted, True, to_compact


# =============================================================================
# 5. ReactiveCompactor (Turn-Boundary Slicing on 413 Error)
# =============================================================================
class ReactiveCompactor:
    """Emergency post-invocation firewall that catches 413 / overflow errors.
    Slices oldest ~20% of dialogue at HumanMessage turn boundaries to preserve AI-Tool pairs."""
    def __init__(self, slice_ratio: float = 0.20, amnesia_guard=None):
        self.slice_ratio = slice_ratio
        self.amnesia_guard = amnesia_guard

    def handle_overflow(self, messages: list) -> list:
        system_messages = [m for m in messages if isinstance(m, SystemMessage)]
        # 압축 요약은 잘라내지 않고 맨 앞에 고정 (복원 블록은 새로 만들 수 있으면 새 것으로 대체)
        pinned_summaries = [m for m in messages if not isinstance(m, SystemMessage) and is_compaction_summary(m)]
        pinned_restores = [m for m in messages
                           if not isinstance(m, SystemMessage) and is_compaction_artifact(m) and not is_compaction_summary(m)]
        dialogue_messages = [m for m in messages if not isinstance(m, SystemMessage) and not is_compaction_artifact(m)]

        if len(dialogue_messages) <= 2:
            if dialogue_messages:
                last_msg = dialogue_messages[-1]
                truncated_content = str(last_msg.content)[:1000] + "\n... [Reactive Compact: Truncated 413 payload]"
                return system_messages + pinned_summaries + [HumanMessage(content=truncated_content)]
            return messages

        # 실제 사용자 턴 경계에서만 자름 (요약·복원 블록·StopHooks 피드백은 경계가 아님)
        turn_boundaries = user_turn_starts(dialogue_messages)

        if len(turn_boundaries) <= 1:
            # Only one or no turn boundary — fall back to simple ratio-based slicing
            slice_count = max(2, int(len(dialogue_messages) * self.slice_ratio))
            sliced_dialogue = dialogue_messages[slice_count:]
            actual_sliced_count = slice_count
        else:
            # Calculate how many turns to slice (at least 1 turn, ~20% of turns)
            total_turns = len(turn_boundaries)
            turns_to_slice = max(1, int(total_turns * self.slice_ratio))
            if turns_to_slice >= total_turns:
                turns_to_slice = total_turns - 1  # Keep at least the last turn

            slice_at_idx = turn_boundaries[turns_to_slice]
            sliced_dialogue = dialogue_messages[slice_at_idx:]
            actual_sliced_count = slice_at_idx

        reactive_summary = SystemMessage(
            content=f"[Reactive Compact (Silent Withholding)]: Emergency sliced oldest {actual_sliced_count} dialogue messages after API 413 overflow."
        )

        recovery_attachments = []
        if self.amnesia_guard:
            recovery_attachments = self.amnesia_guard.create_recovery_attachments()
        if not recovery_attachments:
            recovery_attachments = pinned_restores

        return system_messages + [reactive_summary] + pinned_summaries + recovery_attachments + sliced_dialogue


# =============================================================================
# 6. CompactorMiddleware (AgentMiddleware class with sync + async support)
# =============================================================================
class CompactorMiddleware(AgentMiddleware):
    """LangChain AgentMiddleware implementing 5-Stage Compactor Pipeline.

    - before_model / abefore_model:
        Snip/Micro/Collapse 적용 후에도 토큰이 threshold_tokens를 초과하면
        AutoCompactor를 실행하여 checkpointer state["messages"]를 압축 요약본으로 영구 교체합니다.
        · 요약은 원본 state 기준으로 "이전 턴"만 대상으로 합니다. 현재 턴(질문·도구 호출·결과)은
          원본 그대로 남기고, 이전 턴이 없으면(이미 요약 + 현재 턴뿐이면) 압축하지 않습니다.
          → 같은 턴 안에서 요약이 반복되지 않습니다.
        (전체 원문은 MemoryMiddleware가 transcript.db에 별도 보존합니다.)
    - wrap_model_call / awrap_model_call:
        매 모델 호출마다 비파괴적(ephemeral) Snip → Micro → Collapse를 사본에만 수행하고,
        413/컨텍스트 초과 에러 발생 시 ReactiveCompactor로 즉시 슬라이싱 재시도합니다.
    """

    def __init__(self, llm, threshold_tokens: int = 8000, amnesia_guard=None,
                 swap_dir: str = "./artifacts/swaps", max_consecutive_failures: int = 3,
                 transcript_store: Optional[Any] = None):
        self.snip_compactor = SnipCompactor(age_threshold=2)
        self.micro_compactor = MicroCompactor(max_chars=5000, swap_dir=swap_dir)
        self.context_collapse = ContextCollapse(min_consecutive=3, swap_dir=swap_dir)
        self.auto_compactor = AutoCompactor(
            llm=llm,
            threshold_tokens=threshold_tokens,
            amnesia_guard=amnesia_guard,
            max_consecutive_failures=max_consecutive_failures
        )
        self.reactive_compactor = ReactiveCompactor(slice_ratio=0.20, amnesia_guard=amnesia_guard)
        self.transcript_store = transcript_store

    def _run_ephemeral_pipeline(self, messages: list) -> list:
        """Execute non-destructive Stage 1~3 compaction: Snip → Micro → Collapse."""
        msgs = list(messages)
        msgs, _ = self.snip_compactor.compact(msgs)
        msgs, _ = self.micro_compactor.compact(msgs)
        msgs, _ = self.context_collapse.compact(msgs)
        return msgs

    def _needs_auto_compact(self, messages: list) -> bool:
        """모델이 실제로 보게 될 크기(Stage 1~3 적용 후)가 임계치를 넘고, 요약할 이전 턴이 있는지."""
        if not messages or self.auto_compactor.is_circuit_open:
            return False
        view = self._run_ephemeral_pipeline(messages)
        if self.auto_compactor.get_token_count(view) <= self.auto_compactor.threshold_tokens:
            return False
        return self.auto_compactor.has_completed_turns(messages)

    def _record_compaction_to_transcript(
        self, runtime: Any, compacted_msgs: list, replaced_count: int,
        compacted_source: Optional[list] = None,
    ) -> None:
        if self.transcript_store is None:
            return
        ctx = getattr(runtime, "context", None) if runtime is not None else None
        if ctx is not None and not getattr(ctx, "transcript_enabled", True):
            return
        session_id = getattr(ctx, "session_id", "unknown") if ctx else "unknown"
        user_id = getattr(ctx, "user_id", "default") if ctx else "default"
        if session_id == "unknown" and runtime is not None:
            cfg = getattr(runtime, "config", None)
            if isinstance(cfg, dict):
                session_id = str((cfg.get("configurable") or {}).get("thread_id") or "unknown")
        if session_id == "unknown":
            return
        summary_text = next((str(m.content) for m in compacted_msgs if is_compaction_summary(m)), "")
        compacted_uids = [
            str(m.id) for m in (compacted_source or [])
            if getattr(m, "id", None) and not is_compaction_artifact(m)
        ]
        if summary_text:
            try:
                self.transcript_store.record_compaction(
                    session_id=session_id,
                    user_id=user_id,
                    summary=summary_text,
                    compacted_uids=compacted_uids,
                    replaced_count=replaced_count,
                )
            except Exception:
                pass

    def before_model(self, state: Dict[str, Any], runtime: Any = None) -> Optional[Dict[str, Any]]:
        """토큰 임계치 초과 시 이전 턴을 요약해 checkpointer state['messages']를 압축본으로 교체합니다."""
        messages = list(state.get("messages", []))
        if not self._needs_auto_compact(messages):
            return None

        # Stage 1~3은 모델 입력 사본용이므로, state 교체는 원본 메시지로 합니다 (현재 턴 원본 보존)
        compacted_msgs, was_compacted, compacted_source = self.auto_compactor.compact_with_details(
            messages, force=True
        )
        if not was_compacted:
            return None

        self._record_compaction_to_transcript(runtime, compacted_msgs, len(messages), compacted_source)
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *compacted_msgs]}

    async def abefore_model(self, state: Dict[str, Any], runtime: Any = None) -> Optional[Dict[str, Any]]:
        """비동기 실행 경로용 AutoCompactor state 교체."""
        messages = list(state.get("messages", []))
        if not self._needs_auto_compact(messages):
            return None

        compacted_msgs, was_compacted, compacted_source = await self.auto_compactor.acompact_with_details(
            messages, force=True
        )
        if not was_compacted:
            return None

        await asyncio.to_thread(
            self._record_compaction_to_transcript, runtime, compacted_msgs, len(messages), compacted_source
        )
        return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *compacted_msgs]}

    def _is_overflow_error(self, error: Exception) -> bool:
        """Check if an exception is a context overflow / 413 error."""
        err_str = str(error).lower()
        return any(pattern in err_str for pattern in [
            "413", "prompt_too_long", "context_length_exceeded", "too many tokens", "maximum context length"
        ])

    def wrap_model_call(self, request, handler):
        msgs = self._run_ephemeral_pipeline(request.messages)
        req = request.override(messages=msgs)
        try:
            return handler(req)
        except Exception as e:
            if self._is_overflow_error(e):
                print("\n⚡ [Reactive Compact Triggered] Catching 413 API Error. Performing Turn-Boundary Slicing...")
                reactive_msgs = self.reactive_compactor.handle_overflow(msgs)
                retry_req = request.override(messages=reactive_msgs)
                return handler(retry_req)
            raise e

    async def awrap_model_call(self, request, handler):
        msgs = self._run_ephemeral_pipeline(request.messages)
        req = request.override(messages=msgs)
        try:
            return await handler(req)
        except Exception as e:
            if self._is_overflow_error(e):
                print("\n⚡ [Reactive Compact Triggered] Catching 413 API Error. Performing Turn-Boundary Slicing...")
                reactive_msgs = self.reactive_compactor.handle_overflow(msgs)
                retry_req = request.override(messages=reactive_msgs)
                return await handler(retry_req)
            raise e


def create_compactor_middleware(llm, threshold_tokens: int = 8000, amnesia_guard=None,
                                swap_dir: str = "./artifacts/swaps", max_consecutive_failures: int = 3,
                                transcript_store: Optional[Any] = None):
    """Creates a CompactorMiddleware instance (AgentMiddleware subclass).
    This is the recommended way to create the compactor middleware."""
    return CompactorMiddleware(
        llm=llm,
        threshold_tokens=threshold_tokens,
        amnesia_guard=amnesia_guard,
        swap_dir=swap_dir,
        max_consecutive_failures=max_consecutive_failures,
        transcript_store=transcript_store,
    )
