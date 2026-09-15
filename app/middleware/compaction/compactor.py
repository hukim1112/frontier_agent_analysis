"""
===============================================================================
[H-01] 5-Stage Context Compactor Pipeline
===============================================================================
Source: frontier-agent-lab/modules/claude_code/compactor.py

Claude Code 아키텍처의 5단계 컨텍스트 압축 엔진:
1. SnipCompactor      : Age 기반 오래된 ToolMessage 1줄 스텁 대체
2. MicroCompactor     : 대형 출력(>5000 chars) 디스크 스왑 (idempotency guard 포함)
3. ContextCollapse    : 연속 탐색 스텝(3+) 그룹 접기 (변경성 도구 자동 제외)
4. AutoCompactor      : 토큰 임계치 초과 시 LLM 요약 (+ AmnesiaGuard 연동)
5. ReactiveCompactor  : 413 에러 대응 턴 경계(Turn-Boundary) 기반 슬라이싱

통합 미들웨어: CompactorMiddleware(AgentMiddleware) + create_compactor_middleware()
- wrap_model_call / awrap_model_call 로 Phase 1 (pre-call) + Phase 2 (reactive) 실행
===============================================================================
"""

import os
import re
import json
import time
import uuid
from typing import List, Tuple, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain.agents.middleware import AgentMiddleware


# =============================================================================
# Blocklist: Tools that MUST NOT be collapsed (write/modify/execute operations)
# =============================================================================
UNSAFE_TO_COLLAPSE_TOOLS = {
    # File write/edit operations
    "write_file", "create_file", "edit_file", "replace_file_content",
    "multi_replace_file_content", "write_to_file", "patch_file",
    # Command execution
    "run_command", "execute", "exec", "shell", "bash", "terminal",
    # File delete operations
    "delete_file", "remove", "rm", "rmdir",
    # Plan management
    "update_plan", "create_plan", "set_plan", "enter_plan",
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
        # 턴(Turn) 구분: HumanMessage가 나타날 때마다 새 턴으로 간주
        human_indices = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
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
                swap_filename = f"swap_{uuid.uuid4().hex[:8]}.txt"
                swap_path = os.path.join(self.swap_dir, swap_filename)
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
                        snap_filename = f"collapse_snap_{int(time.time()*1000)}.txt"
                        snap_full_path = os.path.join(self.swap_dir, snap_filename)
                        try:
                            with open(snap_full_path, "w", encoding="utf-8") as sf:
                                sf.write("\n\n".join(raw_lines))
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

    def compact_if_needed(self, messages: list, force: bool = False) -> tuple[list, bool]:
        # 1. Circuit breaker check: halt repeated failing compactions
        if self.is_circuit_open:
            print(f"⚠️ [AutoCompactor Circuit Breaker ACTIVE] {self.consecutive_failures} consecutive failures. Skipping compaction to prevent infinite loop.")
            return messages, False

        tokens = self.get_token_count(messages)
        if not force and tokens <= self.threshold_tokens:
            return messages, False

        # Separate only original L1-L5 system messages; everything else goes into dialogue
        original_system_messages = [m for m in messages if self._is_original_system_msg(m)]
        dialogue_messages = [m for m in messages if not self._is_original_system_msg(m)]

        if len(dialogue_messages) <= 1:
            return messages, False

        to_compact = dialogue_messages[:-1]
        last_message = dialogue_messages[-1]

        history_text = ""
        for m in to_compact:
            if isinstance(m, SystemMessage):
                role = "SYSTEM_NOTE"
            elif m.type == "ai":
                role = "AI"
            elif m.type == "human":
                role = "User"
            else:
                role = m.type.upper()
            history_text += f"{role}: {m.content}\n"

        summary_prompt = (
            "You are a context compaction engine. Please summarize the following conversation history.\n"
            "Structure your summary into 4 clear sections:\n"
            "1. Primary Goal & Intent\n"
            "2. Key Decisions & Architecture\n"
            "3. Code Modifications & Workspace Delta\n"
            "4. Current Task & Next Action\n\n"
            f"Conversation History:\n{history_text}"
        )
        
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

        summary_msg = SystemMessage(content=f"Previous Conversation Summary:\n{summary_text}")

        recovery_attachments = []
        if self.amnesia_guard:
            recovery_attachments = self.amnesia_guard.create_recovery_attachments()

        compacted = original_system_messages + [summary_msg] + recovery_attachments + [last_message]
        return compacted, True


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
        dialogue_messages = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(dialogue_messages) <= 2:
            if dialogue_messages:
                last_msg = dialogue_messages[-1]
                truncated_content = str(last_msg.content)[:1000] + "\n... [Reactive Compact: Truncated 413 payload]"
                return system_messages + [HumanMessage(content=truncated_content)]
            return messages

        # Find HumanMessage turn boundaries for safe slicing
        turn_boundaries = [i for i, m in enumerate(dialogue_messages) if isinstance(m, HumanMessage)]

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

        return system_messages + [reactive_summary] + recovery_attachments + sliced_dialogue


# =============================================================================
# 6. CompactorMiddleware (AgentMiddleware class with sync + async support)
# =============================================================================
class CompactorMiddleware(AgentMiddleware):
    """LangChain AgentMiddleware implementing 5-Stage Compactor Pipeline.
    Supports both wrap_model_call (sync) and awrap_model_call (async).
    
    Pipeline order: Snip → Micro → Collapse → Auto (then Reactive on 413)
    """

    def __init__(self, llm, threshold_tokens: int = 8000, amnesia_guard=None,
                 swap_dir: str = "./artifacts/swaps", max_consecutive_failures: int = 3):
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

    def _run_phase1_pipeline(self, messages: list) -> list:
        """Execute Phase 1 pre-call compaction pipeline: Snip → Micro → Collapse → Auto."""
        msgs = list(messages)
        # Stage 1: Snip (Age-based 1-line truncation) — cheap reduction of old outputs first
        msgs, _ = self.snip_compactor.compact(msgs)
        # Stage 2: Micro (Large payload disk swap) — only remaining large recent outputs
        msgs, _ = self.micro_compactor.compact(msgs)
        # Stage 3: Context Collapse (Continuous research folding)
        msgs, _ = self.context_collapse.compact(msgs)
        # Stage 4: Auto-Compact (Token threshold LLM summary)
        msgs, _ = self.auto_compactor.compact_if_needed(msgs)
        return msgs

    def _is_overflow_error(self, error: Exception) -> bool:
        """Check if an exception is a context overflow / 413 error."""
        err_str = str(error).lower()
        return any(pattern in err_str for pattern in [
            "413", "prompt_too_long", "context_length_exceeded", "too many tokens"
        ])

    def wrap_model_call(self, request, handler):
        msgs = self._run_phase1_pipeline(request.messages)
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
        msgs = self._run_phase1_pipeline(request.messages)
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
                                swap_dir: str = "./artifacts/swaps", max_consecutive_failures: int = 3):
    """Creates a CompactorMiddleware instance (AgentMiddleware subclass).
    This is the recommended way to create the compactor middleware."""
    return CompactorMiddleware(
        llm=llm,
        threshold_tokens=threshold_tokens,
        amnesia_guard=amnesia_guard,
        swap_dir=swap_dir,
        max_consecutive_failures=max_consecutive_failures
    )
