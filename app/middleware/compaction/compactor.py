"""
===============================================================================
[H-01] Claude Code Context Compaction Pipeline
===============================================================================
Source: frontier-agent-lab/modules/claude_code/compactor.py

Claude Code(유출 소스 v2.1.88)의 query.ts가 매 모델 호출 직전에 실행하는 순서를 따릅니다.

  applyToolResultBudget → snip → microcompact → contextCollapse → autocompact
  (+ API가 prompt_too_long을 돌려주면 reactive compact)

각 단계의 근거 등급:
  🟢 원본 소스가 남아 있어 그대로 옮김   🟡 원본 파일이 빌드에서 빠져 있어 호출부 흔적으로 재구성

| 단계 | 클래스 | 원본 위치 | 등급 | 이 구현의 훅 |
|---|---|---|---|---|
| 0. Tool Result Budget | ToolResultBudget | utils/toolResultStorage.ts | 🟢 | wrap_tool_call(도구별) + before_model(라운드 합계) |
| 1. Snip | SnipCompactor | services/compact/snipCompact.ts (없음) | 🟡 | snip 도구 + wrap_model_call 투영 |
| 2. Microcompact (time-based) | MicroCompactor | services/compact/microCompact.ts | 🟢 | before_model (같은 id로 state 교체) |
| 3. Context Collapse | ContextCollapse | services/contextCollapse/ (없음) | 🟡 | before_model 커밋 + wrap_model_call 투영 |
| 4. Auto-Compact | AutoCompactor | autoCompact.ts, compact.ts, prompt.ts | 🟢 | before_model (state 교체) |
| 5. Reactive Compact | ReactiveCompactor | reactiveCompact.ts (없음) | 🟡 | wrap_model_call (413 재시도) |

원본과 다르게 구현한 부분:
- Cached microcompact(서버 캐시를 cache_edits로 편집)는 Anthropic 내부 전용 API라 구현하지 않았습니다.
- 원본 Auto-Compact는 캐시를 공유하는 fork 에이전트에게 대화를 그대로 넘깁니다. 여기서는 모델 제공사와
  무관하게 동작하도록 대화를 텍스트로 직렬화해 요약 LLM에 넘깁니다.
- 원본 Auto-Compact는 대화 전체를 요약합니다. 여기서는 현재 사용자 턴을 원본 그대로 남기고 이전 턴만 요약합니다.
- 압축 임계치는 원본 공식(컨텍스트 윈도우 - 20K - 13K) 대신 threshold_tokens로 직접 지정합니다.
  원본 공식은 claude_code_autocompact_threshold()로 계산할 수 있습니다.

요약·접기·표시 메시지는 HumanMessage로 넣고 additional_kwargs["harness_origin"]과 첫 줄 머리말로
"하네스가 만든 메시지"임을 표시합니다. 턴 경계는 message_markers 모듈의 규칙으로 판별합니다.

modules/claude_code/compactor.py는 이 파일의 사본입니다 (message_markers import 경로만 다름).
===============================================================================
"""

import asyncio
import math
import os
import re
import time
import uuid
from typing import Annotated, Any, Dict, List, NotRequired, Optional, Sequence, Tuple

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain.agents.middleware import AgentMiddleware, AgentState

from app.utils.message_markers import (
    AUTO_GENERATED_NOTE,
    COMPACTION_SUMMARY_HEADER,
    CONTEXT_COLLAPSE_HEADER,
    CONTEXT_EFFICIENCY_HEADER,
    ORIGIN_AMNESIA_RESTORE,
    ORIGIN_COMPACTION_MARKER,
    ORIGIN_COMPACTION_SUMMARY,
    ORIGIN_CONTEXT_COLLAPSE,
    ORIGIN_CONTEXT_EFFICIENCY,
    ORIGIN_SNIP_BOUNDARY,
    SNIP_BOUNDARY_HEADER,
    is_compaction_artifact,
    is_compaction_summary,
    is_user_turn_start,
    last_user_turn_start,
    message_origin,
    origin_kwargs,
    user_turn_starts,
)
from .amnesia_guard import resolve_session_key


# =============================================================================
# 원본 상수 (괄호 안은 원본 파일)
# =============================================================================
DEFAULT_MAX_RESULT_SIZE_CHARS = 50_000          # constants/toolLimits.ts
MAX_TOOL_RESULTS_PER_MESSAGE_CHARS = 200_000    # constants/toolLimits.ts
PREVIEW_SIZE_CHARS = 2_000                      # utils/toolResultStorage.ts (PREVIEW_SIZE_BYTES)
PERSISTED_OUTPUT_TAG = "<persisted-output>"     # utils/toolResultStorage.ts
PERSISTED_OUTPUT_CLOSING_TAG = "</persisted-output>"

TIME_BASED_MC_CLEARED_MESSAGE = "[Old tool result content cleared]"   # services/compact/microCompact.ts
TIME_BASED_MC_GAP_MINUTES = 60                  # services/compact/timeBasedMCConfig.ts (서버 캐시 TTL 1시간)
TIME_BASED_MC_KEEP_RECENT = 5                   # services/compact/timeBasedMCConfig.ts

MAX_OUTPUT_TOKENS_FOR_SUMMARY = 20_000          # services/compact/autoCompact.ts
AUTOCOMPACT_BUFFER_TOKENS = 13_000              # services/compact/autoCompact.ts
MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3        # services/compact/autoCompact.ts
MAX_PTL_RETRIES = 3                             # services/compact/compact.ts
PTL_RETRY_MARKER = "[earlier conversation truncated for compaction retry]"   # services/compact/compact.ts

COLLAPSE_COMMIT_RATIO = 0.9                     # autoCompact.ts 주석: "90% commit / 95% blocking-spawn"

# 디스크 저장 대상에서 빼는 도구. 원본 Read 도구의 maxResultSizeChars = Infinity는 '상한 없음'이 아니라
# '이 래퍼를 적용하지 않음' 표시입니다. Read는 스스로 분량을 제한하고(256KB 초과 파일, 25,000토큰 초과 출력은
# offset/limit을 쓰라는 에러로 거절), 저장한 파일을 다시 Read로 읽으면 순환이 생기기 때문입니다.
# 이 프로젝트의 file_read는 줄 수(기본 250줄)로만 제한하므로, 줄이 매우 긴 파일은 큰 결과가 될 수 있습니다.
UNBOUNDED_RESULT_TOOLS = frozenset({"file_read"})

# 원본 COMPACTABLE_TOOLS(Read, Bash, Grep, Glob, WebSearch, WebFetch, Edit, Write)를 이 프로젝트 도구명으로 옮김
COMPACTABLE_TOOLS = frozenset({
    "file_read", "bash_command", "grep_search", "glob_search",
    "web_search", "web_fetch", "file_edit", "file_writer",
})

_OVERFLOW_PATTERNS = (
    "413", "prompt_too_long", "prompt is too long", "context_length_exceeded",
    "too many tokens", "maximum context length", "request too large",
)


def claude_code_autocompact_threshold(context_window: int,
                                      max_output_tokens: int = MAX_OUTPUT_TOKENS_FOR_SUMMARY) -> int:
    """원본 Auto-Compact 발동 지점: 윈도우 - 요약 출력 예약(최대 20K) - 버퍼 13K.

    200K 윈도우라면 200,000 - 20,000 - 13,000 = 167,000 토큰(83.5%)입니다.
    """
    reserved = min(max_output_tokens, MAX_OUTPUT_TOKENS_FOR_SUMMARY)
    return context_window - reserved - AUTOCOMPACT_BUFFER_TOKENS


def is_prompt_too_long_error(error: Exception) -> bool:
    err = str(error).lower()
    return any(p in err for p in _OVERFLOW_PATTERNS)


# =============================================================================
# 공통 도우미
# =============================================================================
def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in content]
        return "\n".join(p for p in parts if p)
    return str(content or "")


_SYSTEM_REMINDER_RE = re.compile(r"^\s*<system-reminder>[\s\S]*?</system-reminder>\s*")


def _leading_system_reminder(msg: Any) -> str:
    """PromptAssembler가 첫 사용자 메시지 앞에 붙인 CLAUDE.md 블록. 그 메시지가 투영에서 빠질 때 옮겨 붙입니다."""
    content = getattr(msg, "content", "")
    match = _SYSTEM_REMINDER_RE.match(content) if isinstance(content, str) else None
    return match.group(0) if match else ""


def rough_token_count(messages: Sequence[Any]) -> int:
    """원본 roughTokenCountEstimation처럼 글자 수/4로 추정하고 4/3배 여유를 둡니다."""
    chars = sum(len(_content_text(getattr(m, "content", ""))) for m in messages)
    return math.ceil(chars / 4 * 4 / 3)


def _format_size(chars: int) -> str:
    return f"{chars / 1024:.1f}KB" if chars >= 1024 else f"{chars} chars"


def _replace_tool_message(msg: ToolMessage, content: str) -> ToolMessage:
    """같은 id를 유지한 채 내용만 바꾼 ToolMessage (state 교체·transcript 멱등 기록에 필요)."""
    return ToolMessage(
        content=content,
        tool_call_id=msg.tool_call_id,
        name=getattr(msg, "name", None),
        id=getattr(msg, "id", None),
        status=getattr(msg, "status", "success"),
    )


def _apply_replacements(messages: List[Any], replacements: List[Any]) -> List[Any]:
    by_id = {m.id: m for m in replacements if getattr(m, "id", None)}
    if not by_id:
        return messages
    return [by_id.get(getattr(m, "id", None), m) for m in messages]


def group_messages_by_api_round(messages: Sequence[Any]) -> List[List[Any]]:
    """원본 groupMessagesByApiRound: 새 AI 응답이 시작될 때마다 그룹을 나눕니다.

    AI 메시지의 tool_calls와 그 ToolMessage들은 항상 같은 그룹에 들어가므로, 그룹 단위로
    잘라내도 도구 호출·결과 쌍이 깨지지 않습니다. 첫 그룹은 첫 AI 응답 앞의 메시지들입니다.
    """
    groups: List[List[Any]] = []
    current: List[Any] = []
    last_ai_id = object()
    for m in messages:
        if isinstance(m, AIMessage):
            ai_id = getattr(m, "id", None) or id(m)
            if ai_id != last_ai_id and current:
                groups.append(current)
                current = []
            last_ai_id = ai_id
        current.append(m)
    if current:
        groups.append(current)
    return groups


def _marker_message(text: str) -> HumanMessage:
    return HumanMessage(content=text, additional_kwargs=origin_kwargs(ORIGIN_COMPACTION_MARKER))


def truncate_head_for_ptl_retry(messages: Sequence[Any], drop_ratio: float = 0.2) -> Optional[List[Any]]:
    """원본 truncateHeadForPTLRetry: 가장 오래된 API 라운드부터 버립니다 (토큰 차이를 모를 때 20%).

    요약할 것이 남도록 최소 1개 그룹은 남기고, 결과가 AI 메시지로 시작하면 표시 메시지를 앞에 붙입니다.
    """
    items = list(messages)
    if items and message_origin(items[0]) == ORIGIN_COMPACTION_MARKER:
        items = items[1:]
    groups = group_messages_by_api_round(items)
    if len(groups) < 2:
        return None
    drop = min(max(1, int(len(groups) * drop_ratio)), len(groups) - 1)
    sliced = [m for g in groups[drop:] for m in g]
    if sliced and isinstance(sliced[0], AIMessage):
        return [_marker_message(PTL_RETRY_MARKER)] + sliced
    return sliced


def _serialize_for_summary(messages: Sequence[Any], max_chars_per_message: int) -> str:
    """요약 LLM에 넘길 대화 텍스트. 대형 도구 출력이 요약 프롬프트를 키우지 않도록 메시지당 길이를 제한합니다."""
    role_by_origin = {
        ORIGIN_COMPACTION_SUMMARY: "PREVIOUS_SUMMARY",
        ORIGIN_AMNESIA_RESTORE: "RESTORED_CONTEXT",
        ORIGIN_CONTEXT_COLLAPSE: "COLLAPSED_CONTEXT",
    }
    lines = []
    for m in messages:
        origin = message_origin(m)
        if origin in (ORIGIN_SNIP_BOUNDARY, ORIGIN_CONTEXT_EFFICIENCY, ORIGIN_COMPACTION_MARKER):
            role = "HARNESS_NOTE"
        elif origin in role_by_origin:
            role = role_by_origin[origin]
        elif isinstance(m, SystemMessage):
            role = "SYSTEM_NOTE"
        elif isinstance(m, AIMessage):
            role = "Assistant"
        elif isinstance(m, ToolMessage):
            role = f"ToolResult({getattr(m, 'name', None) or 'tool'})"
        else:
            role = "User"
        text = _content_text(getattr(m, "content", ""))
        if len(text) > max_chars_per_message:
            omitted = len(text) - max_chars_per_message
            text = text[:max_chars_per_message] + f" ... [{omitted} chars omitted]"
        if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
            names = ", ".join(tc.get("name", "tool") for tc in m.tool_calls if isinstance(tc, dict))
            text = f"{text} [ToolCalls: {names}]".strip()
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


class CompactionState(AgentState):
    """Compactor가 checkpointer state에 남기는 필드."""
    snipped_message_ids: NotRequired[List[str]]     # Snip이 모델 입력에서 뺀 메시지 id
    collapse_commits: NotRequired[List[Dict[str, Any]]]   # ContextCollapse 커밋 로그
    last_assistant_at: NotRequired[float]           # time-based microcompact 기준 시각


# =============================================================================
# 0. ToolResultBudget (utils/toolResultStorage.ts) 🟢
# =============================================================================
class ToolResultBudget:
    """큰 도구 결과를 디스크에 저장하고, 모델에게는 미리보기와 파일 경로만 보여줍니다.

    - 도구별 상한(기본 50,000자)을 넘는 결과는 도구가 끝나는 즉시 저장합니다 (원본 maybePersistLargeToolResult).
    - 한 라운드(병렬 도구 호출 한 번)의 결과 합계가 200,000자를 넘으면, 새 결과 중 큰 것부터 저장해
      예산 아래로 낮춥니다 (원본 enforceToolResultBudget). 한 번 내린 결정은 state에 남아 이후에도 유지됩니다.
    - file_read 결과는 저장하지 않습니다 (UNBOUNDED_RESULT_TOOLS 참고).
    """

    def __init__(self, storage_dir: str = "./artifacts/tool-results",
                 max_result_chars: int = DEFAULT_MAX_RESULT_SIZE_CHARS,
                 per_message_chars: int = MAX_TOOL_RESULTS_PER_MESSAGE_CHARS,
                 preview_chars: int = PREVIEW_SIZE_CHARS,
                 unbounded_tools: Optional[set] = None,
                 tool_limits: Optional[Dict[str, int]] = None):
        self.storage_dir = storage_dir
        self.max_result_chars = max_result_chars
        self.per_message_chars = per_message_chars
        self.preview_chars = preview_chars
        self.unbounded_tools = set(UNBOUNDED_RESULT_TOOLS if unbounded_tools is None else unbounded_tools)
        self.tool_limits = dict(tool_limits or {})

    def threshold_for(self, tool_name: Optional[str]) -> float:
        """원본 getPersistenceThreshold: 도구가 선언한 상한과 전역 상한 중 작은 값."""
        if tool_name in self.unbounded_tools:
            return math.inf
        return min(self.tool_limits.get(tool_name or "", self.max_result_chars), self.max_result_chars)

    def session_dir(self) -> str:
        # 원본 경로 <project>/<sessionId>/tool-results 처럼 세션별로 나눕니다.
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", resolve_session_key())
        return os.path.join(self.storage_dir, safe)

    def generate_preview(self, content: str) -> Tuple[str, bool]:
        """원본 generatePreview: 한도 안에서 마지막 줄바꿈 위치로 자릅니다 (너무 앞이면 한도 그대로)."""
        if len(content) <= self.preview_chars:
            return content, False
        truncated = content[:self.preview_chars]
        last_newline = truncated.rfind("\n")
        cut = last_newline if last_newline > self.preview_chars * 0.5 else self.preview_chars
        return content[:cut], True

    def persist(self, content: str, tool_call_id: str) -> Optional[str]:
        """결과를 파일로 저장하고 모델에게 보낼 대체 문자열을 반환합니다. 실패하면 None (원본 그대로 전달)."""
        directory = self.session_dir()
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", tool_call_id or uuid.uuid4().hex)
        path = os.path.join(directory, f"{safe_id}.txt")
        try:
            os.makedirs(directory, exist_ok=True)
            # tool_call_id마다 내용이 정해져 있으므로 이미 있으면 다시 쓰지 않습니다 (원본 flag 'wx').
            with open(path, "x", encoding="utf-8") as f:
                f.write(content)
        except FileExistsError:
            pass
        except OSError:
            return None
        preview, has_more = self.generate_preview(content)
        return (
            f"{PERSISTED_OUTPUT_TAG}\n"
            f"Output too large ({_format_size(len(content))}). Full output saved to: {path}\n\n"
            f"Preview (first {_format_size(self.preview_chars)}):\n"
            f"{preview}{chr(10) + '...' + chr(10) if has_more else chr(10)}"
            f"{PERSISTED_OUTPUT_CLOSING_TAG}"
        )

    @staticmethod
    def is_persisted(msg: Any) -> bool:
        return isinstance(msg.content, str) and msg.content.startswith(PERSISTED_OUTPUT_TAG)

    def apply_to_tool_message(self, msg: ToolMessage, tool_name: Optional[str] = None) -> ToolMessage:
        """도구별 상한을 넘는 결과를 저장된 미리보기로 바꿉니다."""
        if not isinstance(msg.content, str) or self.is_persisted(msg):
            return msg   # 원본도 텍스트가 아닌 결과(이미지 등)는 저장하지 않습니다
        name = tool_name or getattr(msg, "name", None)
        if len(msg.content) <= self.threshold_for(name):
            return msg
        replacement = self.persist(msg.content, msg.tool_call_id)
        return _replace_tool_message(msg, replacement) if replacement else msg

    def enforce_round_budget(self, messages: Sequence[Any]) -> List[ToolMessage]:
        """마지막 라운드(대화 끝에 이어진 ToolMessage들)의 합계 예산을 적용하고, 교체할 메시지를 반환합니다."""
        tail: List[ToolMessage] = []
        for m in reversed(messages):
            if not isinstance(m, ToolMessage):
                break
            tail.append(m)
        counted = [m for m in tail if getattr(m, "name", None) not in self.unbounded_tools]
        total = sum(len(_content_text(m.content)) for m in counted)
        if total <= self.per_message_chars:
            return []
        fresh = [m for m in counted if isinstance(m.content, str) and not self.is_persisted(m)]
        replacements = []
        for m in sorted(fresh, key=lambda x: len(x.content), reverse=True):
            replacement = self.persist(m.content, m.tool_call_id)
            if replacement is None:
                continue
            total -= len(m.content) - len(replacement)
            replacements.append(_replace_tool_message(m, replacement))
            if total <= self.per_message_chars:
                break
        return replacements


# =============================================================================
# 1. SnipCompactor (services/compact/snipCompact.ts — 빌드에서 빠짐) 🟡
# =============================================================================
class SnipCompactor:
    """모델이 snip 도구로 고른 메시지를 모델 입력에서 뺍니다 (LLM 요약 없음).

    원본 흔적: utils/messages.ts가 사용자 메시지에 [id:...] 태그를 붙여 "모델이 snip 도구로 메시지를
    지목"하게 하고, 컨텍스트가 커지면 SNIP_NUDGE_TEXT로 사용을 권합니다. 화면(REPL)에는 메시지가 남고
    API로 보내는 뷰(projectSnippedView)에서만 빠집니다. 현재 턴(protected tail)은 snip되지 않습니다.
    도구 인자, 태그 형식, 권유 문구는 원본 파일이 없어 이 구현에서 정한 것입니다.
    """

    TAG_PATTERN = re.compile(r"\[id:([A-Za-z0-9]{8})\]")

    def __init__(self, nudge_ratio: float = 0.7):
        self.nudge_ratio = nudge_ratio

    @staticmethod
    def short_id(message_id: Optional[str]) -> Optional[str]:
        if not message_id:
            return None
        compact = re.sub(r"[^A-Za-z0-9]", "", str(message_id))
        return compact[-8:] if len(compact) >= 8 else None

    def tag_messages(self, messages: Sequence[Any]) -> List[Any]:
        """실제 사용자 메시지와 도구 결과 끝에 [id:xxxxxxxx] 태그를 붙인 사본을 만듭니다."""
        out = []
        for m in messages:
            sid = self.short_id(getattr(m, "id", None))
            taggable = isinstance(m, ToolMessage) or is_user_turn_start(m)
            if not sid or not taggable:
                out.append(m)
                continue
            tag = f"[id:{sid}]"
            if isinstance(m.content, str):
                new_content: Any = f"{m.content}\n{tag}"
            elif isinstance(m.content, list):
                new_content = list(m.content) + [{"type": "text", "text": tag}]
            else:
                out.append(m)
                continue
            out.append(m.model_copy(update={"content": new_content}))
        return out

    def plan_snip(self, messages: Sequence[Any], start_id: str,
                  end_id: Optional[str] = None) -> Tuple[List[str], Optional[str]]:
        """제거할 메시지 id 목록을 계산합니다. 문제가 있으면 ([], 오류 문구)를 반환합니다.

        - 도구 호출(AIMessage)과 그 결과(ToolMessage)는 함께 지웁니다 (API 쌍 규칙).
        - 현재 사용자 턴부터는 지우지 않습니다.
        - 시스템 메시지와 압축 요약·복원 블록은 남깁니다.
        """
        index_by_short = {}
        for i, m in enumerate(messages):
            sid = self.short_id(getattr(m, "id", None))
            if sid:
                index_by_short[sid] = i
        start_key = (start_id or "").strip().removeprefix("[id:").removesuffix("]")
        end_key = (end_id or start_key).strip().removeprefix("[id:").removesuffix("]")
        if start_key not in index_by_short or end_key not in index_by_short:
            return [], f"Unknown message id. Use the 8-character ids shown in [id:...] tags."
        i, j = sorted((index_by_short[start_key], index_by_short[end_key]))

        protected = last_user_turn_start(messages)
        protected = len(messages) if protected is None else protected
        if i >= protected:
            return [], "The current user turn cannot be snipped."
        j = min(j, protected - 1)

        while i > 0 and isinstance(messages[i], ToolMessage):
            i -= 1
        while j + 1 < protected and isinstance(messages[j + 1], ToolMessage):
            j += 1

        ids = [
            m.id for m in messages[i:j + 1]
            if getattr(m, "id", None)
            and not isinstance(m, SystemMessage)
            and not is_compaction_artifact(m)
        ]
        if not ids:
            return [], "Nothing to snip in that range."
        return ids, None

    def project(self, messages: Sequence[Any], snipped_ids: Sequence[str]) -> List[Any]:
        """snip된 메시지를 빼고, 빠진 자리마다 경계 표시 메시지를 하나 넣습니다."""
        snipped = set(snipped_ids or [])
        if not snipped:
            return list(messages)
        out: List[Any] = []
        run, carried = 0, ""
        for m in messages:
            if getattr(m, "id", None) in snipped:
                run += 1
                carried += _leading_system_reminder(m)
                continue
            if run:
                out.append(self._boundary(run, carried))
                run, carried = 0, ""
            out.append(m)
        if run:
            out.append(self._boundary(run, carried))
        return out

    @staticmethod
    def _boundary(count: int, prefix: str = "") -> HumanMessage:
        return HumanMessage(
            content=(f"{prefix}{SNIP_BOUNDARY_HEADER} {count} earlier message(s) were removed from context "
                     f"with the snip tool.\n{AUTO_GENERATED_NOTE}"),
            additional_kwargs=origin_kwargs(ORIGIN_SNIP_BOUNDARY),
        )

    def nudge(self, used_tokens: int, limit_tokens: int) -> Optional[HumanMessage]:
        """컨텍스트가 임계치의 nudge_ratio를 넘으면 snip 사용을 권하는 알림을 만듭니다."""
        if limit_tokens <= 0 or used_tokens < limit_tokens * self.nudge_ratio:
            return None
        pct = int(used_tokens * 100 / limit_tokens)
        return HumanMessage(
            content=(
                f"{CONTEXT_EFFICIENCY_HEADER}\n{AUTO_GENERATED_NOTE}\n"
                f"Context usage is about {pct}% of the compaction threshold "
                f"(~{used_tokens:,} / {limit_tokens:,} tokens). If earlier messages are no longer needed, "
                "call the `snip` tool with their [id:...] tags to remove them before older context "
                "is summarized automatically."
            ),
            additional_kwargs=origin_kwargs(ORIGIN_CONTEXT_EFFICIENCY),
        )

    def build_tool(self):
        compactor = self

        @tool("snip")
        def snip_tool(
            start_id: str,
            tool_call_id: Annotated[str, InjectedToolCallId],
            state: Annotated[dict, InjectedState],
            end_id: str = "",
            reason: str = "",
        ) -> Command:
            """Remove earlier messages from your context window to free space.

            Messages in your context carry tags like [id:ab12cd34]. Pass the id of the first message
            to remove as start_id and, for a range, the id of the last one as end_id. Tool calls and
            their results are removed together, and the current user turn cannot be removed.
            Snipped messages stay in the session log but you will no longer see them, so only snip
            content you are sure you no longer need (old file dumps, finished searches, long logs).
            """
            messages = list(state.get("messages", []))
            ids, error = compactor.plan_snip(messages, start_id, end_id or None)
            if error:
                return Command(update={"messages": [
                    ToolMessage(content=f"Snip failed: {error}", tool_call_id=tool_call_id,
                                name="snip", status="error"),
                ]})
            previous = list(state.get("snipped_message_ids") or [])
            new_ids = [i for i in ids if i not in set(previous)]
            removed = [m for m in messages if getattr(m, "id", None) in set(new_ids)]
            freed = rough_token_count(removed)
            note = f" Reason: {reason}" if reason else ""
            return Command(update={
                "snipped_message_ids": previous + new_ids,
                "messages": [ToolMessage(
                    content=f"Snipped {len(new_ids)} message(s), about {freed:,} tokens freed.{note}",
                    tool_call_id=tool_call_id, name="snip",
                )],
            })

        return snip_tool


# =============================================================================
# 2. MicroCompactor — time-based (services/compact/microCompact.ts) 🟢
# =============================================================================
class MicroCompactor:
    """마지막 AI 응답 이후 오래(기본 60분) 지났다면, 오래된 도구 결과의 본문을 비웁니다.

    서버의 프롬프트 캐시(TTL 1시간)가 이미 만료되어 어차피 전체를 다시 보내야 하므로, 보내기 전에
    줄이는 것이 이득이라는 판단입니다. 최근 keep_recent개의 결과는 남깁니다. LLM 호출은 없습니다.
    원본의 기본값은 enabled=false이고 원격 설정(GrowthBook)으로 켭니다.
    원본에는 서버 캐시를 직접 편집하는 cached microcompact 경로도 있지만 내부 전용 API라 구현하지 않습니다.
    """

    def __init__(self, gap_threshold_minutes: float = TIME_BASED_MC_GAP_MINUTES,
                 keep_recent: int = TIME_BASED_MC_KEEP_RECENT,
                 compactable_tools: Optional[set] = None):
        self.gap_threshold_minutes = gap_threshold_minutes
        self.keep_recent = keep_recent
        self.compactable_tools = set(COMPACTABLE_TOOLS if compactable_tools is None else compactable_tools)

    def evaluate_trigger(self, last_assistant_at: Optional[float], now: Optional[float] = None) -> Optional[float]:
        """발동하면 경과 분을, 아니면 None을 반환합니다 (원본 evaluateTimeBasedTrigger)."""
        if last_assistant_at is None:
            return None
        gap = ((now if now is not None else time.time()) - last_assistant_at) / 60
        return gap if gap >= self.gap_threshold_minutes else None

    def clear_old_results(self, messages: Sequence[Any]) -> Tuple[List[ToolMessage], int]:
        """최근 keep_recent개를 뺀 대상 도구 결과를 비운 교체 메시지와 절약 토큰 추정치를 반환합니다."""
        compactable_ids = [
            tc.get("id") for m in messages if isinstance(m, AIMessage)
            for tc in (getattr(m, "tool_calls", None) or [])
            if tc.get("name") in self.compactable_tools and tc.get("id")
        ]
        # 0이면 전부 지워 작업 맥락이 사라지므로 최소 1개는 남깁니다 (원본과 동일).
        keep = set(compactable_ids[-max(1, self.keep_recent):])
        clear = set(compactable_ids) - keep
        replacements, saved = [], 0
        for m in messages:
            if (isinstance(m, ToolMessage) and m.tool_call_id in clear
                    and m.content != TIME_BASED_MC_CLEARED_MESSAGE):
                saved += rough_token_count([m])
                replacements.append(_replace_tool_message(m, TIME_BASED_MC_CLEARED_MESSAGE))
        return replacements, saved


# =============================================================================
# 3. ContextCollapse (services/contextCollapse/ — 빌드에서 빠짐) 🟡
# =============================================================================
COLLAPSE_SUMMARY_PROMPT = (
    "Summarize the following completed portion of a conversation between a user and a coding agent "
    "in a few sentences. Keep file names, decisions, results, errors and anything the user asked "
    "to remember. Respond with plain text only and do not call tools.\n\n"
)


class ContextCollapse:
    """완료된 이전 턴을 요약으로 "접되", 원본 메시지는 state에 그대로 둡니다.

    원본 흔적(query.ts, logs.ts, autoCompact.ts 주석):
    - 원본 대화를 고쳐 쓰지 않고, 커밋 로그를 매 호출마다 다시 적용해 접힌 뷰를 만듭니다 (read-time projection).
    - 컨텍스트가 90%에 이르면 커밋을 시작하고, 접힌 요약은 별도 에이전트(ctx-agent)가 만듭니다.
    - Collapse가 켜져 있으면 Auto-Compact는 발동하지 않습니다 (둘이 경쟁하면 세밀한 맥락이 사라지므로).
    접는 단위(완료된 사용자 턴 1개씩)와 요약 프롬프트는 원본 파일이 없어 이 구현에서 정한 것입니다.
    """

    def __init__(self, llm, commit_ratio: float = COLLAPSE_COMMIT_RATIO, max_chars_per_message: int = 2000):
        self.llm = llm
        self.commit_ratio = commit_ratio
        self.max_chars_per_message = max_chars_per_message

    @staticmethod
    def _collapsed_ids(commits: Sequence[Dict[str, Any]]) -> set:
        return {mid for c in commits for mid in c.get("message_ids", [])}

    def next_span(self, messages: Sequence[Any], commits: Sequence[Dict[str, Any]]) -> Optional[List[Any]]:
        """아직 접지 않은 가장 오래된 완료 턴의 메시지를 반환합니다. 현재 턴은 대상이 아닙니다."""
        collapsed = self._collapsed_ids(commits)
        starts = user_turn_starts(messages)
        for k in range(len(starts) - 1):
            span = [
                m for m in messages[starts[k]:starts[k + 1]]
                if getattr(m, "id", None) and message_origin(m) is None
                and not isinstance(m, SystemMessage) and m.id not in collapsed
            ]
            if span:
                return span
        return None

    def _summary_prompt(self, span: Sequence[Any]) -> str:
        return COLLAPSE_SUMMARY_PROMPT + _serialize_for_summary(span, self.max_chars_per_message)

    def _build_commit(self, span: Sequence[Any], summary: str) -> Dict[str, Any]:
        return {
            "commit_id": uuid.uuid4().hex[:8],
            "message_ids": [m.id for m in span],
            "summary": summary.strip(),
            "created_at": time.time(),
        }

    def commit(self, span: Sequence[Any]) -> Optional[Dict[str, Any]]:
        try:
            response = self.llm.invoke([HumanMessage(content=self._summary_prompt(span))])
        except Exception as e:
            print(f"⚠️ [ContextCollapse] summary failed, skipping commit: {e}")
            return None
        summary = _content_text(getattr(response, "content", response))
        return self._build_commit(span, summary) if summary.strip() else None

    def project(self, messages: Sequence[Any], commits: Sequence[Dict[str, Any]]) -> List[Any]:
        """커밋된 구간을 요약 메시지 하나로 바꾼 뷰를 만듭니다."""
        if not commits:
            return list(messages)
        commit_by_id = {mid: c for c in commits for mid in c.get("message_ids", [])}
        out: List[Any] = []
        emitted = set()
        for m in messages:
            c = commit_by_id.get(getattr(m, "id", None))
            if c is None:
                out.append(m)
                continue
            if c["commit_id"] not in emitted:
                emitted.add(c["commit_id"])
                # 커밋은 사용자 턴 시작 메시지부터 담으므로, 첫 구성원에 붙은 CLAUDE.md 블록을 요약 앞으로 옮깁니다.
                out.append(HumanMessage(
                    content=(f"{_leading_system_reminder(m)}{CONTEXT_COLLAPSE_HEADER}\n{AUTO_GENERATED_NOTE}\n"
                             f"({len(c['message_ids'])} messages collapsed)\n\n{c['summary']}"),
                    additional_kwargs=origin_kwargs(ORIGIN_CONTEXT_COLLAPSE),
                ))
        return out


# =============================================================================
# 4. AutoCompactor (services/compact/autoCompact.ts, compact.ts, prompt.ts) 🟢
# =============================================================================
# 원본 prompt.ts의 NO_TOOLS_PREAMBLE + BASE_COMPACT_PROMPT + NO_TOOLS_TRAILER
NO_TOOLS_PREAMBLE = """CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.

- Do NOT use any tool.
- You already have all the context you need in the conversation above.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.

"""

BASE_COMPACT_PROMPT = """Your task is to create a detailed summary of the conversation so far, paying close attention to the user's explicit requests and your previous actions.
This summary should be thorough in capturing technical details, code patterns, and architectural decisions that would be essential for continuing development work without losing context.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - Your approach to addressing the user's requests
   - Key decisions, technical concepts and code patterns
   - Specific details like:
     - file names
     - full code snippets
     - function signatures
     - file edits
   - Errors that you ran into and how you fixed them
   - Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
2. Double-check for technical accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Errors and fixes: List all errors that you ran into, and how you fixed them. Pay special attention to specific user feedback that you received, especially if the user told you to do something differently.
5. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
6. All user messages: List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent.
7. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
8. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
9. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first.
                       If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.

Here's an example of how your output should be structured:

<example>
<analysis>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</analysis>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Key Technical Concepts:
   - [Concept 1]
   - [Concept 2]
   - [...]

3. Files and Code Sections:
   - [File Name 1]
      - [Summary of why this file is important]
      - [Summary of the changes made to this file, if any]
      - [Important Code Snippet]
   - [File Name 2]
      - [Important Code Snippet]
   - [...]

4. Errors and fixes:
    - [Detailed description of error 1]:
      - [How you fixed the error]
      - [User feedback on the error if any]
    - [...]

5. Problem Solving:
   [Description of solved problems and ongoing troubleshooting]

6. All user messages:
    - [Detailed non tool use user message]
    - [...]

7. Pending Tasks:
   - [Task 1]
   - [Task 2]
   - [...]

8. Current Work:
   [Precise description of current work]

9. Optional Next Step:
   [Optional Next step to take]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.
"""

NO_TOOLS_TRAILER = (
    "\n\nREMINDER: Do NOT call any tools. Respond with plain text only — "
    "an <analysis> block followed by a <summary> block. "
    "Tool calls will be rejected and you will fail the task."
)


def get_compact_prompt(custom_instructions: Optional[str] = None) -> str:
    prompt = NO_TOOLS_PREAMBLE + BASE_COMPACT_PROMPT
    if custom_instructions and custom_instructions.strip():
        prompt += f"\n\nAdditional Instructions:\n{custom_instructions}"
    return prompt + NO_TOOLS_TRAILER


def format_compact_summary(summary: str) -> str:
    """원본 formatCompactSummary: <analysis> 초안을 지우고 <summary> 태그를 'Summary:' 머리말로 바꿉니다."""
    text = re.sub(r"<analysis>[\s\S]*?</analysis>", "", summary, count=1)
    match = re.search(r"<summary>([\s\S]*?)</summary>", text)
    if match:
        text = text.replace(match.group(0), f"Summary:\n{match.group(1).strip()}", 1)
    text = re.sub(r"\n\n+", "\n\n", text)
    return text.strip()


def get_compact_user_summary_message(summary: str, recent_messages_preserved: bool = False,
                                     suppress_follow_up_questions: bool = True) -> str:
    """원본 getCompactUserSummaryMessage의 본문."""
    text = (
        "This session is being continued from a previous conversation that ran out of context. "
        "The summary below covers the earlier portion of the conversation.\n\n"
        f"{format_compact_summary(summary)}"
    )
    if recent_messages_preserved:
        text += "\n\nRecent messages are preserved verbatim."
    if suppress_follow_up_questions:
        text += (
            "\nContinue the conversation from where it left off without asking the user any further "
            "questions. Resume directly — do not acknowledge the summary, do not recap what was happening, "
            "do not preface with \"I'll continue\" or similar. Pick up the last task as if the break never happened."
        )
    return text


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
    """토큰이 임계치를 넘으면 이전 턴을 원본의 9개 섹션 형식으로 요약해 대화를 교체합니다.

    - 요약에 실패하면 압축하지 않고 실패 횟수만 늘립니다. 3회 연속 실패하면 이 세션에서는 더 시도하지
      않습니다 (원본 circuit breaker: MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3).
    - 요약 호출 자체가 prompt_too_long이면 가장 오래된 API 라운드부터 버리고 최대 3회 다시 시도합니다.
    - 압축 뒤에는 AmnesiaGuard가 최근 파일과 활성 계획을 복원 블록으로 붙입니다
      (원본 POST_COMPACT_MAX_FILES_TO_RESTORE = 5, createPlanAttachmentIfNeeded).
    - 원본 L1~L5 시스템 프롬프트만 맨 앞에 남기고, 나머지 SystemMessage는 요약 대상에 넣습니다.
    """

    MAX_CONSECUTIVE_FAILURES = MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES

    # 이전 버전 파이프라인이 만든 SystemMessage 머리말 (원본 L1~L5 시스템 프롬프트가 아님)
    _PIPELINE_PREFIXES = (
        "[Context Collapsed:",
        "[Compaction Amnesia Guard:",
        "[Reactive Compact",
        "Previous Conversation Summary:",
    )

    def __init__(self, llm, threshold_tokens: int = 8000, amnesia_guard=None,
                 max_consecutive_failures: int = MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES,
                 max_chars_per_message: int = 4000):
        self.llm = llm
        self.threshold_tokens = threshold_tokens
        self.amnesia_guard = amnesia_guard
        self.max_consecutive_failures = max_consecutive_failures
        self.max_chars_per_message = max_chars_per_message
        self.consecutive_failures = 0

    @property
    def is_circuit_open(self) -> bool:
        return self.consecutive_failures >= self.max_consecutive_failures

    def reset_circuit_breaker(self):
        self.consecutive_failures = 0

    def get_token_count(self, messages: list) -> int:
        try:
            return self.llm.get_num_tokens_from_messages(messages)
        except Exception:
            return rough_token_count(messages)

    def _is_original_system_msg(self, msg) -> bool:
        if not isinstance(msg, SystemMessage):
            return False
        return not str(msg.content).startswith(self._PIPELINE_PREFIXES)

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
        """대화를 압축 대상(과거 턴)과 보존 대상(현재 진행 중인 마지막 턴)으로 나눕니다."""
        last_human_idx = last_user_turn_start(dialogue_messages)
        if last_human_idx is not None and last_human_idx > 0:
            return dialogue_messages[:last_human_idx], dialogue_messages[last_human_idx:]

        # 단일 턴: 마지막이 ToolMessage면 그 도구를 부른 AIMessage까지 함께 보존
        if isinstance(dialogue_messages[-1], ToolMessage):
            idx = len(dialogue_messages) - 1
            while idx > 0 and isinstance(dialogue_messages[idx], ToolMessage):
                idx -= 1
            if idx > 0 and isinstance(dialogue_messages[idx], AIMessage):
                return dialogue_messages[:idx], dialogue_messages[idx:]

        return dialogue_messages[:-1], [dialogue_messages[-1]]

    def build_summary_request(self, to_compact: Sequence[Any]) -> str:
        """요약 LLM에 보낼 프롬프트: 직렬화한 대화 뒤에 원본 요약 지시문을 붙입니다."""
        history = _serialize_for_summary(to_compact, self.max_chars_per_message)
        return f"<conversation>\n{history}\n</conversation>\n\n{get_compact_prompt()}"

    def _plan_compaction(self, messages: list, force: bool) -> Optional[tuple[list, list, list]]:
        """압축이 필요하면 (원본 시스템 메시지, 압축 대상, 보존 대상)을, 아니면 None을 반환합니다."""
        if self.is_circuit_open:
            print(f"⚠️ [AutoCompactor Circuit Breaker ACTIVE] {self.consecutive_failures} consecutive failures. Skipping compaction.")
            return None
        if not force and self.get_token_count(messages) <= self.threshold_tokens:
            return None

        original_system_messages = [m for m in messages if self._is_original_system_msg(m)]
        dialogue_messages = [m for m in messages if not self._is_original_system_msg(m)]
        if len(dialogue_messages) <= 1:
            return None

        to_compact, tail_messages = self._split_dialogue_for_compaction(dialogue_messages)
        if not to_compact:
            return None
        return original_system_messages, to_compact, tail_messages

    def _summarize(self, to_compact: Sequence[Any]) -> str:
        to_summarize = list(to_compact)
        for attempt in range(MAX_PTL_RETRIES + 1):
            try:
                response = self.llm.invoke([HumanMessage(content=self.build_summary_request(to_summarize))])
                return _content_text(getattr(response, "content", response))
            except Exception as e:
                truncated = (truncate_head_for_ptl_retry(to_summarize)
                             if attempt < MAX_PTL_RETRIES and is_prompt_too_long_error(e) else None)
                if truncated is None:
                    raise
                print(f"⚠️ [AutoCompactor] summary request too long, dropping oldest API rounds (retry {attempt + 1}/{MAX_PTL_RETRIES})")
                to_summarize = truncated
        raise RuntimeError("unreachable")

    async def _asummarize(self, to_compact: Sequence[Any]) -> str:
        to_summarize = list(to_compact)
        for attempt in range(MAX_PTL_RETRIES + 1):
            try:
                request = [HumanMessage(content=self.build_summary_request(to_summarize))]
                if hasattr(self.llm, "ainvoke"):
                    response = await self.llm.ainvoke(request)
                else:
                    response = await asyncio.to_thread(self.llm.invoke, request)
                return _content_text(getattr(response, "content", response))
            except Exception as e:
                truncated = (truncate_head_for_ptl_retry(to_summarize)
                             if attempt < MAX_PTL_RETRIES and is_prompt_too_long_error(e) else None)
                if truncated is None:
                    raise
                to_summarize = truncated
        raise RuntimeError("unreachable")

    def _record_failure(self, error: Exception) -> None:
        self.consecutive_failures += 1
        print(f"⚠️ [AutoCompactor Failure] summary failed ({self.consecutive_failures}/{self.max_consecutive_failures}): {error}")
        if self.is_circuit_open:
            print(f"🚨 [AutoCompactor Circuit Tripped] {self.max_consecutive_failures} consecutive failures. "
                  "Auto-compaction is disabled for this session.")

    def _finalize_compacted_messages(self, original_system_messages: list, summary_raw: str,
                                     tail_messages: list) -> list:
        summary_msg = build_compaction_summary_message(
            get_compact_user_summary_message(summary_raw, recent_messages_preserved=bool(tail_messages))
        )
        recovery_attachments = []
        if self.amnesia_guard:
            for att in self.amnesia_guard.create_recovery_attachments():
                ak = dict(getattr(att, "additional_kwargs", None) or {})
                for key, value in origin_kwargs(ORIGIN_AMNESIA_RESTORE).items():
                    ak.setdefault(key, value)
                att.additional_kwargs = ak
                recovery_attachments.append(att)
        return original_system_messages + [summary_msg] + recovery_attachments + list(tail_messages)

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
        try:
            summary_raw = self._summarize(to_compact)
        except Exception as e:
            self._record_failure(e)
            return messages, False, []
        self.consecutive_failures = 0
        compacted = self._finalize_compacted_messages(original_system_messages, summary_raw, tail_messages)
        return compacted, True, to_compact

    async def acompact_with_details(self, messages: list, force: bool = False) -> tuple[list, bool, list]:
        plan = self._plan_compaction(messages, force)
        if plan is None:
            return messages, False, []
        original_system_messages, to_compact, tail_messages = plan
        try:
            summary_raw = await self._asummarize(to_compact)
        except Exception as e:
            self._record_failure(e)
            return messages, False, []
        self.consecutive_failures = 0
        compacted = self._finalize_compacted_messages(original_system_messages, summary_raw, tail_messages)
        return compacted, True, to_compact


# =============================================================================
# 5. ReactiveCompactor (reactiveCompact.ts — 빌드에서 빠짐) 🟡
# =============================================================================
class ReactiveCompactor:
    """API가 prompt_too_long(413)을 돌려주면 압축한 뒤 같은 요청을 한 번 다시 보냅니다.

    원본 흔적: query.ts가 413 응답을 사용자에게 바로 보여주지 않고 보류(withhold)한 뒤 복구를 시도합니다.
    여기서는 이전 턴 요약(AutoCompactor)을 먼저 시도하고, 요약할 이전 턴이 없거나 요약이 실패하면
    가장 오래된 API 라운드부터 버립니다 (원본 truncateHeadForPTLRetry와 같은 방식).
    복구 결과는 이번 모델 호출의 입력에만 쓰입니다. state는 다음 before_model의 Auto-Compact가 정리합니다.
    """

    def __init__(self, auto_compactor: Optional[AutoCompactor] = None, drop_ratio: float = 0.2):
        self.auto_compactor = auto_compactor
        self.drop_ratio = drop_ratio

    def handle_overflow(self, messages: Sequence[Any]) -> List[Any]:
        """LLM 없이 가장 오래된 API 라운드를 잘라냅니다. 시스템 메시지와 압축 요약·복원 블록은 맨 앞에 고정합니다."""
        system = [m for m in messages if isinstance(m, SystemMessage)]
        pinned = [m for m in messages if not isinstance(m, SystemMessage) and is_compaction_artifact(m)]
        dialogue = [m for m in messages if not isinstance(m, SystemMessage) and not is_compaction_artifact(m)]

        groups = group_messages_by_api_round(dialogue)
        if len(groups) < 2:
            if not dialogue:
                return list(messages)
            last = dialogue[-1]
            truncated = _content_text(last.content)[:1000] + "\n... [Reactive Compact: truncated oversized message]"
            return system + pinned + [HumanMessage(content=truncated)]

        drop = min(max(1, int(len(groups) * self.drop_ratio)), len(groups) - 1)
        dropped = sum(len(g) for g in groups[:drop])
        marker = _marker_message(
            f"[Reactive Compact] The oldest {dropped} message(s) were dropped after a prompt-too-long error.\n"
            f"{AUTO_GENERATED_NOTE}"
        )
        return system + pinned + [marker] + [m for g in groups[drop:] for m in g]

    def _can_summarize(self, messages: Sequence[Any]) -> bool:
        ac = self.auto_compactor
        return bool(ac) and not ac.is_circuit_open and ac.has_completed_turns(list(messages))

    def recover(self, messages: Sequence[Any]) -> List[Any]:
        if self._can_summarize(messages):
            compacted, ok, _ = self.auto_compactor.compact_with_details(list(messages), force=True)
            if ok:
                return compacted
        return self.handle_overflow(messages)

    async def arecover(self, messages: Sequence[Any]) -> List[Any]:
        if self._can_summarize(messages):
            compacted, ok, _ = await self.auto_compactor.acompact_with_details(list(messages), force=True)
            if ok:
                return compacted
        return self.handle_overflow(messages)


# =============================================================================
# 6. CompactorMiddleware
# =============================================================================
DEFAULT_COMPACTION_CONFIG: Dict[str, Dict[str, Any]] = {
    "tool_result_budget": {
        "enabled": True,
        "max_result_chars": DEFAULT_MAX_RESULT_SIZE_CHARS,
        "per_message_chars": MAX_TOOL_RESULTS_PER_MESSAGE_CHARS,
        "preview_chars": PREVIEW_SIZE_CHARS,
    },
    "snip": {"enabled": True, "nudge_ratio": 0.7},
    "micro": {"enabled": True, "gap_threshold_minutes": TIME_BASED_MC_GAP_MINUTES,
              "keep_recent": TIME_BASED_MC_KEEP_RECENT},
    "collapse": {"enabled": False, "commit_ratio": COLLAPSE_COMMIT_RATIO},
    "auto": {"enabled": True},
    "reactive": {"enabled": True},
}


def _merge_config(config: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    merged = {k: dict(v) for k, v in DEFAULT_COMPACTION_CONFIG.items()}
    for key, value in (config or {}).items():
        if key in merged and isinstance(value, dict):
            merged[key].update(value)
    return merged


class CompactorMiddleware(AgentMiddleware):
    """Claude Code 압축 파이프라인을 LangChain 미들웨어 훅에 나눠 붙입니다.

    - wrap_tool_call   : [0] 도구별 결과 크기 상한 → 디스크 저장
    - before_model     : [0] 라운드 합계 예산 → [2] time-based microcompact → [3] Collapse 커밋 → [4] Auto-Compact
                         ([0][2]는 같은 id로 state의 ToolMessage를 교체, [4]는 state 전체를 요약본으로 교체)
    - after_model      : [2]의 기준 시각(last_assistant_at) 기록
    - wrap_model_call  : [1] Snip 투영·[id] 태그·권유 → [3] Collapse 투영 → 모델 호출 → 413이면 [5] Reactive
                         (투영 결과는 이번 모델 입력 사본에만 쓰고 state에는 저장하지 않음)
    - tools            : [1] snip 도구

    원본 대화 전체는 MemoryMiddleware가 transcript.db에 따로 보존합니다.
    """

    state_schema = CompactionState

    def __init__(self, llm, threshold_tokens: int = 8000, amnesia_guard=None,
                 swap_dir: str = "./artifacts/tool-results",
                 max_consecutive_failures: int = MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES,
                 transcript_store: Optional[Any] = None,
                 config: Optional[Dict[str, Any]] = None):
        cfg = _merge_config(config)
        self.config = cfg
        budget_cfg, snip_cfg, micro_cfg = cfg["tool_result_budget"], cfg["snip"], cfg["micro"]
        collapse_cfg = cfg["collapse"]

        self.tool_result_budget = ToolResultBudget(
            storage_dir=budget_cfg.get("storage_dir", swap_dir),
            max_result_chars=budget_cfg["max_result_chars"],
            per_message_chars=budget_cfg["per_message_chars"],
            preview_chars=budget_cfg["preview_chars"],
            tool_limits=budget_cfg.get("tool_limits"),
        ) if budget_cfg["enabled"] else None
        self.snip_compactor = SnipCompactor(nudge_ratio=snip_cfg["nudge_ratio"]) if snip_cfg["enabled"] else None
        self.micro_compactor = MicroCompactor(
            gap_threshold_minutes=micro_cfg["gap_threshold_minutes"],
            keep_recent=micro_cfg["keep_recent"],
        ) if micro_cfg["enabled"] else None
        self.context_collapse = ContextCollapse(
            llm=llm, commit_ratio=collapse_cfg["commit_ratio"],
        ) if collapse_cfg["enabled"] else None
        self.auto_compactor = AutoCompactor(
            llm=llm,
            threshold_tokens=threshold_tokens,
            amnesia_guard=amnesia_guard,
            max_consecutive_failures=max_consecutive_failures,
        )
        self.auto_enabled = bool(cfg["auto"]["enabled"])
        self.reactive_compactor = ReactiveCompactor(
            auto_compactor=self.auto_compactor if self.auto_enabled else None,
        ) if cfg["reactive"]["enabled"] else None
        self.transcript_store = transcript_store
        self.tools = [self.snip_compactor.build_tool()] if self.snip_compactor else []

    # ── 모델 입력 뷰 ──

    def project_view(self, messages: Sequence[Any], state: Optional[Dict[str, Any]] = None) -> List[Any]:
        """state의 snip·collapse 기록을 적용한 모델 입력 뷰 (태그·권유 메시지 제외)."""
        state = state or {}
        view = list(messages)
        if self.snip_compactor:
            view = self.snip_compactor.project(view, state.get("snipped_message_ids") or [])
        if self.context_collapse:
            view = self.context_collapse.project(view, state.get("collapse_commits") or [])
        return view

    def _decorate_view(self, view: List[Any]) -> List[Any]:
        if not self.snip_compactor:
            return view
        tagged = self.snip_compactor.tag_messages(view)
        nudge = self.snip_compactor.nudge(rough_token_count(view), self.auto_compactor.threshold_tokens)
        return tagged + [nudge] if nudge else tagged

    # ── before_model ──

    def _auto_compact_allowed(self) -> bool:
        # 원본: Collapse가 켜져 있으면 Auto-Compact는 발동하지 않습니다 (413은 Reactive가 처리).
        return self.auto_enabled and self.context_collapse is None

    def _needs_auto_compact(self, view: list) -> bool:
        """모델이 실제로 보게 될 크기가 임계치를 넘고, 요약할 이전 턴이 있는지."""
        if not view or not self._auto_compact_allowed() or self.auto_compactor.is_circuit_open:
            return False
        if self.auto_compactor.get_token_count(view) <= self.auto_compactor.threshold_tokens:
            return False
        return self.auto_compactor.has_completed_turns(view)

    def _collect_replacements(self, state: Dict[str, Any], messages: List[Any]) -> List[Any]:
        replacements: List[Any] = []
        if self.tool_result_budget:
            replacements += self.tool_result_budget.enforce_round_budget(messages)
        if self.micro_compactor:
            gap = self.micro_compactor.evaluate_trigger(state.get("last_assistant_at"))
            if gap is not None:
                cleared, saved = self.micro_compactor.clear_old_results(_apply_replacements(messages, replacements))
                if cleared:
                    print(f"🧹 [Microcompact] {gap:.0f}min idle → cleared {len(cleared)} old tool results (~{saved:,} tokens)")
                    replacements = list({m.id: m for m in replacements + cleared}.values())
        return replacements

    def _collapse_candidate(self, working: List[Any], state: Dict[str, Any],
                            commits: List[Dict[str, Any]]) -> Optional[List[Any]]:
        """Collapse 커밋이 필요하면 접을 구간을, 아니면 None을 반환합니다."""
        if not self.context_collapse:
            return None
        snip_view = (self.snip_compactor.project(working, state.get("snipped_message_ids") or [])
                     if self.snip_compactor else working)
        view = self.context_collapse.project(snip_view, commits)
        limit = self.auto_compactor.threshold_tokens * self.context_collapse.commit_ratio
        if self.auto_compactor.get_token_count(view) < limit:
            return None
        return self.context_collapse.next_span(snip_view, commits)

    def _compaction_update(self, runtime: Any, original: List[Any], compacted: List[Any]) -> Dict[str, Any]:
        kept_ids = {m.id for m in compacted if getattr(m, "id", None)}
        source = [m for m in original if getattr(m, "id", None) and m.id not in kept_ids
                  and not isinstance(m, SystemMessage)]
        self._record_compaction_to_transcript(runtime, compacted, len(original), source)
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *compacted],
            "snipped_message_ids": [],
            "collapse_commits": [],
        }

    def before_model(self, state: Dict[str, Any], runtime: Any = None) -> Optional[Dict[str, Any]]:
        messages = list(state.get("messages", []))
        if not messages:
            return None
        update: Dict[str, Any] = {}

        replacements = self._collect_replacements(state, messages)
        working = _apply_replacements(messages, replacements)
        if replacements:
            update["messages"] = replacements

        commits = list(state.get("collapse_commits") or [])
        span = self._collapse_candidate(working, state, commits)
        if span:
            commit = self.context_collapse.commit(span)
            if commit:
                commits.append(commit)
                update["collapse_commits"] = commits

        view = self.project_view(working, {**state, "collapse_commits": commits})
        if self._needs_auto_compact(view):
            compacted, ok, _ = self.auto_compactor.compact_with_details(view, force=True)
            if ok:
                return self._compaction_update(runtime, working, compacted)
        return update or None

    async def abefore_model(self, state: Dict[str, Any], runtime: Any = None) -> Optional[Dict[str, Any]]:
        messages = list(state.get("messages", []))
        if not messages:
            return None
        update: Dict[str, Any] = {}

        replacements = await asyncio.to_thread(self._collect_replacements, state, messages)
        working = _apply_replacements(messages, replacements)
        if replacements:
            update["messages"] = replacements

        commits = list(state.get("collapse_commits") or [])
        span = self._collapse_candidate(working, state, commits)
        if span:
            commit = await asyncio.to_thread(self.context_collapse.commit, span)
            if commit:
                commits.append(commit)
                update["collapse_commits"] = commits

        view = self.project_view(working, {**state, "collapse_commits": commits})
        if self._needs_auto_compact(view):
            compacted, ok, _ = await self.auto_compactor.acompact_with_details(view, force=True)
            if ok:
                return await asyncio.to_thread(self._compaction_update, runtime, working, compacted)
        return update or None

    # ── after_model ──

    def after_model(self, state: Dict[str, Any], runtime: Any = None) -> Optional[Dict[str, Any]]:
        return {"last_assistant_at": time.time()} if self.micro_compactor else None

    async def aafter_model(self, state: Dict[str, Any], runtime: Any = None) -> Optional[Dict[str, Any]]:
        return self.after_model(state, runtime)

    # ── wrap_tool_call ──

    def _apply_tool_budget(self, request, result):
        if not self.tool_result_budget or not isinstance(result, ToolMessage):
            return result
        tool_name = (getattr(request, "tool_call", None) or {}).get("name")
        return self.tool_result_budget.apply_to_tool_message(result, tool_name)

    def wrap_tool_call(self, request, handler):
        return self._apply_tool_budget(request, handler(request))

    async def awrap_tool_call(self, request, handler):
        return self._apply_tool_budget(request, await handler(request))

    # ── wrap_model_call ──

    def _prepare(self, request) -> Tuple[List[Any], List[Any]]:
        view = self.project_view(request.messages, getattr(request, "state", None))
        return view, self._decorate_view(view)

    def wrap_model_call(self, request, handler):
        view, model_input = self._prepare(request)
        try:
            return handler(request.override(messages=model_input))
        except Exception as e:
            if not (self.reactive_compactor and is_prompt_too_long_error(e)):
                raise
            print("\n⚡ [Reactive Compact] prompt too long — compacting and retrying once...")
            recovered = self.reactive_compactor.recover(view)
            return handler(request.override(messages=recovered))

    async def awrap_model_call(self, request, handler):
        view, model_input = self._prepare(request)
        try:
            return await handler(request.override(messages=model_input))
        except Exception as e:
            if not (self.reactive_compactor and is_prompt_too_long_error(e)):
                raise
            print("\n⚡ [Reactive Compact] prompt too long — compacting and retrying once...")
            recovered = await self.reactive_compactor.arecover(view)
            return await handler(request.override(messages=recovered))

    # ── transcript.db 압축 이벤트 ──

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


def create_compactor_middleware(llm, threshold_tokens: int = 8000, amnesia_guard=None,
                                swap_dir: str = "./artifacts/tool-results",
                                max_consecutive_failures: int = MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES,
                                transcript_store: Optional[Any] = None,
                                config: Optional[Dict[str, Any]] = None):
    """CompactorMiddleware를 만듭니다.

    config는 configs/compaction.config와 같은 구조의 dict입니다 (tool_result_budget / snip / micro /
    collapse / auto / reactive). 빠진 항목은 DEFAULT_COMPACTION_CONFIG 값을 씁니다.
    """
    return CompactorMiddleware(
        llm=llm,
        threshold_tokens=threshold_tokens,
        amnesia_guard=amnesia_guard,
        swap_dir=swap_dir,
        max_consecutive_failures=max_consecutive_failures,
        transcript_store=transcript_store,
        config=config,
    )
