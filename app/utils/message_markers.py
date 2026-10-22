"""
하네스가 대화(state["messages"])에 직접 넣는 메시지의 출처 표식과 "사용자 턴 경계" 판별 규칙.

압축 요약·작업 맥락 복원·자가 수정 피드백은 모두 HumanMessage로 대화에 들어갑니다.
그래서 "HumanMessage = 새 사용자 턴"으로 가정하면 턴을 잘못 셉니다. 턴 경계가 필요한
모든 미들웨어(Compactor, StopHooks, Memory, PromptAssembler)는 이 모듈의 함수만 사용합니다.

규칙:
1. 하네스가 만든 메시지는 additional_kwargs["harness_origin"]에 출처를 기록합니다.
   - compaction_summary : AutoCompactor가 만든 이전 대화 요약
   - amnesia_restore    : AmnesiaGuard가 만든 작업 맥락(파일·계획) 복원 블록
   - stop_hook_feedback : StopHooks가 넣은 코드 검증 실패 피드백
   아래 출처는 Compactor가 모델 입력 사본에만 넣고 state에는 저장하지 않습니다.
   - context_collapse   : ContextCollapse가 접은 이전 턴 요약
   - snip_boundary      : Snip으로 제거된 구간 표시
   - context_efficiency : Snip 사용을 권하는 알림
   - compaction_marker  : 413 대응·요약 재시도에서 앞부분을 잘라냈다는 표시
   PromptAssembler가 넣는 메시지:
   - user_context       : CLAUDE.md + 세션 시작 날짜 (모델 입력 사본의 맨 앞, state에는 저장 안 함)
   - date_change        : 자정이 지나 날짜가 바뀌었다는 알림 (state 끝에 저장)
2. 사용자 턴의 시작 = 출처 표식이 없는 HumanMessage.
3. 표식이 없는 구버전 메시지(체크포인터에 남은 이전 버전 state)는 내용 머리말로 판별합니다.

이 파일은 modules/claude_code/message_markers.py와 내용이 같아야 합니다.
"""

from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import HumanMessage

ORIGIN_KEY = "harness_origin"
ORIGIN_COMPACTION_SUMMARY = "compaction_summary"
ORIGIN_AMNESIA_RESTORE = "amnesia_restore"
ORIGIN_STOP_HOOK = "stop_hook_feedback"
COMPACTION_ORIGINS = frozenset({ORIGIN_COMPACTION_SUMMARY, ORIGIN_AMNESIA_RESTORE})
ORIGIN_CONTEXT_COLLAPSE = "context_collapse"
ORIGIN_SNIP_BOUNDARY = "snip_boundary"
ORIGIN_CONTEXT_EFFICIENCY = "context_efficiency"
ORIGIN_COMPACTION_MARKER = "compaction_marker"
ORIGIN_USER_CONTEXT = "user_context"
ORIGIN_DATE_CHANGE = "date_change"

# 원문 저장소(transcript.db)에 기록하지 않을 메시지 표식 (압축 요약은 record_compaction()으로 따로 기록)
TRANSCRIPT_SKIP_KEY = "transcript_skip"

# 모델이 읽는 머리말: 사람이 보낸 요청이 아니라는 점을 메시지 첫 줄에서 분명히 밝힙니다.
COMPACTION_SUMMARY_HEADER = "[Compaction Summary: Previous Conversation]"
AMNESIA_RESTORE_HEADER = "[Compaction Amnesia Guard: Restoring Recent Work Context]"
STOP_HOOK_TAG = "[stop_hook_blocking]"
CONTEXT_COLLAPSE_HEADER = "[Context Collapse: Earlier Turn Summarized]"
SNIP_BOUNDARY_HEADER = "[Snip Boundary]"
CONTEXT_EFFICIENCY_HEADER = "[Context Efficiency Reminder]"
AUTO_GENERATED_NOTE = (
    "(This message was generated automatically by the agent harness. "
    "It is NOT a new request from the user.)"
)

# 구버전 SystemMessage 요약 머리말 (하위 호환 판별용)
_LEGACY_SUMMARY_PREFIX = "Previous Conversation Summary:"


def origin_kwargs(origin: str) -> Dict[str, Any]:
    """하네스 생성 메시지에 붙일 additional_kwargs를 만듭니다."""
    kwargs: Dict[str, Any] = {ORIGIN_KEY: origin}
    if origin in COMPACTION_ORIGINS:
        kwargs[TRANSCRIPT_SKIP_KEY] = True
    return kwargs


def _kwargs(msg: Any) -> Dict[str, Any]:
    if isinstance(msg, dict):
        value = msg.get("additional_kwargs") or {}
    else:
        value = getattr(msg, "additional_kwargs", None) or {}
    return value if isinstance(value, dict) else {}


def _text(msg: Any) -> str:
    content = msg.get("content", "") if isinstance(msg, dict) else getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [c.get("text", "") if isinstance(c, dict) else str(c) for c in content]
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _is_human(msg: Any) -> bool:
    if isinstance(msg, HumanMessage):
        return True
    if isinstance(msg, dict):
        return str(msg.get("role") or msg.get("type") or "").lower() in ("human", "user")
    return getattr(msg, "type", None) == "human"


def message_origin(msg: Any) -> Optional[str]:
    """하네스가 만든 메시지면 출처를, 사용자·모델·도구가 만든 메시지면 None을 반환합니다."""
    kwargs = _kwargs(msg)
    origin = kwargs.get(ORIGIN_KEY)
    if origin:
        return str(origin)
    text = _text(msg).lstrip()
    if text.startswith((COMPACTION_SUMMARY_HEADER, _LEGACY_SUMMARY_PREFIX)):
        return ORIGIN_COMPACTION_SUMMARY
    if text.startswith(AMNESIA_RESTORE_HEADER):
        return ORIGIN_AMNESIA_RESTORE
    if STOP_HOOK_TAG in text[:60]:
        return ORIGIN_STOP_HOOK
    # 이전 버전의 표식 키 (머리말이 바뀐 경우 대비)
    if kwargs.get("compaction_summary"):
        return ORIGIN_COMPACTION_SUMMARY
    if kwargs.get("amnesia_guard"):
        return ORIGIN_AMNESIA_RESTORE
    if kwargs.get("compaction_artifact"):
        return ORIGIN_COMPACTION_SUMMARY
    return None


def is_compaction_artifact(msg: Any) -> bool:
    """압축 단계가 만든 메시지(요약 또는 복원 블록)인지."""
    return message_origin(msg) in COMPACTION_ORIGINS


def is_compaction_summary(msg: Any) -> bool:
    return message_origin(msg) == ORIGIN_COMPACTION_SUMMARY


def is_stop_hook_feedback(msg: Any) -> bool:
    return _is_human(msg) and message_origin(msg) == ORIGIN_STOP_HOOK


def is_user_turn_start(msg: Any) -> bool:
    """실제 사용자 턴의 시작 메시지인지 (출처 표식이 없는 HumanMessage)."""
    return _is_human(msg) and message_origin(msg) is None


def user_turn_starts(messages: Sequence[Any]) -> List[int]:
    """사용자 턴이 시작되는 메시지 인덱스 목록."""
    return [i for i, m in enumerate(messages) if is_user_turn_start(m)]


def last_user_turn_start(messages: Sequence[Any]) -> Optional[int]:
    """마지막(현재) 사용자 턴이 시작되는 인덱스. 없으면 None."""
    for i in range(len(messages) - 1, -1, -1):
        if is_user_turn_start(messages[i]):
            return i
    return None
