"""
===============================================================================
Background Skill Review — Procedural Memory 학습 루프
===============================================================================
Source: hermes-agent/agent/background_review.py (_SKILL_REVIEW_PROMPT, 리뷰 포크, 도구 화이트리스트)

흐름:
1. MemoryMiddleware가 세션별 "도구 호출 반복 횟수"를 셉니다 (도구를 호출한 모델 응답 1회 = 1).
   기준값(기본 10)에 도달하면 다음 유휴 리뷰에 스킬 리뷰를 포함시킵니다.
2. 이 모듈이 스킬 도구 3종(skills_list, skill_view, skill_manage)만 가진 리뷰 에이전트를 만들고,
   세션 원문을 보여 준 뒤 "스킬 라이브러리를 갱신하라"고 지시합니다.
3. 리뷰 에이전트의 쓰기는 ReviewSession 가드를 거칩니다:
   에이전트 소유 스킬만 수정 / 이번 리뷰에서 읽은 파일만 수정 / 삭제는 보관.

Hermes와 다른 점 (교육용 단순화):
- Hermes는 부모 에이전트의 시스템 프롬프트·도구 목록을 바이트 단위로 물려받는 포크를 만들어
  프롬프트 캐시를 재사용합니다. 여기서는 review_llm으로 새 에이전트를 만들고 원문을 텍스트로 전달합니다.
- Hermes는 메모리와 스킬 신호가 함께 오면 합본 프롬프트(_COMBINED_REVIEW_PROMPT)로 한 번에 리뷰합니다.
  여기서는 기존 Semantic 리뷰와 분리해 스킬 리뷰만 수행합니다.
===============================================================================
"""

import json
import logging
import threading
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage, ToolMessage

from app.middleware.memory.procedural_store import ProceduralMemoryStore, ReviewSession
from app.middleware.memory.skill_tools import SKILL_MANAGE_TOOL, build_skill_tools
from app.middleware.memory.transcript_store import render_transcript_for_summary
from app.utils.message_utils import normalize_content

logger = logging.getLogger(__name__)

# Hermes _LESSON_LAYER_BLOCK + _SKILL_REVIEW_PROMPT + _DO_NOT_CAPTURE_BLOCK 축약본
SKILL_REVIEW_PROMPT = """You maintain the agent's skill library (procedural memory). You will be shown a finished
conversation. Decide whether it taught a reusable way of doing a CLASS of task, and record it.

What a skill IS: the instructions for doing a class of task the most efficient and correct way, to THIS
user's specifications — the procedure, the tools and commands that work, the order, the user's preferences
for how the result should look, and the pitfalls that cost time. A future session should be able to follow
it and produce what the user wants on the first try.
  • Procedure first: numbered steps in the order they are done, with concrete tool calls and decision
    points. Attach each pitfall to the step it affects.
  • A pitfall is a generalizable rule + one clause of WHY, written as an imperative. Not a story of what
    happened in this session.
  • No dates, ticket IDs, file names that only existed in this session, or quoted user text as content —
    the rule must stand without the incident behind it.
  • The same lesson learned twice is ONE rule: strengthen the existing sentence instead of appending.
  • Fix a wrong skill in place: edit the sentence that misled; never append "UPDATE: actually...".

Signals that warrant action (any one is enough):
  • The user corrected your style, format, workflow, or order of steps. Embed the correction in the skill
    that governs that task, so the next session starts already knowing. Write the corrected behavior as
    the DEFAULT steps and output format — never as "if the user asks for X"; next time nobody will ask.
  • A non-trivial technique, workaround, or multi-step tool sequence emerged that would help next time.
  • A skill used in this session turned out wrong, incomplete, or outdated.

Preference order — pick the earliest that fits:
  1. PATCH an existing skill that covers this class of task (call skills_list, then skill_view it first).
  2. ADD A SUPPORT FILE under an existing skill (skill_manage action=write_file, file_path starting with
     references/, templates/, or scripts/), and add a one-line pointer to it in SKILL.md.
  3. CREATE a new CLASS-LEVEL skill when nothing covers it. The name must describe the class of task
     (e.g. "data-integrity-report"), never today's specific file, error string, or "fix-x-today".

SKILL.md format for create:
---
name: <skill-name>
description: Use when <trigger>. <one-line behavior>.
---
# <Title>
## Procedure
1. ...
## Pitfalls
- ...
## Output preferences
- ...

Read-before-write (ENFORCED — skill_manage refuses otherwise): before patching an existing skill, call
skill_view(name) in this review and base old_string on what it returned. On a refusal, view once and retry
once; do not loop.

Protected skills: skills you did not create (user-owned) will refuse your writes. If one needs fixing,
say so in your final reply instead.

Do NOT capture as skills:
  • Environment-dependent failures (missing packages, unset credentials, wrong paths) — the user can fix those.
  • Negative claims about tools ("tool X does not work") — they harden into refusals after the cause is fixed.
  • Transient errors that a retry resolved — the lesson, if any, is the retry pattern.
  • One-off task narratives with no reusable class of work.
  • Unresolved failures: if no working method was found, do not write the dead ends up as a workflow.

If the session ran smoothly with no corrections and no new technique, reply "Nothing to save." and stop.
Otherwise act, then finish with a one-line summary of what you changed.
Write skill content in English."""


class _CancelGuard(AgentMiddleware):
    """새 사용자 턴이 시작되어 cancel_event가 설정되면 다음 모델 호출 전에 리뷰를 끝냅니다."""

    def __init__(self, cancel_event: Optional[threading.Event]):
        super().__init__()
        self.cancel_event = cancel_event

    @hook_config(can_jump_to=["end"])
    def before_model(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        if self.cancel_event is not None and self.cancel_event.is_set():
            return {"jump_to": "end"}
        return None


def build_review_request(store: ProceduralMemoryStore, messages: List[Dict[str, Any]]) -> str:
    """리뷰 에이전트에 줄 사용자 메시지: 세션 원문 + 현재 스킬 목록."""
    transcript = render_transcript_for_summary(messages, budget_chars=16000)
    skills = store.list_skills()
    skill_lines = "\n".join(
        f"- {s['name']} ({'yours' if s['created_by'] == 'agent' else 'user-owned'}): {s['description']}"
        for s in skills
    ) or "(no learned skills yet)"
    return (
        f"<conversation>\n{transcript}\n</conversation>\n\n"
        f"Existing learned skills:\n{skill_lines}\n\n"
        "Review the conversation above and update the skill library following your instructions."
    )


def summarize_skill_actions(messages: List[Any]) -> List[str]:
    """리뷰 에이전트의 skill_manage 결과 중 성공한 쓰기만 사람이 읽을 한 줄 요약으로 변환합니다."""
    verbs = {"create": "created", "patch": "patched", "write_file": "written",
             "remove_file": "removed", "delete": "archived"}
    actions = []
    for m in messages:
        if not isinstance(m, ToolMessage) or m.name != SKILL_MANAGE_TOOL:
            continue
        try:
            data = json.loads(m.content)
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict) or not data.get("success"):
            continue
        verb = verbs.get(data.get("action"), data.get("action"))
        path = f" ({data['file_path']})" if data.get("file_path") else ""
        actions.append(f"Skill '{data.get('name')}' {verb}{path}")
    return actions


def summarize_review_trace(messages: List[Any]) -> List[str]:
    """리뷰 에이전트가 호출한 스킬 도구와 결과(성공/오류)를 순서대로 한 줄씩 정리합니다."""
    trace = []
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        try:
            data = json.loads(m.content)
        except (TypeError, json.JSONDecodeError):
            data = {}
        if isinstance(data, dict) and data.get("success") is False:
            trace.append(f"{m.name} ✗ {str(data.get('error', ''))[:120]}")
        else:
            trace.append(f"{m.name} ✓")
    return trace


def run_skill_review(
    llm: Any,
    store: ProceduralMemoryStore,
    messages: List[Dict[str, Any]],
    cancel_event: Optional[threading.Event] = None,
    max_iterations: int = 16,
) -> Dict[str, Any]:
    """스킬 리뷰 에이전트를 실행하고 변경 요약을 반환합니다.

    Returns:
        {"actions": ["Skill 'x' created", ...], "trace": 도구 호출 순서와 결과,
         "reply": 리뷰 에이전트의 마지막 응답, "cancelled": bool}
    """
    review = ReviewSession()
    tools = build_skill_tools(lambda: store, review=review, cancel_event=cancel_event, include_list=True)
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SKILL_REVIEW_PROMPT,
        middleware=[_CancelGuard(cancel_event)],
    )
    request = HumanMessage(content=build_review_request(store, messages))
    result_messages: List[Any] = []
    stopped_early = False
    try:
        # stream으로 상태를 받아 두면, 반복 상한에 걸려 중단돼도 그때까지의 쓰기와 기록이 남습니다.
        for state in agent.stream(
            {"messages": [request]},
            config={"recursion_limit": max_iterations * 2 + 1},
            stream_mode="values",
        ):
            result_messages = state.get("messages", result_messages)
    except Exception as e:  # GraphRecursionError 등 — 이미 반영된 쓰기는 유지
        stopped_early = True
        logger.warning("[BG-Review] Skill review stopped early: %s", e)

    actions = summarize_skill_actions(result_messages)
    reply = ""
    if result_messages and not isinstance(result_messages[-1], (ToolMessage, HumanMessage)):
        reply = normalize_content(getattr(result_messages[-1], "content", "") or "")
    cancelled = bool(cancel_event is not None and cancel_event.is_set())
    for line in actions:
        logger.info("[BG-Review] %s", line)
    return {
        "actions": actions,
        "trace": summarize_review_trace(result_messages),
        "reply": reply,
        "cancelled": cancelled,
        "stopped_early": stopped_early,
    }
