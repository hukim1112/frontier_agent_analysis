"""Unit and integration tests for EvaluatorHarness v2.0, INTERNAL_FEEDBACK_ORIGINS, and OutputSchemaRepairGuardrail."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.middleware.evaluator import (
    EvalRunner,
    EvaluatorHarness,
    JudgeVerdict,
    extract_current_turn_messages,
    extract_last_ai_answer,
    extract_tool_contexts,
    extract_user_query,
    format_turn_trajectory,
)
from app.middleware.guardrails.guardrails import OutputSchemaRepairGuardrail
from app.middleware.memory.episodic_store import _records_from_messages
from app.middleware.memory.transcript_store import classify_message
from app.utils.message_markers import (
    EVALUATOR_FEEDBACK_TAG,
    ORIGIN_EVALUATOR_FEEDBACK,
    is_evaluator_feedback,
    is_user_turn_start,
    origin_kwargs,
)


class _StubStructuredJudge:
    """with_structured_output() 호출마다 시퀀스 순서대로 판정을 반환하는 테스트용 판사."""

    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls: List[Any] = []

    def with_structured_output(self, schema: Any):
        parent = self

        def _invoke(inp: Any, config: Any = None):
            parent.calls.append((schema, inp))
            if not parent._responses:
                raise RuntimeError("No stubbed judge responses left")
            resp = parent._responses.pop(0)
            return resp(inp) if callable(resp) else resp

        async def _ainvoke(inp: Any, config: Any = None):
            return _invoke(inp, config)

        stub = MagicMock()
        stub.invoke = _invoke
        stub.ainvoke = _ainvoke
        return stub


class _RepairSampleSchema(BaseModel):
    title: str
    score: int


def test_message_markers_and_transcript_isolation():
    """EvaluatorHarness 피드백은 사용자 턴 경계에서 제외되고 transcript/episodic에서 kind='internal'로 분류된다."""
    u1 = HumanMessage(content="첫 질문")
    a1 = AIMessage(content="부족한 초안")
    fb = HumanMessage(
        content=f"⚠️ {EVALUATOR_FEEDBACK_TAG} Rework Required",
        additional_kwargs=origin_kwargs(ORIGIN_EVALUATOR_FEEDBACK),
    )
    a2 = AIMessage(content="보완된 최종 답변")

    assert is_user_turn_start(u1) is True
    assert is_user_turn_start(fb) is False
    assert is_evaluator_feedback(fb) is True

    # 태그만으로 판별하는 하위 호환 경로 검증
    fb_by_tag = HumanMessage(content=f"⚠️ {EVALUATOR_FEEDBACK_TAG} tag-only fallback")
    assert is_evaluator_feedback(fb_by_tag) is True
    assert is_user_turn_start(fb_by_tag) is False

    assert classify_message(u1) == "user"
    assert classify_message(fb) == "internal"

    records = _records_from_messages([u1, a1, fb, a2])
    kinds = [r["kind"] for r in records]
    assert kinds == ["user", "assistant", "internal", "assistant"]

    dict_records = _records_from_messages([
        {"role": "user", "content": "첫 질문"},
        {"role": "human", "content": f"⚠️ {EVALUATOR_FEEDBACK_TAG} dict fallback"},
    ])
    assert [r["kind"] for r in dict_records] == ["user", "internal"]


def test_turn_slicing_and_trajectory_formatting():
    """멀티턴 대화에서 현재 턴만 추출하고 tool_calls를 누락 없이 포맷한다."""
    msgs = [
        HumanMessage(content="이전 턴 질문"),
        AIMessage(content="이전 턴 답변"),
        HumanMessage(content="현재 턴 질문"),
        AIMessage(content="", tool_calls=[{"name": "file_read", "args": {"file_path": "a.json"}, "id": "tc1"}]),
        ToolMessage(content='{"ok": true}', tool_call_id="tc1", name="file_read"),
        HumanMessage(
            content=f"⚠️ {EVALUATOR_FEEDBACK_TAG} 피드백",
            additional_kwargs=origin_kwargs(ORIGIN_EVALUATOR_FEEDBACK),
        ),
        AIMessage(content="현재 턴 최종 답변"),
    ]
    turn = extract_current_turn_messages(msgs)
    assert len(turn) == 5
    assert extract_user_query(msgs) == "현재 턴 질문"
    assert extract_last_ai_answer(msgs) == "현재 턴 최종 답변"
    assert extract_tool_contexts(msgs) == ['{"ok": true}']

    traj_txt = format_turn_trajectory(turn[:-1])
    assert "이전 턴 질문" not in traj_txt
    assert "file_read" in traj_txt
    assert '"file_path": "a.json"' in traj_txt


def test_evaluator_harness_rework_loop_and_circuit_breaker():
    """EvaluatorHarness(mode='rework') 반려 -> jump_to='model' 재작업 -> 승인 및 max_reworks 한도 검증."""
    with tempfile.TemporaryDirectory() as tmpdir:
        judge_llm = _StubStructuredJudge([
            JudgeVerdict(
                evidence="계산 근거 누락",
                reason="공식과 점수가 빠짐",
                is_approved=False,
                score=40,
                feedback="Value_Score 공식과 계산 점수를 포함하세요.",
            ),
            JudgeVerdict(
                evidence="공식과 50.95점 모두 포함됨",
                reason="요구사항 충족",
                is_approved=True,
                score=96,
                feedback="",
            ),
        ])
        ev = EvaluatorHarness(mode="rework", max_reworks=2, judge_llm=judge_llm, log_dir=tmpdir, verbose=False)
        fake_agent_llm = GenericFakeChatModel(messages=iter([
            AIMessage(content="LP-001이 1등입니다."),
            AIMessage(content="공식에 따른 LP-001 점수는 50.95점입니다."),
        ]))
        agent = create_agent(model=fake_agent_llm, tools=[], middleware=[ev])

        res = asyncio.run(agent.ainvoke(
            {"messages": [HumanMessage(content="LP-001 점수 보고")]},
            config={"configurable": {"thread_id": "test-rework-ok"}},
        ))
        assert "50.95점" in res["messages"][-1].content
        assert len(ev.verdict_history) == 2
        assert ev.verdict_history[0]["is_approved"] is False
        assert ev.verdict_history[1]["is_approved"] is True
        assert sum(1 for m in res["messages"] if is_user_turn_start(m)) == 1
        assert sum(1 for m in res["messages"] if is_evaluator_feedback(m)) == 1

        judge_always_reject = _StubStructuredJudge([
            JudgeVerdict(evidence="부족", reason="반려1", is_approved=False, score=30, feedback="고쳐주세요"),
            JudgeVerdict(evidence="여전히 부족", reason="반려2", is_approved=False, score=35, feedback="한도 초과"),
        ])
        ev_cap = EvaluatorHarness(mode="rework", max_reworks=1, judge_llm=judge_always_reject, log_dir=tmpdir, verbose=False)
        fake_agent_llm_2 = GenericFakeChatModel(messages=iter([
            AIMessage(content="초안 1"),
            AIMessage(content="재작성 2"),
        ]))
        agent_cap = create_agent(model=fake_agent_llm_2, tools=[], middleware=[ev_cap])
        res_cap = asyncio.run(agent_cap.ainvoke(
            {"messages": [HumanMessage(content="테스트")]},
            config={"configurable": {"thread_id": "test-rework-cap"}},
        ))
        assert res_cap["messages"][-1].content == "재작성 2"
        assert len(ev_cap.verdict_history) == 2


def test_evaluator_harness_shadow_mode_and_eval_runner_export():
    """EvaluatorHarness(mode='shadow') 비동기 JSONL 적재 및 EvalRunner 임포트 유지 검증."""
    assert EvalRunner is not None

    with tempfile.TemporaryDirectory() as tmpdir:
        judge_llm = _StubStructuredJudge([
            JudgeVerdict(
                evidence="답변 근거 미흡",
                reason="파일명 미인용 및 계산 누락",
                is_approved=False,
                score=55,
                feedback="근거 파일명을 명시하세요.",
            )
        ])
        shadow_ev = EvaluatorHarness(
            mode="shadow",
            sample_rate=1.0,
            judge_llm=judge_llm,
            log_dir=tmpdir,
            verbose=False,
        )
        fake_llm = GenericFakeChatModel(messages=iter([AIMessage(content="대략 100원입니다.")]))
        agent = create_agent(model=fake_llm, tools=[], middleware=[shadow_ev])

        async def _run():
            await agent.ainvoke(
                {"messages": [HumanMessage(content="laptops_market_data.json 요약해줘")]},
                config={"configurable": {"thread_id": "shadow-t1"}},
            )
            await shadow_ev.await_pending(timeout=5.0)

        asyncio.run(_run())

        stream_file = Path(tmpdir) / "eval_audit_stream.jsonl"
        assert stream_file.exists()
        lines = [json.loads(ln) for ln in stream_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1
        assert lines[0]["mode"] == "shadow"
        assert lines[0]["score"] == 55


def test_output_schema_repair_guardrail_tool_calls_skip_and_retry_preserves_content():
    """tool_calls 응답은 수선 건너뛰기(sync/async), repair() 재시도 시 원본 bad_content를 유지하는지 검증."""
    seen_bad_contents: List[str] = []
    attempt_replies = [
        '{"title": "initial", "score": "still_invalid"}',
        '{"title": "initial", "score": 95}',
    ]

    def _fake_repair_llm(prompt_val: Any, **_kwargs: Any) -> AIMessage:
        msgs = prompt_val.to_messages() if hasattr(prompt_val, "to_messages") else prompt_val
        human_msg = msgs[-1].content if msgs else ""
        seen_bad_contents.append(str(human_msg))
        return AIMessage(content=attempt_replies[len(seen_bad_contents) - 1])

    stub_llm = RunnableLambda(_fake_repair_llm)
    with patch("app.middleware.guardrails.guardrails.init_chat_model", return_value=stub_llm):
        guard = OutputSchemaRepairGuardrail(pydantic_schema=_RepairSampleSchema, max_retry=2)

    # 1) repair() 재시도 루프가 1차 실패 후에도 current_content를 에러 문자열로 덮어쓰지 않고 원본 보존
    bad_raw = '{"title": "initial", "score": "bad_int"}'
    repaired = guard.repair(bad_raw)
    assert json.loads(repaired) == {"title": "initial", "score": 95}
    assert len(seen_bad_contents) == 2
    assert bad_raw in seen_bad_contents[0]
    assert bad_raw in seen_bad_contents[1]

    # 2) wrap_model_call / awrap_model_call 모두 tool_calls 존재 시 수선 생략
    tool_msg_direct = AIMessage(
        content="calling tool...",
        tool_calls=[{"name": "file_read", "args": {"file_path": "x.py"}, "id": "tc1"}],
    )
    tool_msg_wrapped = SimpleNamespace(
        result=[
            AIMessage(
                content="calling wrapped tool...",
                tool_calls=[{"name": "file_read", "args": {"file_path": "y.py"}, "id": "tc2"}],
            )
        ]
    )

    with patch.object(guard, "repair", side_effect=AssertionError("repair should not run on tool_calls")), patch.object(
        guard, "arepair", side_effect=AssertionError("arepair should not run on tool_calls")
    ):
        res_direct = guard.wrap_model_call(SimpleNamespace(), lambda _req: tool_msg_direct)
        res_wrapped = guard.wrap_model_call(SimpleNamespace(), lambda _req: tool_msg_wrapped)
        assert res_direct is tool_msg_direct
        assert res_wrapped is tool_msg_wrapped

        async def _async_handler(_req: Any):
            return tool_msg_wrapped

        res_async = asyncio.run(guard.awrap_model_call(SimpleNamespace(), _async_handler))
        assert res_async is tool_msg_wrapped
