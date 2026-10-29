"""
Claude Code 압축 파이프라인 (app/middleware/compaction/compactor.py) 테스트.

단계별 단위 테스트 + create_agent 그래프 통합 테스트.
"""

import asyncio
import os
import time

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from app.middleware.compaction.compactor import (
    PERSISTED_OUTPUT_TAG,
    PTL_RETRY_MARKER,
    TIME_BASED_MC_CLEARED_MESSAGE,
    AutoCompactor,
    ContextCollapse,
    MicroCompactor,
    ReactiveCompactor,
    SnipCompactor,
    ToolResultBudget,
    claude_code_autocompact_threshold,
    create_compactor_middleware,
    format_compact_summary,
    group_messages_by_api_round,
    truncate_head_for_ptl_retry,
)
from app.utils.message_markers import (
    ORIGIN_COMPACTION_MARKER,
    ORIGIN_CONTEXT_COLLAPSE,
    ORIGIN_SNIP_BOUNDARY,
    is_compaction_summary,
    message_origin,
)


class ScriptedModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*messages):
    return ScriptedModel(messages=iter([m if isinstance(m, AIMessage) else AIMessage(content=m) for m in messages]))


class RecordingLLM:
    """요약용 가짜 LLM: 받은 프롬프트를 기록하고, 지정한 예외를 순서대로 던진 뒤 응답합니다."""

    def __init__(self, text="<analysis>draft</analysis>\n<summary>\n1. Primary Request and Intent: build API\n</summary>",
                 errors=()):
        self.text = text
        self.errors = list(errors)
        self.prompts = []

    def invoke(self, messages, *args, **kwargs):
        self.prompts.append(messages[-1].content)
        if self.errors:
            raise self.errors.pop(0)
        return AIMessage(content=self.text)

    def get_num_tokens_from_messages(self, messages):
        return sum(len(str(m.content)) for m in messages) // 4


def tool_round(call_id, name, result, ai_id=None, result_id=None):
    return [
        AIMessage(content="", id=ai_id, tool_calls=[{"name": name, "args": {}, "id": call_id}]),
        ToolMessage(content=result, tool_call_id=call_id, name=name, id=result_id),
    ]


# =============================================================================
# 0. ToolResultBudget
# =============================================================================
def test_tool_result_budget_persists_large_result_with_preview(tmp_path):
    budget = ToolResultBudget(storage_dir=str(tmp_path), max_result_chars=100, preview_chars=30)
    msg = ToolMessage(content="line\n" * 100, tool_call_id="call_1", name="bash_command", id="m1")

    out = budget.apply_to_tool_message(msg)

    assert out.id == "m1" and out.tool_call_id == "call_1"
    assert out.content.startswith(PERSISTED_OUTPUT_TAG)
    assert "Output too large" in out.content and "Preview (first 30 chars)" in out.content
    saved = list(tmp_path.rglob("call_1.txt"))
    assert len(saved) == 1 and saved[0].read_text(encoding="utf-8") == "line\n" * 100
    # 이미 저장된 결과는 다시 처리하지 않음
    assert budget.apply_to_tool_message(out) is out


def test_tool_result_budget_never_persists_file_read(tmp_path):
    """원본 Read 도구처럼 file_read 결과는 저장하지 않습니다 (저장 파일을 다시 읽는 순환 방지)."""
    budget = ToolResultBudget(storage_dir=str(tmp_path), max_result_chars=10)
    msg = ToolMessage(content="x" * 1000, tool_call_id="r1", name="file_read")
    assert budget.apply_to_tool_message(msg) is msg
    assert not list(tmp_path.rglob("*.txt"))


def test_tool_result_budget_round_limit_persists_largest_first(tmp_path):
    budget = ToolResultBudget(storage_dir=str(tmp_path), max_result_chars=10**9, per_message_chars=2500,
                              preview_chars=100)
    messages = [
        HumanMessage(content="q"),
        AIMessage(content="", tool_calls=[{"name": "web_fetch", "args": {}, "id": i} for i in ("a", "b", "c")]),
        ToolMessage(content="A" * 1000, tool_call_id="a", name="web_fetch", id="ta"),
        ToolMessage(content="B" * 3000, tool_call_id="b", name="web_fetch", id="tb"),
        ToolMessage(content="C" * 1200, tool_call_id="c", name="web_fetch", id="tc"),
    ]
    replaced = budget.enforce_round_budget(messages)
    assert [m.id for m in replaced] == ["tb"], "가장 큰 결과 하나만 저장하면 예산 안으로 들어옵니다"
    assert budget.enforce_round_budget(messages[:2]) == []


# =============================================================================
# 1. SnipCompactor
# =============================================================================
def _snip_messages():
    return [
        SystemMessage(content="sys", id="sys-00000000"),
        HumanMessage(content="old question", id="hum-00000001"),
        *tool_round("c1", "file_read", "OLD FILE", ai_id="ai0-00000002", result_id="too-00000003"),
        AIMessage(content="old answer", id="ai1-00000004"),
        HumanMessage(content="current question", id="hum-00000005"),
        *tool_round("c2", "grep_search", "NEW RESULT", ai_id="ai2-00000006", result_id="too-00000007"),
    ]


def test_snip_plan_keeps_tool_pairs_and_protects_current_turn():
    snip = SnipCompactor()
    messages = _snip_messages()

    # ToolMessage만 지목해도 그 결과를 부른 AIMessage까지 함께 지움
    ids, error = snip.plan_snip(messages, "00000003")
    assert error is None and ids == ["ai0-00000002", "too-00000003"]

    # 범위 끝이 현재 턴을 넘으면 현재 턴 직전에서 멈춤
    ids, _ = snip.plan_snip(messages, "[id:00000001]", "00000007")
    assert ids == ["hum-00000001", "ai0-00000002", "too-00000003", "ai1-00000004"]

    assert snip.plan_snip(messages, "00000006")[1] == "The current user turn cannot be snipped."
    assert snip.plan_snip(messages, "zzzzzzzz")[1].startswith("Unknown message id")


def test_snip_projection_tags_and_carries_claude_md_block():
    snip = SnipCompactor()
    messages = _snip_messages()
    messages[1] = messages[1].model_copy(update={"content": "<system-reminder>\nRULES\n</system-reminder>\n\nold question"})

    view = snip.project(messages, ["hum-00000001", "ai0-00000002", "too-00000003", "ai1-00000004"])
    assert [message_origin(m) for m in view[:2]] == [None, ORIGIN_SNIP_BOUNDARY]   # sys, boundary
    assert view[1].content.startswith("<system-reminder>"), "빠진 첫 메시지의 CLAUDE.md 블록은 경계 표시로 옮겨야 합니다"
    assert "4 earlier message(s)" in view[1].content
    assert len(view) == 5

    tagged = snip.tag_messages(view)
    assert tagged[2].content.endswith("[id:00000005]")      # 실제 사용자 메시지
    assert tagged[4].content.endswith("[id:00000007]")      # 도구 결과
    assert "[id:" not in tagged[1].content                  # 하네스 메시지에는 태그 없음
    assert messages[5].content == "current question", "태그는 사본에만 붙어야 합니다"


def test_snip_nudge_threshold():
    snip = SnipCompactor(nudge_ratio=0.7)
    assert snip.nudge(600, 1000) is None
    assert "70%" in snip.nudge(700, 1000).content


# =============================================================================
# 2. MicroCompactor (time-based)
# =============================================================================
def test_time_based_microcompact_trigger_and_keep_recent():
    micro = MicroCompactor(gap_threshold_minutes=60, keep_recent=2)
    now = time.time()
    assert micro.evaluate_trigger(now - 59 * 60, now) is None
    assert micro.evaluate_trigger(now - 61 * 60, now) == pytest.approx(61)
    assert micro.evaluate_trigger(None, now) is None

    messages = [HumanMessage(content="q")]
    for i in range(4):
        messages += tool_round(f"c{i}", "bash_command", f"OUT{i} " * 50, result_id=f"t{i}")
    messages += tool_round("w", "roll_dice", "6", result_id="tw")   # 대상 도구가 아니면 비우지 않음

    replaced, saved = micro.clear_old_results(messages)
    assert [m.id for m in replaced] == ["t0", "t1"]
    assert all(m.content == TIME_BASED_MC_CLEARED_MESSAGE for m in replaced)
    assert saved > 0


# =============================================================================
# 3. ContextCollapse
# =============================================================================
def test_context_collapse_commit_and_projection_keep_originals():
    llm = RecordingLLM(text="turn 1 summary")
    collapse = ContextCollapse(llm=llm)
    messages = _snip_messages()

    span = collapse.next_span(messages, [])
    assert [m.id for m in span] == ["hum-00000001", "ai0-00000002", "too-00000003", "ai1-00000004"]
    commit = collapse.commit(span)
    assert commit["summary"] == "turn 1 summary" and "OLD FILE" in llm.prompts[0]

    view = collapse.project(messages, [commit])
    assert [message_origin(m) for m in view].count(ORIGIN_CONTEXT_COLLAPSE) == 1
    assert len(view) == 1 + 1 + 3   # sys + 접힌 요약 + 현재 턴
    assert len(messages) == 8, "원본 메시지 목록은 그대로"


# =============================================================================
# 4. AutoCompactor
# =============================================================================
def _three_turns():
    return [
        SystemMessage(content="System Rules"),
        HumanMessage(content="Turn 1"), AIMessage(content="Ans 1"),
        HumanMessage(content="Turn 2"), AIMessage(content="Ans 2"),
        HumanMessage(content="Turn 3"),
    ]


def test_auto_compactor_uses_nine_section_prompt_and_strips_analysis():
    llm = RecordingLLM()
    compactor = AutoCompactor(llm=llm, threshold_tokens=1)
    compacted, ok = compactor.compact_if_needed(_three_turns())

    assert ok
    prompt = llm.prompts[0]
    assert "9. Optional Next Step" in prompt and "Do NOT call any tools" in prompt
    assert "Turn 1" in prompt and "Turn 3" not in prompt, "현재 턴은 요약하지 않고 원본으로 남깁니다"

    summary = compacted[1]
    assert is_compaction_summary(summary)
    assert "draft" not in summary.content and "Summary:\n1. Primary Request and Intent" in summary.content
    assert "Recent messages are preserved verbatim." in summary.content
    assert [m.content for m in compacted[2:]] == ["Turn 3"]


def test_format_compact_summary_without_tags_is_unchanged():
    assert format_compact_summary("1. Goal\n\n\n2. Next") == "1. Goal\n\n2. Next"


def test_auto_compactor_failure_keeps_conversation_and_trips_breaker():
    llm = RecordingLLM(errors=[RuntimeError("timeout")] * 3)
    compactor = AutoCompactor(llm=llm, threshold_tokens=1, max_consecutive_failures=3)
    messages = _three_turns()

    for expected in (1, 2, 3):
        result, ok = compactor.compact_if_needed(messages, force=True)
        assert not ok and result == messages, "요약에 실패하면 대화를 바꾸지 않습니다"
        assert compactor.consecutive_failures == expected
    assert compactor.is_circuit_open
    assert compactor.compact_if_needed(messages, force=True) == (messages, False)
    assert len(llm.prompts) == 3, "차단기가 열린 뒤에는 요약 LLM을 부르지 않습니다"

    compactor.reset_circuit_breaker()
    assert not compactor.is_circuit_open


def test_auto_compactor_ptl_retry_drops_oldest_api_rounds():
    llm = RecordingLLM(errors=[RuntimeError("prompt is too long: 250000 tokens > 200000")])
    compactor = AutoCompactor(llm=llm, threshold_tokens=1)
    messages = [HumanMessage(content="Turn 1")]
    for i in range(9):
        messages += tool_round(f"c{i}", "bash_command", f"ROUND-{i}")
    messages += [AIMessage(content="Ans 1"), HumanMessage(content="Turn 2")]

    _, ok = compactor.compact_if_needed(messages)
    assert ok and len(llm.prompts) == 2
    # 요약 대상 11개 그룹(첫 질문 + 도구 라운드 9 + 답변) 중 20%인 2개를 앞에서부터 버림
    assert "Turn 1" in llm.prompts[0] and "Turn 1" not in llm.prompts[1]
    assert "ROUND-0" not in llm.prompts[1] and "ROUND-1" in llm.prompts[1]
    assert PTL_RETRY_MARKER in llm.prompts[1]


def test_truncate_head_for_ptl_retry_keeps_pairs_and_marker():
    messages = [HumanMessage(content="q")] + tool_round("c1", "file_read", "r1") + tool_round("c2", "file_read", "r2")
    groups = group_messages_by_api_round(messages)
    assert [len(g) for g in groups] == [1, 2, 2]

    sliced = truncate_head_for_ptl_retry(messages)
    assert sliced[0].content == PTL_RETRY_MARKER and message_origin(sliced[0]) == ORIGIN_COMPACTION_MARKER
    assert isinstance(sliced[1], AIMessage) and isinstance(sliced[2], ToolMessage)
    assert truncate_head_for_ptl_retry([HumanMessage(content="only")]) is None


def _long_single_turn(rounds=6, size=400):
    messages = [SystemMessage(content="System Rules"), HumanMessage(content="BIG-TASK request")]
    for i in range(rounds):
        messages += tool_round(f"c{i}", "bash_command", f"OUT-{i} " + "y" * size)
    return messages


def test_auto_compactor_summarizes_current_turn_without_prior_turns():
    """원본 compactConversation처럼, 요약할 이전 턴이 없으면 현재 턴까지 요약합니다."""
    llm = RecordingLLM()
    compacted, ok = AutoCompactor(llm=llm, threshold_tokens=100).compact_if_needed(_long_single_turn())

    assert ok and "BIG-TASK request" in llm.prompts[0] and "OUT-5" in llm.prompts[0]
    assert isinstance(compacted[0], SystemMessage) and is_compaction_summary(compacted[1])
    assert len(compacted) == 2, "남겨 두는 메시지가 없습니다"
    assert "Recent messages are preserved verbatim." not in compacted[1].content


def test_auto_compactor_includes_current_turn_when_it_alone_exceeds_threshold():
    llm = RecordingLLM()
    messages = _three_turns()[:-1] + _long_single_turn()[1:]
    compacted, ok = AutoCompactor(llm=llm, threshold_tokens=100).compact_if_needed(messages)

    assert ok and len(llm.prompts) == 1, "이전 턴 요약 뒤 곧바로 다시 요약하지 않고 한 번에 처리"
    assert "Turn 1" in llm.prompts[0] and "BIG-TASK request" in llm.prompts[0]
    assert len(compacted) == 2


def test_middleware_compacts_long_single_turn_once(tmp_path):
    compactor = create_compactor_middleware(llm=RecordingLLM(), threshold_tokens=100, swap_dir=str(tmp_path),
                                            config={"micro": {"enabled": False}})
    update = compactor.before_model({"messages": _long_single_turn()[1:]})
    assert update is not None
    state_after = update["messages"][1:]   # [0]은 RemoveMessage(REMOVE_ALL_MESSAGES)
    assert is_compaction_summary(state_after[0])
    assert compactor.before_model({"messages": state_after}) is None, "요약·복원 블록만 남으면 다시 압축하지 않습니다"


def test_auto_compactor_empty_summary_is_a_failure():
    """원본은 요약 텍스트가 없으면 오류로 처리합니다. 빈 요약으로 대화를 덮어쓰면 안 됩니다."""
    compactor = AutoCompactor(llm=RecordingLLM(text=""), threshold_tokens=1)
    messages = _three_turns()
    result, ok = compactor.compact_if_needed(messages)
    assert not ok and result == messages and compactor.consecutive_failures == 1


def test_claude_code_threshold_formula():
    assert claude_code_autocompact_threshold(200_000) == 167_000   # 83.5%


# =============================================================================
# 5. ReactiveCompactor
# =============================================================================
def test_reactive_handle_overflow_drops_oldest_rounds():
    sliced = ReactiveCompactor().handle_overflow(_three_turns() + [AIMessage(content="Ans 3"), HumanMessage(content="Turn 4")])
    contents = [m.content for m in sliced]
    assert "Turn 1" not in contents and "Turn 4" in contents
    assert isinstance(sliced[0], SystemMessage)
    assert message_origin(sliced[1]) == ORIGIN_COMPACTION_MARKER


def test_reactive_handle_overflow_keeps_current_request():
    """한 요청으로 이어지는 긴 작업에서 앞부분을 잘라도 사용자 요청은 남습니다."""
    sliced = ReactiveCompactor().handle_overflow(_long_single_turn(rounds=10))
    contents = [str(m.content) for m in sliced]
    assert message_origin(sliced[1]) == ORIGIN_COMPACTION_MARKER
    assert contents[2] == "BIG-TASK request"
    assert isinstance(sliced[3], AIMessage) and not any("OUT-0" in c for c in contents)


def test_reactive_recover_summarizes_single_turn():
    reactive = ReactiveCompactor(auto_compactor=AutoCompactor(llm=RecordingLLM(), threshold_tokens=10**9))
    recovered = reactive.recover(_long_single_turn())
    assert is_compaction_summary(recovered[1]) and len(recovered) == 2


def test_model_fallback_propagates_same_overflow_errors_as_compactor():
    from app.middleware.compaction.compactor import is_prompt_too_long_error
    from app.middleware.error_control.self_recovery.model_fallback import _is_context_overflow

    for text in ("Error code: 400 - {'error': {'message': 'prompt is too long: 215000 tokens > 200000 maximum'}}",
                 "request too large", "context_length_exceeded"):
        assert is_prompt_too_long_error(Exception(text)) and _is_context_overflow(Exception(text))


def test_reactive_recover_prefers_summary():
    reactive = ReactiveCompactor(auto_compactor=AutoCompactor(llm=RecordingLLM(), threshold_tokens=10**9))
    recovered = reactive.recover(_three_turns())
    assert is_compaction_summary(recovered[1]) and recovered[-1].content == "Turn 3"


# =============================================================================
# 6. CompactorMiddleware in a real graph
# =============================================================================
class Inspect(AgentMiddleware):
    def __init__(self):
        self.inputs = []

    def wrap_model_call(self, request, handler):
        self.inputs.append(list(request.messages))
        return handler(request)


def _agent(model, middleware, tools=()):
    return create_agent(model=model, tools=list(tools), middleware=middleware, checkpointer=MemorySaver())


def test_middleware_snip_tool_end_to_end(tmp_path):
    inspect = Inspect()
    compactor = create_compactor_middleware(llm=RecordingLLM(), threshold_tokens=10**9, swap_dir=str(tmp_path))
    model = scripted("첫 답변")
    agent = _agent(model, [compactor, inspect])
    cfg = {"configurable": {"thread_id": "snip"}}

    out1 = agent.invoke({"messages": [HumanMessage(content="첫 질문")]}, cfg)
    first_id = SnipCompactor.short_id(out1["messages"][0].id)
    assert f"[id:{first_id}]" in str(inspect.inputs[-1][0].content)

    model.messages = iter([
        AIMessage(content="", tool_calls=[{"name": "snip", "args": {"start_id": first_id,
                                                                     "end_id": SnipCompactor.short_id(out1["messages"][1].id)},
                                           "id": "s1"}]),
        AIMessage(content="정리 완료"),
    ])
    out2 = agent.invoke({"messages": [HumanMessage(content="이전 대화는 지워도 돼")]}, cfg)

    assert out2["snipped_message_ids"] == [m.id for m in out1["messages"][:2]]
    assert out2["messages"][0].content == "첫 질문", "state에는 원본이 남습니다"
    last_input = inspect.inputs[-1]
    assert message_origin(last_input[0]) == ORIGIN_SNIP_BOUNDARY
    assert all("첫 질문" not in str(m.content) for m in last_input)


def test_middleware_time_based_microcompact_updates_state_in_place(tmp_path):
    @tool
    def bash_command(command: str) -> str:
        """Run a shell command."""
        return f"output of {command} " * 20

    compactor = create_compactor_middleware(
        llm=RecordingLLM(), threshold_tokens=10**9, swap_dir=str(tmp_path),
        config={"micro": {"keep_recent": 1}},
    )
    model = scripted(
        AIMessage(content="", tool_calls=[{"name": "bash_command", "args": {"command": "a"}, "id": "c1"}]),
        AIMessage(content="", tool_calls=[{"name": "bash_command", "args": {"command": "b"}, "id": "c2"}]),
        "done",
        "다음 턴 답변",
    )
    agent = _agent(model, [compactor], tools=[bash_command])
    cfg = {"configurable": {"thread_id": "micro"}}
    out1 = agent.invoke({"messages": [HumanMessage(content="run a and b")]}, cfg)
    ids_before = [m.id for m in out1["messages"]]

    agent.update_state(cfg, {"last_assistant_at": time.time() - 2 * 3600})   # 2시간 자리를 비움
    out2 = agent.invoke({"messages": [HumanMessage(content="계속")]}, cfg)

    tool_msgs = [m for m in out2["messages"] if isinstance(m, ToolMessage)]
    assert tool_msgs[0].content == TIME_BASED_MC_CLEARED_MESSAGE
    assert "output of b" in tool_msgs[1].content, "최근 keep_recent개는 남깁니다"
    assert [m.id for m in out2["messages"]][:len(ids_before)] == ids_before, "같은 id로 교체 (중복 없음)"


def test_middleware_collapse_suppresses_auto_compact(tmp_path):
    inspect = Inspect()
    summary_llm = RecordingLLM(text="collapsed turn")
    compactor = create_compactor_middleware(
        llm=summary_llm, threshold_tokens=40, swap_dir=str(tmp_path),
        config={"collapse": {"enabled": True}, "snip": {"enabled": False}},
    )
    model = scripted("첫 답변 " * 30, "둘째 답변")
    agent = _agent(model, [compactor, inspect])
    cfg = {"configurable": {"thread_id": "collapse"}}
    agent.invoke({"messages": [HumanMessage(content="첫 질문 " * 30)]}, cfg)
    out = agent.invoke({"messages": [HumanMessage(content="둘째 질문")]}, cfg)

    assert len(out["collapse_commits"]) == 1
    assert not any(is_compaction_summary(m) for m in out["messages"]), "Collapse가 켜지면 Auto-Compact는 발동하지 않습니다"
    assert str(out["messages"][0].content).startswith("첫 질문"), "원본은 state에 남습니다"
    origins = [message_origin(m) for m in inspect.inputs[-1]]
    assert origins.count(ORIGIN_CONTEXT_COLLAPSE) == 1
    assert "COLLAPSED" not in summary_llm.prompts[0]


def test_middleware_persists_large_tool_result_in_state(tmp_path):
    @tool
    def web_fetch(url: str) -> str:
        """Fetch a URL."""
        return "<html>" + "x" * 60_000

    compactor = create_compactor_middleware(llm=RecordingLLM(), threshold_tokens=10**9, swap_dir=str(tmp_path))
    model = scripted(AIMessage(content="", tool_calls=[{"name": "web_fetch", "args": {"url": "u"}, "id": "w1"}]), "ok")
    out = _agent(model, [compactor], tools=[web_fetch]).invoke(
        {"messages": [HumanMessage(content="fetch")]}, {"configurable": {"thread_id": "budget"}})

    result = next(m for m in out["messages"] if isinstance(m, ToolMessage))
    assert result.content.startswith(PERSISTED_OUTPUT_TAG) and len(result.content) < 3_000
    assert (tmp_path / "budget" / "w1.txt").read_text(encoding="utf-8").startswith("<html>")


def test_auto_compact_summary_is_not_streamed_to_user(tmp_path):
    """서버(app/server.py)는 astream_events의 모델 토큰 중 exclude_from_stream 태그가 없는 것을 답변으로 보냅니다.
    before_model에서 만드는 압축 요약 토큰이 사용자 답변에 섞이면 안 됩니다."""
    summary_llm = scripted("<analysis>INTERNAL-SUMMARY</analysis>\n<summary>\n1. Primary Request and Intent: x\n</summary>")
    compactor = create_compactor_middleware(
        llm=summary_llm, threshold_tokens=40, swap_dir=str(tmp_path), config={"snip": {"enabled": False}},
    )
    agent = _agent(scripted("첫 답변 " * 30, "둘째 답변"), [compactor])
    cfg = {"configurable": {"thread_id": "stream-tag"}}
    agent.invoke({"messages": [HumanMessage(content="첫 질문 " * 30)]}, cfg)

    async def streamed_to_user():
        tokens = []
        async for ev in agent.astream_events({"messages": [HumanMessage(content="둘째 질문")]}, cfg, version="v2"):
            if ev["event"] == "on_chat_model_stream" and "exclude_from_stream" not in ev.get("tags", []):
                tokens.append(str(ev["data"]["chunk"].content))
        return "".join(tokens)

    streamed = asyncio.run(streamed_to_user())
    assert any(is_compaction_summary(m) for m in agent.get_state(cfg).values["messages"]), "압축이 실제로 일어나야 함"
    assert "INTERNAL-SUMMARY" not in streamed
    assert "둘째 답변" in streamed
