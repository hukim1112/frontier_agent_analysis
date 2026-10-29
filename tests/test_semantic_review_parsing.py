"""
tests/test_semantic_review_parsing.py — Semantic 백그라운드 리뷰 응답 파싱 회귀 테스트

리뷰 LLM은 모델마다 memory 작업을 다른 모양으로 답합니다.
- gpt 계열: memory(action="add", target="user", content="...")
- gemini 계열: ```json [{"action": "add", ...}] ```
두 모양 모두 MEMORY.md / USER.md에 반영되어야 합니다.

백그라운드 리뷰 경로는 add만 바로 반영하고 replace/remove는 pending.json에 제안으로만 남깁니다
(Hermes _background_delete_gate).
"""

import threading
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.context import AgentContext
from app.middleware.memory import MemoryMiddleware, SemanticMemoryStore


@pytest.fixture
def store_and_mw(tmp_path):
    store = SemanticMemoryStore(memory_dir=str(tmp_path))
    store.load_from_disk()
    store.add("memory", "Project runs on Python 3.12.")
    mw = MemoryMiddleware(semantic_store=store, episodic_store=None)
    yield store, mw
    mw._review_queue.shutdown()


def test_function_call_format(store_and_mw):
    store, mw = store_and_mw
    mw._parse_and_execute_memory_ops(
        'memory(action="add", target="user", content="User prefers pytest.")\n'
        'memory(action="replace", target="memory", old_text="Python 3.12", content="Project runs on Python 3.12.3.")',
        store=store,
    )
    assert store.user_entries == ["User prefers pytest."]
    assert store.memory_entries == ["Project runs on Python 3.12."]          # replace는 반영되지 않음
    assert [(p["action"], p["old_text"]) for p in store.list_pending()] == [("replace", "Python 3.12")]


def test_json_code_fence_format(store_and_mw):
    """gemini-3.8-flash가 실제로 돌려준 모양 — 이전에는 아무것도 저장되지 않았음."""
    store, mw = store_and_mw
    mw._parse_and_execute_memory_ops(
        '```json\n[\n  {"action": "add", "target": "user", "content": "User prefers pytest."},\n'
        '  {"action": "replace", "target": "memory", "old_text": "Python 3.12", '
        '"content": "Project runs on Python 3.12.3."}\n]\n```',
        store=store,
    )
    assert store.user_entries == ["User prefers pytest."]
    assert store.memory_entries == ["Project runs on Python 3.12."]
    assert [p["action"] for p in store.list_pending()] == ["replace"]


def test_background_remove_is_staged_then_applied_or_discarded(store_and_mw):
    """무인 리뷰의 remove는 바로 지우지 않고 대기 → apply_pending으로만 반영됩니다."""
    store, mw = store_and_mw
    mw._parse_and_execute_memory_ops('memory(action="remove", target="memory", old_text="Python")', store=store)
    assert store.memory_entries == ["Project runs on Python 3.12."]
    (proposal,) = store.list_pending()

    mw._parse_and_execute_memory_ops('memory(action="remove", target="memory", old_text="3.12")', store=store)
    second = store.list_pending()[1]
    assert store.discard_pending(second["id"])["success"]

    assert store.apply_pending(proposal["id"])["success"]
    assert store.memory_entries == []
    assert store.list_pending() == []


@pytest.mark.parametrize("payload", [
    '{"action": "add", "target": "user", "content": "User\'s editor is Vim."}',
    '{"operations": [{"action": "add", "target": "user", "content": "User\'s editor is Vim."}]}',
])
def test_bare_json_object_and_apostrophe(store_and_mw, payload):
    """코드 펜스 없는 JSON 객체도 받으며, 작은따옴표가 들어간 내용도 잘리지 않아야 합니다."""
    store, mw = store_and_mw
    mw._parse_and_execute_memory_ops(payload, store=store)
    assert store.user_entries == ["User's editor is Vim."]


@pytest.mark.parametrize("response", [
    'memory(action="add", target="user", content="User\'s editor is Vim.")',
    "memory(action='add', target='user', content='User prefers \"short\" answers.')",
])
def test_function_call_format_with_inner_quotes(store_and_mw, response):
    """함수 호출 형태에서도 다른 종류의 따옴표가 내용에 들어 있으면 잘리지 않아야 합니다."""
    store, mw = store_and_mw
    mw._parse_and_execute_memory_ops(response, store=store)
    assert len(store.user_entries) == 1
    assert store.user_entries[0] in ("User's editor is Vim.", 'User prefers "short" answers.')


@pytest.mark.parametrize("response", ["Nothing to save.", "```json\n[]\n```"])
def test_nothing_to_save(store_and_mw, response):
    store, mw = store_and_mw
    mw._parse_and_execute_memory_ops(response, store=store)
    assert store.user_entries == []
    assert store.memory_entries == ["Project runs on Python 3.12."]


# ── Semantic 리뷰 트리거 (memory_nudge_interval) ──

def _runtime(session_id):
    ctx = AgentContext(session_id=session_id, user_id="default",
                       episodic_memory_enabled=False, transcript_enabled=False)
    return SimpleNamespace(context=ctx, config={"configurable": {"thread_id": session_id}})


def _turn(mw, rt, history, text):
    history.append(HumanMessage(content=text))
    mw.before_agent({"messages": history}, rt)
    history.append(AIMessage(content="ok"))
    mw.after_agent({"messages": history}, rt)


class _ReviewLLM:
    """Semantic 리뷰용 가짜 LLM. release가 설정될 때까지 응답을 미룰 수 있습니다 (취소 테스트용)."""

    def __init__(self):
        self.calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def invoke(self, prompt):
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=5)
        return AIMessage(content="Nothing to save.")


@pytest.fixture
def nudge_mw(tmp_path):
    store = SemanticMemoryStore(memory_dir=str(tmp_path))
    store.load_from_disk()
    mw = MemoryMiddleware(semantic_store=store, episodic_store=None, review_llm=_ReviewLLM(),
                          memory_nudge_interval=3, finalize_idle_settle_s=3600.0)
    yield mw
    mw._review_queue.shutdown()


def test_background_review_runs_at_turn_end_only_every_n_user_turns(nudge_mw):
    """nudge 조건(사용자 턴 3회)을 만족한 턴이 끝나면 바로 Semantic 리뷰가 돌고, 끝나면 카운터가 0으로."""
    rt, history = _runtime("s1"), []
    for i in range(6):
        _turn(nudge_mw, rt, history, f"turn {i}")
        nudge_mw.wait_for_reviews(timeout=5.0)
    assert nudge_mw.review_llm.calls == 2                      # 3턴째, 6턴째
    assert nudge_mw._memory_turns["s1"] == 0 and "s1" not in nudge_mw._memory_review_pending
    assert not nudge_mw._review_queue.pending("s1")            # 에피소드 추출은 꺼져 있어 예약 없음


def test_new_turn_cancels_running_review_and_retries_next_turn(nudge_mw):
    """리뷰 도중 사용자가 다시 말하면 리뷰를 멈추고, 기준값 도달 표시를 남겨 다음 턴 종료 때 다시 실행합니다."""
    llm, rt, history = nudge_mw.review_llm, _runtime("s2"), []
    for i in range(2):
        _turn(nudge_mw, rt, history, f"turn {i}")
    llm.release.clear()                                        # 3턴째 리뷰를 LLM 호출 중에 붙잡아 둠
    _turn(nudge_mw, rt, history, "turn 2")
    assert llm.started.wait(timeout=5.0)

    history.append(HumanMessage(content="turn 3"))
    nudge_mw.before_agent({"messages": history}, rt)           # 새 턴 시작 → 실행 중인 리뷰 취소
    llm.release.set()
    nudge_mw.wait_for_reviews(timeout=5.0)
    assert "s2" in nudge_mw._memory_review_pending              # 취소됐으므로 아직 미완료

    history.append(AIMessage(content="ok"))
    nudge_mw.after_agent({"messages": history}, rt)            # 이번 턴 종료 → 다시 실행
    nudge_mw.wait_for_reviews(timeout=5.0)
    assert llm.calls == 2 and "s2" not in nudge_mw._memory_review_pending


def test_memory_turn_counter_resumes_from_history_after_restart(nudge_mw):
    """처음 보는 세션은 (이전 사용자 턴 수 % 기준값)에서 시작합니다 (Hermes turn_context.py)."""
    history = [HumanMessage(content=f"old {i}") for i in range(4)]   # 이전 턴 4개 → 4 % 3 = 1
    nudge_mw.before_agent({"messages": history + [HumanMessage(content="new")]}, _runtime("s2"))
    assert nudge_mw._memory_turns["s2"] == 2


# ── 세션별 Frozen Snapshot ──

def test_snapshot_is_frozen_per_session_when_switching(nudge_mw):
    store = nudge_mw._get_semantic_store("default")
    rt_a, rt_b = _runtime("A"), _runtime("B")
    nudge_mw.before_agent({"messages": [HumanMessage(content="hi")]}, rt_a)

    store.add("memory", "Fact written during session A.")
    nudge_mw.before_agent({"messages": [HumanMessage(content="hi")]}, rt_b)
    assert "Fact written during session A." in rt_b.context.recalled_memory    # 새 세션은 최신 디스크

    nudge_mw.before_agent({"messages": [HumanMessage(content="back")]}, rt_a)
    assert "Fact written during session A." not in rt_a.context.recalled_memory  # A는 첫 턴 스냅샷 유지
