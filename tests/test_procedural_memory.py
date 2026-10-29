"""
tests/test_procedural_memory.py — Procedural Memory (학습 스킬) 회귀 테스트

LLM 없이 결정적으로 실행합니다 (GenericFakeChatModel 기반 ScriptedModel).
- ProceduralMemoryStore: 검증, 출처(created_by), 백그라운드 가드, read-before-write, Curator 생애주기
- MemoryMiddleware: 도구 호출 반복 카운터, skill_manage 리셋, pending 이월, 취소 시 카운터 보존
- run_skill_review: 리뷰 에이전트가 스킬을 생성/패치하고 결과를 요약
"""

import json
import threading
from datetime import timedelta

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

from app.context import AgentContext
from app.middleware.memory import (
    MemoryMiddleware,
    ProceduralMemoryStore,
    ReviewSession,
    SemanticMemoryStore,
    run_skill_review,
)
from app.middleware.memory.procedural_store import _now


class ScriptedModel(GenericFakeChatModel):
    """미리 정해진 AIMessage를 순서대로 반환하는 가짜 모델 (도구 바인딩 허용)."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*messages):
    return ScriptedModel(messages=iter([m if isinstance(m, AIMessage) else AIMessage(content=m) for m in messages]))


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def skill_md(name, description="Use when building a data report. Validate inputs first.", body="1. Step one."):
    return f"---\nname: {name}\ndescription: {description}\n---\n# Title\n## Procedure\n{body}\n"


@pytest.fixture
def store(tmp_path):
    s = ProceduralMemoryStore(skills_dir=str(tmp_path / "skills"))
    s.load_from_disk()
    return s


# =============================================================================
# 1. ProceduralMemoryStore — 검증과 출처
# =============================================================================

@pytest.mark.parametrize("name, content, fragment", [
    ("Bad Name", skill_md("Bad Name"), "Invalid skill name"),
    ("ok-name", "# no frontmatter\n", "frontmatter"),
    ("ok-name", "---\nname: ok-name\n---\nbody", "description"),
    ("ok-name", skill_md("other-name"), "must match"),
])
def test_create_validation_rejects_bad_input(store, name, content, fragment):
    result = store.create(name, content)
    assert not result["success"] and fragment in result["error"]


def test_create_rejects_oversized_skill(tmp_path):
    small = ProceduralMemoryStore(skills_dir=str(tmp_path / "s"), max_skill_chars=200)
    result = small.create("big", skill_md("big", body="x" * 500))
    assert not result["success"] and "references/" in result["error"]


def test_provenance_foreground_is_user_owned_background_is_agent(store):
    store.create("user-skill", skill_md("user-skill"))
    store.create("agent-skill", skill_md("agent-skill"), review=ReviewSession())
    assert not store.is_agent_managed("user-skill")
    assert store.is_agent_managed("agent-skill")


@pytest.mark.parametrize("bad_path", ["../escape.md", "notes/x.md", "/abs/x.md", "references", "C:/x.md"])
def test_write_file_rejects_unsafe_paths(store, bad_path):
    store.create("s", skill_md("s"))
    result = store.write_file("s", bad_path, "data")
    assert not result["success"]


def test_view_lists_linked_files_and_counts_foreground_use(store):
    store.create("s", skill_md("s"))
    store.write_file("s", "references/formats.md", "table rules")
    result = store.view("s")
    assert result["linked_files"] == {"references": ["references/formats.md"]}
    assert store.view("s", file_path="references/formats.md")["content"] == "table rules"
    record = store.load_usage()["s"]
    assert record["view_count"] == 2 and record["use_count"] == 2 and record["last_used_at"]


def test_catalog_is_frozen_until_reload(store):
    assert store.format_for_prompt() is None
    store.create("s", skill_md("s"))
    assert store.format_for_prompt() is None, "세션 중에는 카탈로그가 바뀌지 않아야 합니다"
    store.load_from_disk()
    block = store.format_for_prompt()
    assert "PROCEDURAL MEMORY" in block and "- s: Use when building a data report" in block


def test_patch_requires_unique_match_and_keeps_frontmatter(store):
    store.create("s", skill_md("s", body="1. Step.\n2. Step."))
    assert "matched 2 times" in store.patch("s", "Step.", "Done.")["error"]
    assert "structure" in store.patch("s", "name: s", "nam: s")["error"]
    assert store.patch("s", "1. Step.", "1. Load data.")["success"]
    assert "1. Load data." in store.view("s")["content"]


# =============================================================================
# 2. 백그라운드 가드 — 소유권, read-before-write, 보관
# =============================================================================

def test_background_cannot_touch_user_owned_skill(store):
    store.create("user-skill", skill_md("user-skill"))
    review = ReviewSession()
    store.view("user-skill", review=review)
    result = store.patch("user-skill", "1. Step one.", "1. Changed.", review=review)
    assert not result["success"] and "user-owned" in result["error"]
    assert not store.delete("user-skill", review=review)["success"]


def test_background_patch_requires_view_in_same_review(store):
    store.create("s", skill_md("s"), review=ReviewSession())
    stale_review = ReviewSession()
    refused = store.patch("s", "1. Step one.", "1. Changed.", review=stale_review)
    assert refused.get("read_before_write_required")

    store.view("s", review=stale_review)
    assert store.patch("s", "1. Step one.", "1. Changed.", review=stale_review)["success"]
    # 리뷰 중 열람은 사용 기록에 집계하지 않음 (Curator stale 판단이 리뷰 때문에 흐려지지 않도록)
    assert store.load_usage()["s"]["use_count"] == 0


def test_background_overwrite_of_support_file_requires_view(store):
    review = ReviewSession()
    store.create("s", skill_md("s"), review=review)
    assert store.write_file("s", "references/a.md", "v1", review=review)["success"], "새 파일은 읽기 불필요"
    assert store.write_file("s", "references/a.md", "v2", review=ReviewSession()).get("read_before_write_required")


def test_background_delete_archives_and_restore_brings_back(store):
    review = ReviewSession()
    store.create("s", skill_md("s"), review=review)
    assert store.delete("s", review=review)["success"]
    assert [x["name"] for x in store.list_skills()] == []
    assert store.load_usage()["s"]["state"] == "archived"
    assert store.restore("s")["success"]
    assert [x["name"] for x in store.list_skills()] == ["s"]


def test_foreground_delete_removes_skill(store):
    store.create("s", skill_md("s"))
    assert store.delete("s")["success"]
    assert "s" not in store.load_usage()
    assert not (store.skills_dir / ".archive" / "s").exists()


# =============================================================================
# 3. Curator — 사용 기록 기반 생애주기
# =============================================================================

def test_lifecycle_marks_stale_then_archives_only_agent_skills(store):
    review = ReviewSession()
    store.create("agent-skill", skill_md("agent-skill"), review=review)
    store.create("user-skill", skill_md("user-skill"))

    report = store.apply_lifecycle(now=_now() + timedelta(days=20))
    assert report == {"marked_stale": ["agent-skill"], "archived": []}

    store.view("agent-skill")  # 다시 쓰면 active로 복귀
    assert store.load_usage()["agent-skill"]["state"] == "active"

    report = store.apply_lifecycle(now=_now() + timedelta(days=40))
    assert report["archived"] == ["agent-skill"]
    assert [x["name"] for x in store.list_skills()] == ["user-skill"], "사용자 소유 스킬은 건드리지 않음"


# =============================================================================
# 4. MemoryMiddleware — 트리거 카운터
# =============================================================================

@tool
def lookup(query: str) -> str:
    """Look something up."""
    return f"result for {query}"


def _middleware(tmp_path, interval=2, review_llm=None):
    sem = SemanticMemoryStore(memory_dir=str(tmp_path / "mem"))
    sem.load_from_disk()
    proc = ProceduralMemoryStore(skills_dir=str(tmp_path / "skills"))
    proc.load_from_disk()
    return MemoryMiddleware(
        semantic_store=sem,
        episodic_store=None,
        review_llm=review_llm or scripted("Nothing to save."),
        transcript_enabled=False,
        finalize_idle_settle_s=3600.0,  # 테스트 중 자동 디스패치 방지
        procedural_store=proc,
        skill_nudge_interval=interval,
    )


def _ctx(session_id):
    return AgentContext(
        session_id=session_id,
        semantic_learning_enabled=False,
        episodic_memory_enabled=False,
        transcript_enabled=False,
    )


def _run(mw, model, session_id, text="go"):
    agent = create_agent(model=model, tools=[lookup], middleware=[mw], checkpointer=MemorySaver(),
                         context_schema=AgentContext)
    agent.invoke({"messages": [HumanMessage(content=text)]},
                 config={"configurable": {"thread_id": session_id}}, context=_ctx(session_id))


def test_skill_tools_registered_and_catalog_injected(tmp_path):
    mw = _middleware(tmp_path)
    assert {"skill_view", "skill_manage"} <= {t.name for t in mw.tools}

    ctx = _ctx("s0")
    runtime = type("R", (), {"context": ctx, "config": {"configurable": {"thread_id": "s0"}}})()
    mw.before_agent({"messages": [HumanMessage(content="hi")]}, runtime)
    assert "PROCEDURAL MEMORY (learned skills) [0 skills]" in ctx.recalled_memory


def test_tool_iterations_reach_nudge_and_review_runs_at_turn_end(tmp_path):
    """도구 호출 응답이 기준값(2)에 닿은 턴이 끝나면 바로 스킬 리뷰가 돌고, 끝나면 카운터가 0으로."""
    mw = _middleware(tmp_path, interval=2)
    model = scripted(
        call("lookup", {"query": "a"}, "c1"),
        call("lookup", {"query": "b"}, "c2"),
        "done",
    )
    _run(mw, model, "sess")
    mw.wait_for_reviews(timeout=5.0)
    assert mw.last_skill_reviews["sess"]["reply"] == "Nothing to save."
    assert mw._skill_iters["sess"] == 0 and "sess" not in mw._skill_review_pending


def test_pending_skill_review_sticks_until_completed(tmp_path):
    """기준값 도달 후 리뷰가 새 턴 때문에 취소된 세션은, 카운터가 0이어도 다음 턴 종료 때 다시 리뷰합니다."""
    mw = _middleware(tmp_path, interval=2)
    mw._skill_review_pending.add("sess")
    assert mw._skill_nudge_reached("sess")
    assert mw._background_review_due(_ctx("sess"), "sess", True)["procedural"] is True


def test_parallel_tool_calls_count_as_one_iteration(tmp_path):
    mw = _middleware(tmp_path, interval=5)
    parallel = AIMessage(content="", tool_calls=[
        {"name": "lookup", "args": {"query": "a"}, "id": "p1", "type": "tool_call"},
        {"name": "lookup", "args": {"query": "b"}, "id": "p2", "type": "tool_call"},
    ])
    _run(mw, scripted(parallel, "done"), "par")
    assert mw._skill_iters["par"] == 1


def test_foreground_skill_manage_resets_counter(tmp_path):
    mw = _middleware(tmp_path, interval=10)
    model = scripted(
        call("lookup", {"query": "a"}, "c1"),
        call("skill_manage", {"action": "create", "name": "report", "content": skill_md("report")}, "c2"),
        "saved",
    )
    _run(mw, model, "fg")
    # skill_manage 호출 응답도 반복 1회로 세지만, 도구 실행 시점에 0으로 리셋됨
    assert mw._skill_iters["fg"] == 0
    assert not mw.procedural_store.is_agent_managed("report"), "사용자 지시로 만든 스킬은 사용자 소유"


def test_cancelled_skill_review_keeps_counter(tmp_path):
    mw = _middleware(tmp_path, interval=1)
    mw._skill_iters["c"] = 3
    mw._skill_review_pending.add("c")
    cancel = threading.Event()
    cancel.set()
    result = mw._review_procedural_memory([{"role": "user", "content": "x"}], "c", cancel_event=cancel)
    assert result["cancelled"]
    assert mw._skill_iters["c"] == 3 and "c" in mw._skill_review_pending


def test_procedural_switch_off_hides_tools_and_skips_review(tmp_path):
    mw = _middleware(tmp_path, interval=1)
    ctx = AgentContext(procedural_memory_enabled=False)
    runtime = type("R", (), {"context": ctx})()
    assert {"skill_view", "skill_manage"} <= mw._disabled_tool_names(runtime)
    mw._skill_iters["off"] = 5
    assert mw._background_review_due(ctx, "off", True)["procedural"] is False


# =============================================================================
# 5. run_skill_review — 리뷰 에이전트
# =============================================================================

TRANSCRIPT = [
    {"role": "user", "content": "Build the integrity report from sales.csv"},
    {"role": "assistant", "content": "", "tool_calls": [{"name": "file_read", "args": {"path": "sales.csv"}}]},
    {"role": "tool", "tool_name": "file_read", "content": "parse error: corrupted"},
    {"role": "user", "content": "Use the backup file when the main one is corrupted, and put the conclusion first."},
    {"role": "assistant", "content": "Done with backup; conclusion first."},
]


def test_review_creates_agent_owned_skill(store):
    llm = scripted(
        call("skills_list", {}, "r1"),
        call("skill_manage", {"action": "create", "name": "data-integrity-report",
                              "content": skill_md("data-integrity-report")}, "r2"),
        "Created data-integrity-report.",
    )
    result = run_skill_review(llm, store, TRANSCRIPT)
    assert result["actions"] == ["Skill 'data-integrity-report' created"]
    assert store.is_agent_managed("data-integrity-report")
    assert result["reply"] == "Created data-integrity-report."


def test_review_patch_is_refused_until_viewed(store):
    store.create("data-integrity-report", skill_md("data-integrity-report"), review=ReviewSession())
    patch_args = {"action": "patch", "name": "data-integrity-report",
                  "old_string": "1. Step one.", "new_string": "1. Fall back to the backup file."}
    llm = scripted(
        call("skill_manage", patch_args, "r1"),                       # 읽기 전 → 거부
        call("skill_view", {"name": "data-integrity-report"}, "r2"),  # 열람
        call("skill_manage", patch_args, "r3"),                       # 재시도 → 성공
        "Patched.",
    )
    result = run_skill_review(llm, store, TRANSCRIPT)
    assert result["actions"] == ["Skill 'data-integrity-report' patched"]
    assert "backup file" in (store.skills_dir / "data-integrity-report" / "SKILL.md").read_text(encoding="utf-8")


def test_review_keeps_partial_actions_when_iteration_limit_hit(store):
    """반복 상한에 걸려 중단돼도, 그 전에 반영된 쓰기는 결과와 trace에 남아야 합니다."""
    llm = scripted(
        call("skill_manage", {"action": "create", "name": "report", "content": skill_md("report")}, "r1"),
        *[call("skill_view", {"name": "report"}, f"v{i}") for i in range(10)],
    )
    result = run_skill_review(llm, store, TRANSCRIPT, max_iterations=3)
    assert result["stopped_early"]
    assert result["actions"] == ["Skill 'report' created"]
    assert result["trace"][0] == "skill_manage ✓"


def test_review_nothing_to_save(store):
    result = run_skill_review(scripted("Nothing to save."), store, TRANSCRIPT)
    assert result["actions"] == [] and result["reply"] == "Nothing to save."
