"""
tests/test_semantic_review_parsing.py — Semantic 백그라운드 리뷰 응답 파싱 회귀 테스트

리뷰 LLM은 모델마다 memory 작업을 다른 모양으로 답합니다.
- gpt 계열: memory(action="add", target="user", content="...")
- gemini 계열: ```json [{"action": "add", ...}] ```
두 모양 모두 MEMORY.md / USER.md에 반영되어야 합니다.
"""

import pytest

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
    assert store.memory_entries == ["Project runs on Python 3.12.3."]


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
    assert store.memory_entries == ["Project runs on Python 3.12.3."]


@pytest.mark.parametrize("payload", [
    '{"action": "add", "target": "user", "content": "User\'s editor is Vim."}',
    '{"operations": [{"action": "add", "target": "user", "content": "User\'s editor is Vim."}]}',
])
def test_bare_json_object_and_apostrophe(store_and_mw, payload):
    """코드 펜스 없는 JSON 객체도 받으며, 작은따옴표가 들어간 내용도 잘리지 않아야 합니다."""
    store, mw = store_and_mw
    mw._parse_and_execute_memory_ops(payload, store=store)
    assert store.user_entries == ["User's editor is Vim."]


@pytest.mark.parametrize("response", ["Nothing to save.", "```json\n[]\n```"])
def test_nothing_to_save(store_and_mw, response):
    store, mw = store_and_mw
    mw._parse_and_execute_memory_ops(response, store=store)
    assert store.user_entries == []
    assert store.memory_entries == ["Project runs on Python 3.12."]
