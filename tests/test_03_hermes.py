"""
Test Suite for Hermes 4-Layer Memory & Closed Learning Loop

실습 및 주피터 노트북에 주입하기 전 검증하는 파이썬 테스트 스크립트.
"""

import os
import sys
import tempfile
import asyncio
import pytest
from dotenv import load_dotenv

# 전역 프로젝트 루트 추가 및 .env 로드
sys.path.insert(0, ".")
load_dotenv()

from app.utils import get_llm
from app.utils.context import AgentContext
from app.utils.message_utils import normalize_content

from modules.hermes.memory_store import SemanticMemoryStore
from modules.hermes.session_store import EpisodicStore
from modules.hermes.memory_tool import create_memory_tool
from modules.hermes.session_search_tool import create_session_search_tool, create_session_recall_tool
from modules.hermes.memory_middleware import MemoryMiddleware
from modules.hermes.prompt_assembler import PromptAssembler


def test_01_l3_semantic_memory_generation_and_retrieval():
    """1. L3 Semantic Memory 생성 (직접 add + LLM 리뷰) 및 Frozen Snapshot 인출 테스트."""
    print("\n=== [Test 1] L3 Semantic Memory 생성 & 인출 테스트 ===")
    tmp_dir = tempfile.mkdtemp()
    
    # 1-1. 스토어 생성
    semantic_store = SemanticMemoryStore(memory_dir=tmp_dir)
    semantic_store.load_from_disk()
    
    # 1-2. 직접 생성 (Add)
    res1 = semantic_store.add("memory", "Project runs on Python 3.12 + LangChain 0.3.")
    res2 = semantic_store.add("user", "User prefers Korean responses with technical terms in English.")
    print(f"✅ [Direct Add] Memory: {res1['message']}, User: {res2['message']}")

    # 1-3. Closed Learning Loop (LLM 팩트 자동 추출)
    llm = get_llm(model_name="google_vertexai:gemini-3.5-flash", temperature=0.0)
    episodic_store = EpisodicStore(os.path.join(tmp_dir, "episodic.db"))
    middleware = MemoryMiddleware(semantic_store=semantic_store, episodic_store=episodic_store, review_llm=llm)

    l1_conversation = [
        {"role": "user", "content": "앞으로 파이썬 코드를 작성할 때는 단위 테스트로 zawsze pytest를 써줘."},
        {"role": "assistant", "content": "네, 알겠습니다! pytest를 기본 테스트 프레임워크로 기억하겠습니다."}
    ]
    
    # LLM 백그라운드 리뷰 실행 (XML / Function call 형태 파싱)
    middleware._review_semantic_memory(l1_conversation)

    # 1-4. 파일 및 엔트리 확인
    user_md_content = open(os.path.join(tmp_dir, "USER.md"), encoding="utf-8").read()
    print("\n📂 [USER.md 최종 내용]:")
    print(user_md_content)
    assert "pytest" in user_md_content, "LLM 추출 결과가 USER.md에 포함되어야 함!"

    # 1-5. Frozen Snapshot vs Reloaded Snapshot 비교
    snapshot_s1 = semantic_store.format_for_prompt("memory")
    print(f"📌 [Session 1 Frozen Snapshot]:\n{snapshot_s1}")
    assert snapshot_s1 is None, "Session 1 스냅샷은 세션 시작 시점(빈 상태)을 유지해야 함!"

    # Session 2 시작 (load_from_disk)
    session2_store = SemanticMemoryStore(memory_dir=tmp_dir)
    session2_store.load_from_disk()
    snapshot_s2 = session2_store.format_for_prompt("memory")
    print(f"\n📌 [Session 2 Reloaded Snapshot]:\n{snapshot_s2}")
    assert "Python 3.12" in snapshot_s2, "Session 2 스냅샷에는 저장된 팩트가 포함되어야 함!"
    print("✅ Test 1 Passed!")


@pytest.mark.asyncio
async def test_02_l2_episodic_memory():
    """2. L2 Episodic Memory (세션 DB 요약 + FTS5 검색 + Anchor Lineage 인출) 테스트."""
    print("\n=== [Test 2] L2 Episodic Memory 테스트 ===")
    tmp_dir = tempfile.mkdtemp()
    db_path = os.path.join(tmp_dir, "episodic.db")
    
    episodic_store = EpisodicStore(db_path=db_path)
    await episodic_store.setup()

    messages = [
        {"role": "user", "content": "안녕! 오늘 Hermes 4계층 메모리 아키텍처에 대해 논의하고 싶어."},
        {"role": "assistant", "content": "네! 어떤 메모리 계층부터 이야기를 시작할까요?"},
        {"role": "user", "content": "L2 에피소드 메모리의 SQLite FTS5 세션 검색 메커니즘을 설명해줘."},
        {"role": "assistant", "content": "L2 메모리는 세션 요약을 FTS5 테이블에 인덱싱하고 Anchor 기반 메시지를 인출합니다."},
        {"role": "user", "content": "정말 고마워!"}
    ]

    # 세션 종료 및 요약 생성
    llm = get_llm(model_name="google_vertexai:gemini-3.5-flash", temperature=0.0)
    summary = await episodic_store.finalize_session("session_101", messages, llm=llm)
    print(f"✅ [Session Finalize Summary]: {summary}")

    # FTS5 세션 검색
    results = await episodic_store.search_sessions(query="Hermes FTS5", top_k=2)
    print(f"✅ [FTS5 Search Results]: {results}")
    assert len(results) > 0, "FTS5 검색 결과가 존재해야 함!"

    # Anchor 기반 Lineage 인출
    anchored = await episodic_store.get_anchored_view("session_101", anchor_keyword="FTS5", window=1)
    print(f"✅ [Anchored Lineage Messages]: {len(anchored)} msgs recalled.")
    await episodic_store.close()
    print("✅ Test 2 Passed!")


if __name__ == "__main__":
    test_01_l3_semantic_memory_generation_and_retrieval()
    asyncio.run(test_02_l2_episodic_memory())
    print("\n🎉 ALL HERMES MEMORY TESTS PASSED SUCCESSFULLY!")
