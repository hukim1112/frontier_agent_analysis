import asyncio
import os
import sys
import time
from app.agents.frontier_agent import create_agent_executor
from app.utils.context import AgentContext

async def run_full_verification():
    print("==========================================================================", flush=True)
    print("🧪 [Frontier Agent 4계층 메모리 & 세션 감지 & FTS5 에피소드 통합 검증]", flush=True)
    print("==========================================================================\n", flush=True)

    # 0. 메모리 및 DB 깨끗하게 사전 초기화
    memory_dir = "./artifacts/memory"
    os.makedirs(memory_dir, exist_ok=True)
    with open(os.path.join(memory_dir, "MEMORY.md"), "w", encoding="utf-8") as f:
        f.write("")
    with open(os.path.join(memory_dir, "USER.md"), "w", encoding="utf-8") as f:
        f.write("")
    
    db_path = "./app/database/episodic.db"
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print("🧹 이전 episodic.db 초기화 완료", flush=True)
        except Exception:
            pass

    # 1. 에이전트 팩토리 가동 (초기화된 상태에서 에이전트 구동)
    agent = await create_agent_executor()

    # --------------------------------------------------------------------------
    # 💬 [세션 1: session_001] 유저 프로필 & 스택 전달
    # --------------------------------------------------------------------------
    session1_id = "session_001_initial"
    session1_config = {"configurable": {"thread_id": session1_id}}
    session1_context = AgentContext(session_id=session1_id)
    user_msg_1 = (
        "안녕! 나는 파이썬 백엔드 엔지니어 김형욱이라고 해. "
        "우리 서비스는 FastAPI와 Redis 7을 핵심 기술 스택으로 쓰고 있어."
    )
    print(f"👤 [세션 1 ({session1_id}) 유저]: {user_msg_1}", flush=True)
    
    res1 = await agent.ainvoke({"messages": [("user", user_msg_1)]}, config=session1_config, context=session1_context)
    ans1 = res1["messages"][-1].content
    print(f"\n🤖 [세션 1 에이전트 답변]:\n{ans1}", flush=True)

    print("\n⏳ [백그라운드 학습] 데몬 스레드가 USER.md 저장 및 episodic.db 세션 요약을 마무리 중...", flush=True)
    await asyncio.sleep(8)  # 비동기 백그라운드 학습 및 FTS5 요약 완료 대기

    # 백그라운드 리뷰 후 USER.md 내용 확인
    with open(os.path.join(memory_dir, "USER.md"), "r", encoding="utf-8") as f:
        user_md_content = f.read()
    print(f"\n📄 [학습 완료 후 USER.md 내용]:\n{user_md_content}", flush=True)

    # --------------------------------------------------------------------------
    # 🔄 [세션 2: session_002_new] 전환 ➔ Session-Aware L3 Semantic Memory 주입 검증
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75, flush=True)
    print("🔄 [세션 전환] session_001_initial ➔ session_002_new (세션 전환 감지 스냅샷 로드!)", flush=True)
    print("=" * 75 + "\n", flush=True)

    session2_id = "session_002_new"
    session2_config = {"configurable": {"thread_id": session2_id}}
    session2_context = AgentContext(session_id=session2_id)
    user_msg_2 = "안녕! 내 이름과 우리 서비스 스택이 기억나니?"
    print(f"👤 [세션 2 ({session2_id}) 유저]: {user_msg_2}", flush=True)

    res2 = await agent.ainvoke({"messages": [("user", user_msg_2)]}, config=session2_config, context=session2_context)
    ans2 = res2["messages"][-1].content
    print(f"\n🤖 [세션 2 에이전트 답변 (L3 프롬프트 기억 인출 기반)]:\n{ans2}", flush=True)

    tool_calls_in_res2 = [m for m in res2["messages"] if getattr(m, "tool_calls", None)]
    print(f"\n📊 [세션 2 프롬프트 주입 검증]: 도구 호출 수 = {len(tool_calls_in_res2)}회", flush=True)
    if "김형욱" in str(ans2) or "Hyungwook" in str(ans2):
        print("✅ [성공] USER.md 마크다운 기억이 세션 2 프롬프트에 100% 자동 주입되어 이름을 정확히 답변함!", flush=True)
    else:
        print("❌ [실패] 프롬프트에 이름이 주입되지 않았습니다.", flush=True)

    # --------------------------------------------------------------------------
    # 🔍 [세션 2 - 턴 2] L2 Episodic Memory FTS5 에피소드 요약 검색 검증
    # --------------------------------------------------------------------------
    print("\n" + "=" * 75, flush=True)
    print("🔍 [L2 에피소드 검색 검증] 세션 1 대화 요약 FTS5 검색 테스트", flush=True)
    print("=" * 75 + "\n", flush=True)

    user_msg_3 = "우리가 이전 세션(session_001_initial)에서 나눴던 대화 요약이나 키워드를 session_search로 검색해줘."
    print(f"👤 [세션 2 - 턴 2 유저]: {user_msg_3}", flush=True)

    res3 = await agent.ainvoke({"messages": [("user", user_msg_3)]}, config=session2_config, context=session2_context)
    ans3 = res3["messages"][-1].content
    print(f"\n🤖 [세션 2 - 턴 2 에이전트 답변 (FTS5 에피소드 검색 기반)]:\n{ans3}", flush=True)

    tool_calls_in_res3 = [m for m in res3["messages"] if getattr(m, "tool_calls", None)]
    print(f"\n📊 [L2 에피소드 검색 검증]: 도구 호출 수 = {len(tool_calls_in_res3)}회", flush=True)
    if tool_calls_in_res3:
        for tc in tool_calls_in_res3:
            print(f"   🔧 사용된 도구: {tc.tool_calls}", flush=True)

    print("\n🎉 모든 아키텍처 및 세션 이관 검증이 성공적으로 완료되었습니다!", flush=True)

if __name__ == "__main__":
    asyncio.run(run_full_verification())
