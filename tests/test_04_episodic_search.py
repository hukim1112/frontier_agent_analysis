"""
Test 04: 5개 세션 채팅 데이터 구축, FTS5 세션 검색 & Anchor 인출 테스트 및 episodic.db 초기화
"""

import os
import sys
import json
import asyncio
from dotenv import load_dotenv

sys.path.insert(0, ".")
load_dotenv()

from app.utils import get_llm
from modules.hermes.session_store import EpisodicStore

# WSL 호스트 호환 경로
SCENARIOS_DIR = "/mnt/c/Users/hyoun/Desktop/harness_agent/notebooks/chat"

scenario_04 = {
    "session_id": "scenario_04",
    "messages": [
        {"role": "user", "content": "안녕! 나는 AI 연구소 연구원 최지훈이야. GPU 클러스터를 활용한 LLM 파인튜닝 실험 환경을 맞추고 있어."},
        {"role": "assistant", "content": "안녕하세요 최지훈 연구원님! GPU 클러스터 및 파인튜닝 학습 파이프라인 협업을 지원합니다. 컴퓨팅 노드 사양을 알려주세요."},
        {"role": "user", "content": "응, 우리는 NVIDIA A100 GPU가 장착된 gpu-node-a100 서버 대역을 쓰고 있어. 실험 트래킹은 Weights & Biases (wandb) 팀 프로젝트인 ml-research-llm 공간으로 기록해줘."},
        {"role": "assistant", "content": "확인했습니다. 대상 노드는 gpu-node-a100 이며, 실험 트래킹 모니터링은 Weights & Biases의 ml-research-llm 공간으로 매핑하겠습니다."},
        {"role": "user", "content": "그리고 중요한 체크포인트 보존 규칙이 있어. 모델 체크포인트 파일은 용량이 크기 때문에 최근 3개의 최신 epoch 파일만 남기고 이전 체크포인트는 자동으로 S3 버킷으로 아카이빙해야 해."},
        {"role": "assistant", "content": "확인했습니다. 디스크 용량 관리를 위해 최신 3개 epoch 체크포인트만 로컬에 보관하고 초과분은 S3 버킷으로 자동 아카이빙하는 정책을 적용하겠습니다."},
        {"role": "user", "content": "좋아, 고마워! 그럼 PyTorch 2.3 기반 분산 학습 분산 실행 스크립트 초안을 작성해줄 수 있어?"},
        {"role": "assistant", "content": "네, 최지훈 연구원님. gpu-node-a100 노드와 wandb 연동, S3 아카이빙을 포함한 PyTorch 2.3 분산 학습 가이드를 제공하겠습니다."}
    ]
}

scenario_05 = {
    "session_id": "scenario_05",
    "messages": [
        {"role": "user", "content": "안녕, 나는 전자결제 시스템 개발팀 한정우 차장이야. 신규 PG 결제 게이트웨이 API 연동 규격을 조율하려 해."},
        {"role": "assistant", "content": "안녕하세요 한정우 차장님! PG 결제 게이트웨이 및 금융 보안 규격 연동 협업을 지원하겠습니다. 결제 엔드포인트 주소를 말씀해주세요."},
        {"role": "user", "content": "응, 대상 게이트웨이는 api.payments.internal 대역이고 PCI-DSS 금융 보안 표준 준수가 필수적이야."},
        {"role": "assistant", "content": "확인했습니다. 결제 게이트웨이 주소는 api.payments.internal 이며 PCI-DSS 금융 보안 컴플라이언스를 적용하겠습니다."},
        {"role": "user", "content": "보안 규정상 OAuth 2.0 Access Token의 만료 시간은 15분으로 제한되어야 하고, 모든 결제 트랜잭션의 감사 로그(audit log)는 암호화하여 7년간 보관되어야 해."},
        {"role": "assistant", "content": "확인했습니다. OAuth 2.0 토큰 만료 15분 제약과 결제 감사 로그 7년 암호화 보존 의무 규칙을 동기화하겠습니다."},
        {"role": "user", "content": "고마워! 그럼 그 조건에 맞춘 결제 연동 모듈 가이드를 작성해 줘."},
        {"role": "assistant", "content": "네, 한정우 차장님. api.payments.internal 보안 결제 연동 가이드라인을 바로 작성해 드리겠습니다."}
    ]
}


def load_all_5_scenarios():
    scenarios = []
    # 01~03 파일 로드
    for i in range(1, 4):
        file_path = os.path.join(SCENARIOS_DIR, f"scenario_0{i}.json")
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                scenarios.append(json.load(f))
        else:
            print(f"⚠️ Scenario file not found at: {file_path}")

    # 04, 05 메모리 데이터 추가
    scenarios.append(scenario_04)
    scenarios.append(scenario_05)
    return scenarios


async def run_episodic_search_experiment():
    db_path = "./app/database/episodic.db"
    print(f"=== [Episodic Store 5-Session Search & Reset Experiment] ===")
    print(f"Target DB Path: {db_path}\n")

    episodic_store = EpisodicStore(db_path=db_path)
    await episodic_store.setup()
    llm = get_llm("google_vertexai:gemini-3.5-flash", temperature=0.0)

    # 1. 5개 세션 데이터 로드 및 DB 저장 (Finalize Session)
    scenarios = load_all_5_scenarios()
    print(f"📦 Total Scenarios Loaded: {len(scenarios)}")
    assert len(scenarios) == 5, f"5개 세션이 모두 로드되어야 합니다! (현재: {len(scenarios)}개)"

    for sc in scenarios:
        sid = sc["session_id"]
        msgs = sc["messages"]
        summary = await episodic_store.finalize_session(sid, msgs, llm=llm)
        print(f"  - [{sid}] Finalized! Summary: {summary[:80]}...")

    print("\n" + "=" * 65)
    print("🔍 [FTS5 세션 검색 테스트 (5개 주제 정밀 매칭)]")
    print("=" * 65)

    # 2. 5개 주제별 검색 쿼리 및 기대 세션 매칭 테스트
    test_queries = [
        ("marketing campaign server S3 Slack", "scenario_01"),
        ("Oracle DB query customer_pii dashboard", "scenario_02"),
        ("K8s cluster Vault secrets retention", "scenario_03"),
        ("GPU fine-tuning PyTorch wandb checkpoint", "scenario_04"),
        ("payment gateway OAuth PCI-DSS audit", "scenario_05"),
    ]

    matched_count = 0
    for q, expected_sid in test_queries:
        print(f"\n🔎 쿼리: '{q}' (기대 세션: {expected_sid})")
        results = await episodic_store.search_sessions(query=q, top_k=2)
        if results:
            top_match = results[0]
            matched_sid = top_match["session_id"]
            if matched_sid == expected_sid:
                matched_count += 1
                print(f"  Result: ✅ MATCHED! -> [{matched_sid}] Summary: {top_match['summary'][:90]}...")
                print(f"          Keywords: {top_match['keywords']}")
            else:
                print(f"  Result: ⚠️ MISMATCH -> Found [{matched_sid}] instead of [{expected_sid}]")
        else:
            print("  Result: ❌ No matching session found.")

    print(f"\n📊 Total FTS5 Session Matches: {matched_count}/5 Successful!")

    print("\n" + "=" * 65)
    print("📍 [Anchor 기반 메시지 인출 테스트 (scenario_01 ~ 05)]")
    print("=" * 65)

    # 3. Anchor 기반 세션 메시지 인출
    rec1 = await episodic_store.get_anchored_view("scenario_01", anchor_keyword="192.168.10.45", window=1)
    print(f"\n[scenario_01 Anchor '192.168.10.45' (IP주소) 인출 결과]: {len(rec1)} 개 메시지")
    for m in rec1[:3]:
        print(f"  - [{m['role']}] {m['content'][:70]}...")

    rec3 = await episodic_store.get_anchored_view("scenario_03", anchor_keyword="Vault", window=1)
    print(f"\n[scenario_03 Anchor 'Vault' (비밀키) 인출 결과]: {len(rec3)} 개 메시지")
    for m in rec3[:3]:
        print(f"  - [{m['role']}] {m['content'][:70]}...")

    rec5 = await episodic_store.get_anchored_view("scenario_05", anchor_keyword="PCI-DSS", window=1)
    print(f"\n[scenario_05 Anchor 'PCI-DSS' (보안규격) 인출 결과]: {len(rec5)} 개 메시지")
    for m in rec5[:3]:
        print(f"  - [{m['role']}] {m['content'][:70]}...")

    # 4. episodic.db 초기화 (Reset)
    await episodic_store.close()

    print("\n" + "=" * 65)
    print("🧹 [episodic.db Reset 초기화 수행]")
    print("=" * 65)
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ DB File '{db_path}' has been deleted/reset successfully.")
    
    for ext in ["-wal", "-shm"]:
        p = db_path + ext
        if os.path.exists(p):
            os.remove(p)

    print("\n🎉 All 5-Session Search & Reset Tests Completed Successfully!")


if __name__ == "__main__":
    asyncio.run(run_episodic_search_experiment())
