# 🧠 Frontier Agent Lab (글로벌 에이전트 아키텍처 실습)

본 프로젝트는 글로벌 탑티어 에이전트인 **Claude Code**의 핵심 하네스 설계와 **Hermes Agent**의 계층형 메모리 설계를 역공학하여, LangChain 및 LangGraph 환경에서 직접 구현하고 검증하는 **실습 교안 및 레퍼런스 코드베이스**입니다.

---

## 🌟 핵심 설계 철학 및 아키텍처 구성

### 1. 🛡️ Claude Code 핵심 하네스 아키텍처

- **① 5-Layer 16대 모듈 프롬프트 어셈블러 (`app/middleware/prompt/`)**: 정적 글로벌 캐시 영역(L1~L2)과 세션별 동적 영역(L3~L5)을 경계 마커(`__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__`)로 분리해 조립하며, `CLAUDE.md`와 날짜는 대화 앞 별도 `<system-reminder>` 메시지로 주입합니다.
- **② Claude Code 컨텍스트 압축 파이프라인 & Amnesia Guard (`app/middleware/compaction/`)**: Tool Result Budget ➔ Snip ➔ Microcompact ➔ Context Collapse ➔ Auto-Compact ➔ Reactive Compact 순서로 컨텍스트를 압축하고, 압축 직후 Amnesia Guard가 활성 계획(Plan)과 최근 5개 파일 스냅샷을 복원합니다.
- **③ Self-Recovery & Self-Correction 에러 컨트롤 및 가드레일 (`app/middleware/`)**: 턴 시작 시 입력 안전성·주제 이탈 가드레일을 검사하고, 모델 폴백(Failover), 도구 예외 복구, 호출 상한 제어 및 생성 코드 문법 오류 자가 수정(Stop Hooks)을 수행합니다.

---

### 2. 🧠 Hermes 4-Layer Memory System (`app/middleware/memory/`)

- **L1 Working Memory (`checkpoints.db`)**: 세션별 런타임 단기 대화 상태 체크포인팅(압축본).
- **L2 Episodic Memory (`transcript.db` + `episodic.db`)**: 모든 대화·도구 원문을 실시간 기록(`transcript.db`)하고, 유휴 시점에 다중 주제 요약 및 FTS5 색인(`episodic.db`)을 생성하여 2단계 인출(`session_search` ➔ `session_recall`)을 지원합니다.
- **L3 Semantic Memory (`MEMORY.md` / `USER.md`)**: 에이전트 영구 팩트 및 사용자 프로필을 백그라운드 리뷰를 통해 마크다운으로 관리합니다.
- **L4 Procedural Memory (`SKILL.md`)**: 반복된 도구 해결 절차를 `SKILL.md`로 학습·저장하고, 프롬프트에는 스킬 카탈로그(이름·설명)만 노출한 뒤 필요 시 `skill_view`로 열람합니다.

---

## 📂 프로젝트 구조

```text
frontier_agent_analysis/
├── app/                              # 🧠 메인 애플리케이션 및 하네스 모듈
│   ├── agents/                       #   ├── main_agent.py (21종 도구 + 9단계 미들웨어 오케스트레이터)
│   │                                 #   └── chatbot.py, analyst.py, scraper.py
│   ├── middleware/                   #   📁 체계화된 하네스 미들웨어 패키지
│   │   ├── prompt/                   #   │   └── prompt_assembler.py (16-Module 5-Layer 조립기)
│   │   ├── compaction/               #   │   └── compactor.py (Claude Code 압축 파이프라인), amnesia_guard.py
│   │   ├── error_control/            #   │   └── self_recovery.py, self_correction.py
│   │   ├── guardrails.py             #   │   └── InputSafety & TopicAlignment (before_agent + jump_to end)
│   │   ├── memory/                   #   │   └── transcript_store.py, episodic_store.py, review_queue.py, semantic_store.py, memory_middleware.py,
│   │                                 #   │       procedural_store.py, skill_tools.py, skill_review.py
│   │   ├── evaluator/                #   │   └── evaluator.py (LLM-as-a-Judge), eval_runner.py (노트북 시나리오 평가)
│   │   └── observability/            #   │   └── agent_log_tracer.py, agent_tracer.py (노트북 실습용 트레이서), visualizer.py, viz.py
│   ├── prompts/                      #   └── SUPERVISOR.py, CHATBOT.py, ANALYST.py, SCRAPER.py
│   ├── tools/                        #   └── supervisor_tools.py, custom_tools.py, navigator.py, analyst.py
│   ├── utils/                        #   └── context.py, database/, log_analyzer.py
│   ├── server.py                     #   └── FastAPI 백엔드 서버
│   ├── chainlit_ui.py                #   └── Chainlit 대화형 채팅 UI (포트 8080)
│   └── streamlit_ui.py               #   └── Streamlit 보조 UI
│
├── notebooks/                        # 📗 수강생 단계별 실습 주피터 노트북 (필수 4종 + 참고 1종)
│   ├── 01_react_vs_frontier_agents.ipynb    # Step 1: ReAct vs 프론티어 에이전트 루프 & 관측성
│   ├── 02_claude_code_harness.ipynb         # Step 2: 5계층 프롬프트, 5대 압축 파이프라인, Amnesia Guard
│   ├── 03_hermes_memory_architecture.ipynb  # Step 3: 계층형 메모리(transcript.db + episodic.db) 및 FTS5 인출
│   ├── 04_hermes_procedural_memory.ipynb    # Step 4: Procedural Memory — 스킬 학습 루프, 전이 실험, Curator
│   └── (optional)_agent_guardrails.ipynb    # (Optional) 도구 실행 직전 안전장치(ToolGuard) & 간접 인젝션 방어
│
├── configs/                          # ⚙️ 런타임 제어 설정 파일
│   ├── compaction.config             #   └── 압축 임계치 및 스왑 디렉터리 설정
│   ├── guardrail.config              #   └── 입력 보안 및 주제 정렬 임계치
│   ├── hitl.config                   #   └── 도구 실행 사용자 승인(HITL) 규칙
│   ├── logging.config                #   └── 감사 로그 수집 설정
│   ├── memory.config                 #   └── 시맨틱/에피소딕/원문(transcript) 메모리 활성화 설정
│   └── model.config                  #   └── 기본 및 폴백 LLM 모델 지정
│
├── scripts/                          # 🔄 데이터 마이그레이션 스크립트
│   └── migrate_episodic_to_transcript.py # └── 구 episodic.db messages ➔ transcript.db 이관 스크립트
│
├── tests/                            # 🧪 검증 테스트 수트
│   ├── test_02_claude_code.py        #   └── Claude Code 기본 하네스 테스트
│   ├── test_prompt_assembler_advanced.py # └── 16대 모듈, 바이패스 주입, MCP 델타 고급 테스트
│   ├── test_compactor.py             #   └── 압축 파이프라인 단계별·그래프 통합 테스트
│   └── test_harness_regressions.py   #   └── 원문/색인 분리, 유휴 큐, 가드레일 before_agent, Auto-Compact before_model E2E 회귀 테스트
│
├── .devcontainer/                    # 🐳 GitHub Codespaces 원클릭 실행 환경
│   └── devcontainer.json             #   └── Python 3.12, Playwright, noVNC, Chainlit 자동 바인딩
│
├── install/                          # 📦 설치 스크립트 및 의존성 목록
│   ├── requirements.txt
│   └── install_all.sh
│
└── .env.example                      # 🔑 환경 변수 템플릿 (Codespaces 자동 생성 대응)
```

---

## 🚀 빠른 시작 (Quick Start)

### 방법 A: GitHub Codespaces (권장: 설치 없이 1초 시작)
1. GitHub 저장소 상단의 **`Code` -> `Codespaces` -> `Create codespace on main`** 클릭
2. 사전 빌드된 Docker 이미지(`hukimartia/agent-lab:latest`) 기반으로 Python, Playwright, noVNC, Chainlit 환경이 자동 구성됩니다.
3. 생성된 `.env` 파일에 발급받으신 LLM API Key(`OPENAI_API_KEY` 또는 `GOOGLE_API_KEY`)를 입력하면 즉시 준비 완료!

### 방법 B: 로컬 WSL2 (Ubuntu) 환경
```bash
# 1. 저장소 클론 및 이동
git clone https://github.com/hukim1112/frontier_agent_analysis.git
cd frontier_agent_analysis

# 2. 의존성 패키지 설치
pip install -r install/requirements.txt

# 3. 환경 변수 설정
cp .env.example .env
# .env 파일을 열어 API 키 입력
```

---

## 🖥️ UI 및 서버 구동

### 1. Chainlit 채팅 UI 구동 (권장)
```bash
chainlit run app/chainlit_ui.py --port 8080
```
- 브라우저에서 `http://localhost:8080` 접속
- **기본 로그인 계정**:
  - **ID**: `user`
  - **Password**: `1234`
- 21종 도구 제어, 실시간 5계층 프롬프트 조립, 메모리 자동 인출 및 대화 트레이스를 인터랙티브하게 확인 가능합니다.

### 2. 백엔드 FastAPI 서버 구동
```bash
python app/server.py --port 8000
```
- `http://localhost:8000/docs`에서 Swagger 인터페이스로 에이전트 엔드포인트 테스트 가능.
