# 🧠 Frontier Agent Lab (글로벌 에이전트 아키텍처 실습)

본 프로젝트는 글로벌 탑티어 에이전트인 **Claude Code**의 핵심 하네스 설계와 **Hermes Agent**의 계층형 메모리 설계를 역공학하여, LangChain 및 LangGraph 환경에서 직접 구현하고 검증하는 **실습 교안 및 레퍼런스 코드베이스**입니다.

---

## 🌟 핵심 설계 철학 및 아키텍처 구성

### 1. 🛡️ Claude Code 핵심 하네스 아키텍처

#### ① 5-Layer 16대 모듈 프롬프트 어셈블러 (`app/middleware/prompt/`)
- **Layer 1: Global Constitution (Modules 0~3, 5~6)**: 정체성, 보안 가이드, 린 엔지니어링, 비가역적 파괴 작업 승인 규칙, 간결한 톤&스타일, 효율적 출력 정책.
- **Layer 2: Capabilities & Tool Guide (Module 4)**: Anti-Raw Bash 가드레일, 알파벳순 정렬된 도구 스키마, 온디맨드 스킬 카탈로그.
- **🔻 `__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__`**: 정적 영역(L1~L2)과 동적 영역(L3~L5)을 분리하여 **KV Cache Hit Rate(97%+) 극대화**.
- **Layer 3: Session Interaction (Module 8)**: 대화형 CLI 명령 안내(`! <cmd>`), 슬래시 커맨드 인식.
- **Layer 4: Environment & Memory Index (Modules 9~12)**: CWD, Session ID, Date 등 환경 메타데이터, 언어 정책, 출력 스타일, 인출된 시맨틱 메모리.
- **Layer 5: Resource Control & Project Rules (Modules 13~16)**: MCP 지침, 권한 면제 Scratchpad 샌드박스, FRC(False Response Claim) 방지 수칙, 도구 결과 요약 정책 및 `AGENT.md` 로컬 규칙.
- **User Context Bypass Injection**: `CLAUDE.md`와 Git 스냅샷 동결(`@lru_cache`, 2000자 절단)을 시스템 프롬프트 외부(`messages[0]`의 `<system-reminder>`)로 1회 우회 주입.

#### ② 5대 컨텍스트 압축 파이프라인 & Amnesia Guard (`app/middleware/compaction/`)
- **Snip Compact**: 2턴 이상 경과한 오래된 대용량 툴 결과를 1줄 스텁(`[Output snipped]`)으로 변환 (토큰 90%+ 절감).
- **Microcompact**: 5,000자 초과 대용량 툴 로그를 디스크 스왑 파일(`.claude/swaps/`)로 덤프 후 포인터만 잔류.
- **Context Collapse**: 3회 이상 연속 탐색 툴 실행 내역을 단일 접힌 블록으로 병합.
- **Auto-Compact**: 토큰 임계치 초과 시 4대 영역 구조화 요약 생성.
- **Amnesia Guard**: 압축으로 인해 활성 계획(Plan)과 수정 중인 최근 5개 파일 스냅샷이 소실(기억상실)되는 것을 방지하기 위해 복구 어태치먼트를 시스템 프롬프트로 재주입.
- **Reactive Compact**: API 413 `prompt_too_long` 에러 발생 시 에러를 은폐(Silent Withholding)하고 오래된 대화 20% 절단 후 자동 재시도.

#### ③ Self-Recovery & Self-Correction 에러 컨트롤 (`app/middleware/error_control/`)
- **ModelFallbackMiddleware**: 메인 LLM API 장애 시 백업 모델로 자동 failover.
- **ToolErrorHandlerMiddleware**: 도구 실행 예외를 크래시 대신 `ToolMessage`로 변환하여 에이전트의 자가 회복 유도.
- **ModelCallLimitMiddleware**: 무한 루프 방지 (호출 횟수 상한 제어).
- **StopHooksMiddleware**: 에이전트가 생성한 파이썬 코드의 문법/들여쓰기 오류를 사전 인터셉트하여 blockingError를 주입, 스스로 코드를 자가 수정(Self-Correction)하도록 강제.

---

### 2. 🧠 Hermes 4-Layer Memory System (`app/middleware/memory/`)
- **L1 Working Memory (`checkpoints.db`)**: `AsyncSqliteSaver` 기반 세션별 런타임 단기 상태 체크포인팅.
- **L2 Episodic Memory (`episodic.db`)**: SQLite FTS5 전문 검색 엔진 기반 대형 과거 세션 데이터 탐색 및 JIT 인출.
- **L3 Semantic Memory (`MEMORY.md` / `USER.md`)**: 에이전트 팩트 및 사용자 프로필 관리. 백그라운드 리뷰를 통한 스마트 마크다운 병합(`add`, `replace`, `remove`).
- **L4 Procedural Memory**: 프롬프트 어셈블러 및 스킬 카탈로그(`SKILL.md`)와 연동된 절차 기억.

---

## 📂 프로젝트 구조

```text
frontier_agent_analysis/
├── app/                              # 🧠 메인 애플리케이션 및 하네스 모듈
│   ├── agents/                       #   ├── main_agent.py (19종 도구 + 9단계 미들웨어 오케스트레이터)
│   │                                 #   └── chatbot.py, analyst.py, scraper.py
│   ├── middleware/                   #   📁 체계화된 하네스 미들웨어 패키지
│   │   ├── prompt/                   #   │   └── prompt_assembler.py (16-Module 5-Layer 조립기)
│   │   ├── compaction/               #   │   └── compactor.py, amnesia_guard.py
│   │   ├── error_control/            #   │   └── self_recovery.py, self_correction.py
│   │   ├── memory/                   #   │   └── semantic_store.py, episodic_store.py, memory_middleware.py
│   │   └── observability/            #   │   └── agent_log_tracer.py, visualizer.py
│   ├── prompts/                      #   └── SUPERVISOR.py, CHATBOT.py, ANALYST.py, SCRAPER.py
│   ├── tools/                        #   └── supervisor_tools.py, custom_tools.py, navigator.py, analyst.py
│   ├── utils/                        #   └── context.py, database/, log_analyzer.py
│   ├── server.py                     #   └── FastAPI 백엔드 서버
│   ├── chainlit_ui.py                #   └── Chainlit 대화형 채팅 UI (포트 8080)
│   └── streamlit_ui.py               #   └── Streamlit 보조 UI
│
├── modules/                          # 🛠️ 실습 교안 전용 독립 모듈
│   ├── claude_code/                  #   └── prompt_assembler, compactor, amnesia_guard, self_correction
│   └── common/                       #   └── agent_tracer.py (토큰/지연시간 정밀 감사 트레이서)
│
├── notebooks/                        # 📗 수강생 단계별 실습 주피터 노트북 (3종)
│   ├── 01_react_vs_frontier_agents.ipynb    # Step 1: ReAct vs 프론티어 에이전트 루프 & 관측성
│   ├── 02_claude_code_harness.ipynb         # Step 2: 5계층 프롬프트, 5대 압축 파이프라인, Amnesia Guard
│   └── 03_hermes_memory_architecture.ipynb  # Step 3: 계층형 메모리 및 에피소딕 FTS5 인출
│
├── configs/                          # ⚙️ 런타임 제어 설정 파일
│   ├── compaction.config             #   └── 압축 임계치 및 스왑 디렉터리 설정
│   ├── guardrail.config              #   └── 입력 보안 및 주제 정렬 임계치
│   ├── hitl.config                   #   └── 도구 실행 사용자 승인(HITL) 규칙
│   ├── logging.config                #   └── 감사 로그 수집 설정
│   ├── memory.config                 #   └── 시맨틱/에피소딕 메모리 활성화 설정
│   └── model.config                  #   └── 기본 및 폴백 LLM 모델 지정
│
├── tests/                            # 🧪 62개 검증 테스트 수트
│   ├── test_02_claude_code.py        #   └── Claude Code 기본 하네스 테스트
│   ├── test_prompt_assembler_advanced.py # └── 16대 모듈, 바이패스 주입, MCP 델타 고급 테스트
│   └── test_compactor.py             #   └── 5대 압축 파이프라인 정밀 테스트
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

## 🧪 테스트 실행 검증

하네스의 16대 프롬프트 모듈, 5대 컨텍스트 압축기, Amnesia Guard, 에러 복구 미들웨어가 정상 작동하는지 전체 테스트를 수행할 수 있습니다:

```bash
pytest -v
# 총 62개 테스트 케이스 100% 통과 확인 (tests/test_*.py)
```

---

## 🖥️ UI 및 서버 구동

### 1. Chainlit 채팅 UI 구동 (권장)
```bash
chainlit run app/chainlit_ui.py -w --port 8080
```
- 브라우저에서 `http://localhost:8080` 접속
- 19종 도구 제어, 실시간 5계층 프롬프트 조립, 메모리 자동 인출 및 대화 트레이스를 인터랙티브하게 확인 가능합니다.

### 2. 백엔드 FastAPI 서버 구동
```bash
python app/server.py --port 8000
```
- `http://localhost:8000/docs`에서 Swagger 인터페이스로 에이전트 엔드포인트 테스트 가능.
