# 🧠 Frontier Agent Lab (글로벌 에이전트 아키텍처 실습)

본 프로젝트는 글로벌 탑티어 에이전트인 **Claude Code**의 핵심 하네스 설계와 **Hermes Agent**의 계층형 메모리 설계를 역공학하여, LangChain 및 LangGraph 환경에서 직접 구현하고 검증하는 **실습 교안 및 레퍼런스 코드베이스**입니다.

---

## 🌟 핵심 설계 철학 및 아키텍처 구성

### 1. 🛡️ Claude Code 핵심 하네스 아키텍처

#### ① 5-Layer 16대 모듈 프롬프트 어셈블러 (`app/middleware/prompt/`)
- **Layer 1: Global Constitution (Modules 0~3, 5~6)**: 정체성, 보안 가이드, 린 엔지니어링, 비가역적 파괴 작업 승인 규칙, 간결한 톤&스타일, 효율적 출력 정책.
- **Layer 2: Tool Guide & Skills (Module 4)**: Anti-Raw Bash 가드레일, 온디맨드 스킬 카탈로그. 도구 정의는 원본처럼 시스템 프롬프트가 아닌 API `tools` 필드로 보내며, 미들웨어가 이름순(내장 → MCP)으로 정렬합니다.
- **🔻 `__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__`**: 정적 영역(L1~L2, 전 사용자 공용 캐시)과 동적 영역(L3~L5, 세션별)을 분리합니다.
- **Layer 3: Session Interaction (Module 8)**: 대화형 CLI 명령 안내(`! <cmd>`), 슬래시 커맨드 인식.
- **Layer 4: Environment & Memory Index (Modules 9~12)**: CWD, OS, Session ID 등 환경 메타데이터, 언어 정책, 출력 스타일, 인출된 메모리.
- **Layer 5: Resource Control & Project Rules (Modules 13~16)**: MCP 지침, 권한 면제 Scratchpad 샌드박스, FRC(Function Result Clearing: 오래된 도구 결과 정리) 공지, 도구 결과 요약 정책 및 `AGENT.md` 로컬 규칙.
- **세션당 1회 계산**: 동적 섹션은 원본 `systemPromptSection`처럼 세션에서 한 번 만들어 재사용합니다. MCP 지침과 메모리(턴마다 에피소드 검색)만 매 호출 다시 계산합니다.
- **시스템 컨텍스트 / 사용자 컨텍스트**: 원본 `getSystemContext`·`getUserContext`처럼 Git 스냅샷(브랜치·status 2000자 절단·최근 커밋, `@lru_cache` 동결)은 시스템 프롬프트 맨 끝에 `gitStatus:`로, `CLAUDE.md`와 세션 시작 날짜는 대화 맨 앞 별도 `<system-reminder>` 메시지로 넣습니다. 자정이 지나면 앞부분은 그대로 두고 대화 끝에 `date_change` 알림을 붙입니다.

#### ② Claude Code 컨텍스트 압축 파이프라인 & Amnesia Guard (`app/middleware/compaction/`)
Claude Code(유출 소스 v2.1.88) `query.ts`의 실행 순서를 따릅니다. 🟢는 원본 소스를 옮긴 단계, 🟡는 원본 파일이 빌드에서 빠져 있어 호출부 흔적으로 재구성한 단계입니다. 단계별 on/off와 수치는 `configs/compaction.config`에서 바꿉니다.
- **Tool Result Budget** 🟢: 도구 결과가 50,000자를 넘으면 `artifacts/tool-results/<세션>/`에 저장하고 모델에는 `<persisted-output>` 미리보기(2,000자)와 파일 경로만 전달. 한 라운드 합계는 200,000자 예산. `file_read` 결과는 저장하지 않음: 도구가 스스로 분량을 제한하고(이 프로젝트: 기본 250줄), 저장 파일을 다시 읽는 순환도 생기기 때문.
- **Snip** 🟡: 모델이 `snip` 도구로 `[id:...]` 태그가 붙은 메시지를 지목해 컨텍스트에서 뺌. 도구 호출·결과 쌍은 함께 빠지고 현재 턴은 보호. state에는 원본이 남고 모델 입력에서만 빠짐. 컨텍스트가 임계치의 70%를 넘으면 snip 사용을 권하는 알림을 붙임.
- **Microcompact (time-based)** 🟢: 마지막 응답 후 60분 이상 지나면(서버 캐시 만료) 최근 5개를 뺀 오래된 도구 결과 본문을 `[Old tool result content cleared]`로 비움. 같은 id로 state를 교체하므로 `transcript.db` 원문은 그대로.
- **Context Collapse** 🟡 (기본 꺼짐): 컨텍스트가 임계치의 90%에 이르면 완료된 이전 턴을 1개씩 LLM으로 요약해 커밋 로그에 쌓고, 모델 입력에서만 접어서 보여줌. 켜면 Auto-Compact는 발동하지 않음(원본 동작).
- **Auto-Compact (`before_model`)** 🟢: 토큰 임계치 초과 시 **이전 턴까지만** 원본의 9개 섹션 프롬프트로 요약해 checkpointer state를 교체(`RemoveMessage(REMOVE_ALL_MESSAGES)`). 요약 실패 시 압축하지 않고, 3회 연속 실패하면 세션 내 중단(circuit breaker). 요약 요청 자체가 너무 길면 오래된 API 라운드부터 버리고 최대 3회 재시도. 원문은 `transcript.db`에 보존되며 압축 이벤트(`kind=compaction`)가 기록됩니다.
- **Amnesia Guard**: 압축 후 활성 계획(Plan)과 최근 5개 파일 스냅샷을 복원 블록으로 재주입 (원본 `POST_COMPACT_MAX_FILES_TO_RESTORE = 5`).
- **Reactive Compact** 🟡: API 413 `prompt_too_long` 에러 시(`ModelFallbackMiddleware`가 413을 가로채지 않고 재전파) 이전 턴 요약을 시도하고, 안 되면 가장 오래된 API 라운드 20%를 잘라낸 뒤 한 번 재시도.

#### ③ Self-Recovery & Self-Correction 에러 컨트롤 및 가드레일 (`app/middleware/`)
- **Guardrails (`before_agent`)**: `InputSafetyGuardrail` 및 `TopicAlignmentGuardrail`이 `before_agent` 훅(`can_jump_to=["end"]`)에서 턴당 1회만 검사를 수행하며, 차단 시 메모리 인출을 건너뛰고 `transcript.db`에는 `kind=blocked`로 기록(검색·회상·학습 제외).
- **ModelFallbackMiddleware**: 메인 LLM API 장애 시 백업 모델로 자동 failover (`request.override(model=...)` 사용, 413 컨텍스트 초과 에러는 Compactor로 재전파).
- **ToolErrorHandlerMiddleware**: 도구 실행 예외를 크래시 대신 `ToolMessage`로 변환하여 에이전트의 자가 회복 유도.
- **ModelCallLimitMiddleware**: 무한 루프 방지 (호출 횟수 상한 제어).
- **StopHooksMiddleware**: 에이전트가 생성한 파이썬 코드의 문법/들여쓰기 오류를 사전 인터셉트하여 blockingError를 주입, 스스로 코드를 자가 수정(Self-Correction)하도록 강제.

---

### 2. 🧠 Hermes 4-Layer Memory System (`app/middleware/memory/`)
- **L1 Working Memory (`checkpoints.db`)**: `AsyncSqliteSaver` 기반 세션별 런타임 단기 상태 체크포인팅 (**압축본**).
- **L2 Episodic — 원문 저장소 (`transcript.db`, `TranscriptStore`)**: 모든 사용자·AI·도구 결과 원문 및 압축 이벤트를 실시간 Append-only로 기록하는 **전체본**. `episodic_memory_enabled` / `episodic_learning_enabled`와 독립된 `transcript.enabled` 스위치로 동작.
- **L2 Episodic — 요약 색인 (`episodic.db`, `EpisodicStore` + `IdleReviewQueue`)**: 마지막 턴 후 15초 유휴 대기(최대 30분) 뒤 `transcript.db` 원문을 읽어 **다중 주제(Multi-Topic) 구조화 요약(`topics_json`, `tool_names`)** 및 SQLite FTS5 색인을 생성. `session_search`(색인 검색 + 원문 FTS 폴백) ➔ `session_recall`(`transcript.db` Anchor ±window 원문 인출) 2단계 인출 지원.
- **L3 Semantic Memory (`MEMORY.md` / `USER.md`)**: 에이전트 팩트 및 사용자 프로필 관리. 유휴 큐 백그라운드 리뷰를 통한 스마트 마크다운 병합(`add`, `replace`, `remove`).
- **L4 Procedural Memory (`artifacts/users/{user_id}/skills/`, `ProceduralMemoryStore`)**: 작업 유형별 절차를 `SKILL.md`로 학습하는 기억. 도구 호출이 10회 쌓인 세션은 유휴 리뷰에서 스킬 도구만 가진 리뷰 에이전트가 스킬을 생성·patch(`skill_review.py`). 카탈로그(이름+설명)만 L4에 세션 단위로 주입하고 본문은 `skill_view`로 열람. 백그라운드 쓰기는 에이전트 소유 스킬만, 이번 리뷰에서 읽은 파일만 허용. 사람이 작성한 번들 스킬(`main_agent/skills/`)은 Layer 2.2 카탈로그로 별도 노출.

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
├── notebooks/                        # 📗 수강생 단계별 실습 주피터 노트북 (4종)
│   ├── 01_react_vs_frontier_agents.ipynb    # Step 1: ReAct vs 프론티어 에이전트 루프 & 관측성
│   ├── 02_claude_code_harness.ipynb         # Step 2: 5계층 프롬프트, 5대 압축 파이프라인, Amnesia Guard
│   ├── 03_hermes_memory_architecture.ipynb  # Step 3: 계층형 메모리(transcript.db + episodic.db) 및 FTS5 인출
│   └── 04_hermes_procedural_memory.ipynb    # Step 4: Procedural Memory — 스킬 학습 루프, 전이 실험, Curator
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
├── tests/                            # 🧪 132개 검증 테스트 수트
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

## ️ UI 및 서버 구동

### 1. Chainlit 채팅 UI 구동 (권장)
```bash
chainlit run app/chainlit_ui.py -w --port 8080
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
