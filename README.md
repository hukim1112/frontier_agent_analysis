# 🧠 Frontier Agent Lab (글로벌 에이전트 아키텍처 실습)

본 프로젝트는 Hermes Agent 및 Claude Code의 핵심 아키텍처인 **4계층 메모리 시스템(4-Layer Memory)**, **자율 학습 루프(Closed Learning Loop)**, **멀티 모델 라우팅(Multi-Model Routing)**, **컨텍스트 압축(Compactor & Amnesia Guard)**이 구현된 프론티어 에이전트 코드베이스 실습 환경입니다.

---

## 🌟 핵심 아키텍처 및 구현 특징

### 1. 🧠 Hermes 4-Layer Memory System
- **L1 Working Memory (`checkpoints.db`)**: `AsyncSqliteSaver` 기반으로 대화 세션 단기 상태 및 스레드 체크포인팅.
- **L2 Episodic Memory (`episodic.db`)**: SQLite FTS5 전문 검색 엔진 기반으로 10개 이상 대형 과거 세션 데이터 탐색 및 Anchor 키워드 기반 메시지 맥락 인출.
- **L3 Semantic Memory (`MEMORY.md` / `USER.md`)**: 에이전트 팩트 및 유저 프로필 선호도 보관. LLM 백그라운드 리뷰를 통한 스마트 병합(`add`, `replace`, `remove`).
- **L4 Procedural Memory & Dynamic Prompting**: `PROMPT.md`, `MCP.md`, `AGENT.md`, `SKILL.md`를 5개 계층으로 조립하는 `PromptAssembler` 미들웨어.

### 2. ⚡ 멀티 모델 라우팅 (Multi-Model Routing)
- **메인 추론 에이전트 (`gemini-3.5-flash`)**: 대용량 문맥 처리, 복잡한 파이썬 코딩 및 다중 도구 제어 전담.
- **백그라운드 학습 데몬 (`openai:gpt-4o-mini`)**: 대화 완료 후 비동기 데몬 스레드에서 **1.35초 초고속**으로 팩트 수집 및 메모리 마크다운 업데이트.


### 3. 🛡️ Claude Code 하네스 아키텍처 (Context Compaction & Memory Safety)
- **AutoCompactor (`compactor.py`)**: 컨텍스트 토큰 임계치 초과 시 대화 내역 요약 및 이전 대화 압축.
- **Amnesia Guard (`amnesia_guard.py`)**: 컨텍스트 압축 시 작업 중이던 파일 스냅샷과 작업 계획(Plan Text)이 상실되어 에이전트가 단기 기억상실(Amnesia)에 빠지는 것을 막기 위해 가공 파일 스냅샷을 SystemMessage로 자동 복구 주입하는 미들웨어.
- **Stop Hooks & Self-Correction (`stop_hooks.py`, `self_correction.py`)**: 압축 직후 에이전트가 루프나 멍때림(Stuck State)에 빠졌을 때 자가 치유(Self-Healing) 조치를 격발하는 가드레일.

### 4. 📊 정밀 트레이싱 및 감사 관측성 (AgentTracer)
- LLM 토큰 수치(Prompt/Completion Token), 지연시간(Latency), 도구 격발 내역 추적 및 주피터 노트북 시각화(`display_trace()`) 지원.

---

## 🚀 시작하기 (환경 세팅)

로컬 WSL2(우분투) 환경에서 다음 명령어를 실행하여 의존성 패키지를 설치하고 환경을 설정하세요.

```bash
# 1. install 폴더로 이동하여 패키지 설치
cd install
bash install_all.sh
```

### 환경 변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 사용할 API 키를 설정하세요.

```env
OPENAI_API_KEY="your-openai-api-key"
GOOGLE_API_KEY="your-gemini-api-key"
```
---

## 📂 프로젝트 구조

```text
frontier-agent-lab/
├── app/                    # 🧠 핵심 애플리케이션 및 API 서버
│   ├── agents/             #   └── frontier_agent.py (Hermes 메모리 결합), chatbot.py
│   ├── prompts/            #   └── PROMPT.md, MCP.md, AGENT.md, SKILL.md
│   ├── tools/              #   └── common_tools, memory_tools
│   ├── utils/              #   └── get_llm, AgentContext
│   ├── server.py           #   └── FastAPI 서버 (동적 에이전트 로더 & config 반영)
│   └── ui.py               #   └── Streamlit 대화형 웹 UI
│
├── modules/                # 🛠️ 글로벌 에이전트 핵심 아키텍처 모듈
│   ├── hermes/             #   └── memory_store, session_store, memory_middleware, prompt_assembler
│   ├── claude_code/        #   └── compactor, amnesia_guard, stop_hooks
│   └── common/             #   └── agent_tracer (토큰 & 지연시간 정밀 트레이서)
│
├── notebooks/              # 📗 수강생 실습용 주피터 노트북 (3종)
│   ├── 01_react_vs_frontier_agents.ipynb
│   ├── 02_claude_code_harness.ipynb
│   └── 03_hermes_memory_architecture.ipynb
│
├── tests/                  # 🧪 정식 pytest 수트 (FTS5 10개 시나리오 포함)
│   ├── test_02_claude_code.py
│   ├── test_03_hermes.py
│   └── test_04_episodic_search.py
│
├── artifacts/              # 📂 메모리 및 대화 데이터셋 보관함
│   ├── memory/             #   └── MEMORY.md, USER.md
│   ├── chat/               #   └── scenario_01.json ~ scenario_10.json
│   └── logs/               #   └── {session_id}.jsonl 감사 로그
│
├── configs/                # ⚙️ memory.config, logging.config, hitl.config
└── README.md               # 📖 본 프로젝트 통합 설명서
```


---

## ⚙️ 설정 파일 제어 (`configs/`)

모든 메모리 및 로깅 미들웨어 옵션은 설정 파일(`load_config`)을 통해 유연하게 제어됩니다:

- **`configs/memory.config`**:
  ```json
  {
    "episodic_memory_enabled": true,
    "semantic_memory_enabled": true,
    "memory_learning_enabled": true,
    "memory_dir": "./artifacts/memory",
    "episodic_db_path": "./app/database/episodic.db"
  }
  ```
- **`configs/logging.config`**:
  ```json
  {
    "logging_enabled": true,
    "log_path": "./artifacts/agent_audit_trail.json"
  }
  ```

---

## 🖥️ 서버 및 웹 UI 실행 방법

### 1. 백엔드 FastAPI 서버 가동
```bash
python app/server.py --port 8000
```
* `http://localhost:8000/agents` 경로에서 동적으로 마운트된 `frontier_agent` 목록 확인 가능.

### 2. Streamlit 웹 채팅 UI 가동
```bash
streamlit run app/ui.py
```
* 브라우저에서 `http://localhost:8501` 접속 후 **`frontier_agent`**를 선택하여 대화 및 세션 간 장기 기억 이관 테스트 진행.

---

### 💬 내가 만든 에이전트를 웹 화면에 바로 추가하여 대화하기

이 프로젝트는 **서버를 껐다 켤 필요 없이, 에이전트 파일만 `app/agents/` 폴더에 넣으면 웹 화면이 실시간으로 알아채고 에이전트를 추가**해 줍니다. 

실습 도중 나만의 에이전트를 완성했거나 새로 만들고 싶다면, 아래의 3단계만 따라 해 보세요.

#### 1단계. 에이전트 파일 만들기
`app/agents/` 폴더 안에 원하는 이름으로 파이썬 파일(예: `my_agent.py`)을 새로 만듭니다.

#### 2단계. 에이전트 코드 작성하기 (그대로 복사해서 붙여넣기)
새로 만든 파일(`my_agent.py`) 안에 아래의 코드를 그대로 복사해서 붙여넣고 저장합니다. 

```python
# app/agents/my_agent.py

from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from app.utils import get_llm
from app.utils.context import AgentContext

# 1) UI에 표시될 에이전트의 소개 정보 (필수)
AGENT_METADATA = {
    "name": "my_agent", 
    "description": "더하기 도구가 탑재된 나만의 실습용 ReAct 에이전트"
}

# 2) 에이전트가 사용할 실제 도구 정의 (생략 없이 작동 가능한 도구 예시)
@tool
def add_numbers(a: int, b: int) -> int:
    """두 정수 a와 b를 더한 결과를 반환합니다. 더하기 연산이 필요할 때 사용하세요."""
    return a + b

# 3) 에이전트를 생성하는 함수 (서버가 이 함수를 찾아 실행합니다)
async def create_agent_executor():
    # 1. LLM 모델 생성 (Gemini 3.5 Flash 모델 활용)
    llm = get_llm(model_name="gemini-3.5-flash", temperature=0.0)
    
    # 2. 대화 기억 보존을 위한 체크포인터 셋업
    memory = MemorySaver()
    
    # 3. 도구 목록 정의
    tools = [add_numbers]
    
    # 4. 에이전트 최종 구축
    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        context_schema=AgentContext
    )
    return agent
```

#### 3단계. 웹 브라우저 새로고침하고 대화하기
1. 띄워져 있는 웹 채팅 화면([http://localhost:8501](http://localhost:8501))으로 이동하여 **새로고침(F5)**을 누릅니다.
2. 왼쪽 메뉴의 **"Select Agent" 드롭다운 상자**를 누르면, 방금 만든 `my_agent`가 실시간으로 감지되어 목록에 추가되어 있습니다.
3. 해당 에이전트를 선택하고 대화를 시작해 보세요!
   *(예: "37 더하기 84는 뭐야?" 라고 물어보면 에이전트가 탑재된 `add_numbers` 도구를 호출하여 정상적으로 덧셈 결과를 답변합니다.)*

---

