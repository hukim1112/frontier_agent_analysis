"""
app/agents/chatbot.py — 3대 코어 요소 + Mission 01 커스텀 도구 기준 완성본

[하네스 구성 요소]
1. 코어 3요소: 프롬프트(CHATBOT_SYSTEM_PROMPT), 도구(active_tools), 단기기억(AsyncSqliteSaver)
2. 서비스 인터페이스: AGENT_METADATA (UI 프로필 및 FastAPI 레지스트리)
"""

import os
import json
import aiosqlite
from datetime import date
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.agents import create_agent
from app.utils import init_chat_model
from app.tools import tools_chatbot
from app.tools.custom_tools import roll_dice, convert_currency
from app.context import AgentContext

# 1. 서비스 메타데이터 (FastAPI 레지스트리 및 Chainlit UI ChatProfile 자동 등록)
AGENT_METADATA = {
    "name": "chatbot",
    "description": "3대 코어(프롬프트, 도구, 단기메모리)와 커스텀 도구(주사위/환율)가 완결 탑재된 고양이 챗봇",
    "roles": ["main"],  # main: UI 대화 전용 (서브에이전트 위임 대상 제외)
}

CHATBOT_SYSTEM_PROMPT = f"""당신은 귀엽고 친밀한 고양이 페르소나를 가진 챗봇 에이전트입니다.
사용자의 질문에 대해 재치있고 흥미롭게 대화를 하세요. 답변은 한국어로 제공하세요.

[파일 저장 규칙]
사용자의 요청으로 파일이나 코드를 생성/저장하는 경우, 프로젝트 루트가 아닌 `artifacts/` 폴더 하위에 저장하세요.

오늘의 날짜 : {date.today().strftime("%Y-%m-%d")}
"""


def _load_config(path: str, default: dict) -> dict:
    """설정 파일을 로드합니다. 실패 시 기본값을 반환합니다."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


async def create_agent_executor(checkpointer=None):
    # 2. LLM 초기화 (configs/model.config 기반)
    model_cfg = _load_config("./configs/model.config", {
        "model_name": "gemini-3.8-flash",
        "temperature": 0.0,
    })
    llm = init_chat_model(
        model=model_cfg.get("model_name", "gemini-3.8-flash"),
        temperature=model_cfg.get("temperature", 0.0)
    )
    
    # 3. 단기 기억 (SQLite 기반 대화 세션 체크포인터)
    db_dir = "app/database"
    os.makedirs(db_dir, exist_ok=True)
    checkpoints_path = os.path.join(db_dir, "checkpoints.db")

    conn = None
    if checkpointer is None:
        conn = await aiosqlite.connect(checkpoints_path, check_same_thread=False)
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA busy_timeout=5000;")
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()
    
    # 4. 도구 바인딩 (범용 10대 도구 + Mission 01 커스텀 도구 2종 결합)
    active_tools = list(tools_chatbot) + [roll_dice, convert_currency]
    
    # 5. 하네스로 결합된 최종 에이전트 인스턴스 구축
    chatbot_agent = create_agent(
        model=llm,
        tools=active_tools,
        system_prompt=CHATBOT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        context_schema=AgentContext
    )

    chatbot_agent.registered_tools = active_tools
    chatbot_agent.checkpointer_conn = conn
    return chatbot_agent
