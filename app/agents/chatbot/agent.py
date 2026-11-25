"""
app/agents/chatbot/agent.py — 3대 코어 요소 + 커스텀 도구 기본 챗봇 에이전트

패키지 구조:
  app/agents/chatbot/
   ├── __init__.py    (create_agent_executor, AGENT_METADATA, CHATBOT_SYSTEM_PROMPT, tools_chatbot)
   ├── agent.py       (에이전트 조립 및 팩토리)
   ├── prompt.py      (CHATBOT_SYSTEM_PROMPT)
   └── tools.py       (범용 9대 도구 바인딩)

[하네스 구성 요소]
1. 코어 3요소: 프롬프트(CHATBOT_SYSTEM_PROMPT), 도구(active_tools), 단기기억(공유 SQLite 체크포인터)
2. 서비스 인터페이스: AGENT_METADATA (UI 프로필 및 FastAPI 레지스트리)
"""

import os
import json
from langchain.agents import create_agent

from app.utils import init_chat_model
from app.utils.database import get_shared_checkpointer
from app.context import AgentContext
from .prompt import CHATBOT_SYSTEM_PROMPT
from .tools import tools_chatbot
from app.tools.custom_tools import roll_dice, convert_currency

# 1. 서비스 메타데이터 (FastAPI 레지스트리 및 Chainlit UI ChatProfile 자동 등록)
AGENT_METADATA = {
    "name": "chatbot",
    "description": "3대 코어(프롬프트, 도구, 단기메모리)와 커스텀 도구(주사위/환율)가 완결 탑재된 고양이 챗봇",
    "roles": ["main"],  # main: UI 대화 전용 (서브에이전트 위임 대상 제외)
}


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

    # 3. 단기 기억 (공유 SQLite 체크포인터 SSOT)
    if checkpointer is None:
        checkpointer = await get_shared_checkpointer()

    # 4. 도구 바인딩 (범용 9대 도구 + 커스텀 도구 2종 결합)
    active_tools = list(tools_chatbot) + [roll_dice, convert_currency]

    # 5. 하네스로 결합된 최종 에이전트 인스턴스 구축
    return create_agent(
        model=llm,
        tools=active_tools,
        system_prompt=CHATBOT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        context_schema=AgentContext,
    )
