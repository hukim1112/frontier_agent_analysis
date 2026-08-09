from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from app.utils import get_llm
from app.prompts import CHATBOT_SYSTEM_PROMPT
from app.tools import common_tools
from app.utils.context import AgentContext

AGENT_METADATA = {
    "name": "chatbot",
    "description": "도구 및 모니터링이 활성화된 기준완성형 챗봇 (서버/UI 테스트용)"
}

def create_agent_executor():
    # 1. 일원화된 utils 유틸의 LLM 팩토리 활용 (openai: 접두사로 공급자 강제 매칭)
    llm = get_llm(model_name="gemini-3.5-flash", temperature=0.0)
    
    # 2. 대화 세션별 체크포인터 메모리 저장소 셋업
    memory = MemorySaver()
    
    # 3. 범용 8대 도구가 탑재된 스마트 챗봇 에이전트 구축
    chatbot_agent = create_agent(
        model=llm,
        tools=common_tools,
        system_prompt=CHATBOT_SYSTEM_PROMPT,
        checkpointer=memory,
        context_schema=AgentContext
    )
    return chatbot_agent
