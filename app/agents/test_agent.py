import asyncio
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage

AGENT_METADATA = {
    "name": "test_agent",
    "description": "런타임에 동적으로 로드된 특수 더미 에이전트"
}

def create_agent_executor():
    class DummyExecutor:
        def __init__(self):
            self.checkpointer = MemorySaver()
            
        async def ainvoke(self, input_dict, config=None, context=None):
            return {
                "messages": [
                    AIMessage(content="안녕하세요! 저는 실시간 핫 로드된 Test Agent입니다! 동적 임포트 테스트에 성공했습니다.")
                ]
            }
            
        async def astream_events(self, input_dict, config=None, context=None, version=None):
            # UI에 토큰 단위 스트리밍 효과를 출력하기 위해 on_chat_model_stream 이벤트를 흉내 냅니다.
            full_text = "안녕하세요! 저는 실시간 핫 로드된 Test Agent입니다! 동적 임포트 및 실시간 토큰 스트리밍 테스트에 성공했습니다. 🎉"
            words = full_text.split(" ")
            
            for i, word in enumerate(words):
                chunk_text = word + (" " if i < len(words) - 1 else "")
                yield {
                    "event": "on_chat_model_stream",
                    "tags": [],
                    "data": {
                        "chunk": AIMessage(content=chunk_text)
                    }
                }
                await asyncio.sleep(0.05)  # 글자 타이핑 효과를 모의하기 위한 비동기 딜레이
            
    return DummyExecutor()
