import asyncio
from modules.hermes.session_store import EpisodicStore
from modules.hermes.session_search_tool import create_session_search_tool

async def test():
    store = EpisodicStore('app/database/episodic.db')
    await store.setup()
    
    # Test search_sessions directly
    res = await store.search_sessions('session_001')
    print("Direct search 'session_001':", res)
    
    res2 = await store.search_sessions('FastAPI')
    print("Direct search 'FastAPI':", res2)
    
    # Test tool
    tool = create_session_search_tool(store)
    tool_res = tool.invoke({"query": "session_001"})
    print("Tool search 'session_001':", tool_res)
    
    await store.close()

if __name__ == '__main__':
    asyncio.run(test())
