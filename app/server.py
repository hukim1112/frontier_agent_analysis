# 리눅스 환경에서 시스템 SQLite 버전이 낮을 경우 pysqlite3를 대신 사용하도록 강제함
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import sys
import os
from dotenv import load_dotenv

# 1. Setup Project Root Path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "src"))

# 2. 환경 변수 로드 (.env 파일 명시적 지정)
dotenv_path = os.path.join(project_root, ".env")
if os.path.exists(dotenv_path):
    print(f"Loading .env from: {dotenv_path}")
    load_dotenv(dotenv_path)
    
    # LangSmith Project Setting (Server Specific)
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "llmops-agent-server"
    print(f"📈 LangSmith Tracing Enabled. Project: {os.environ['LANGSMITH_PROJECT']}")
else:
    print("Warning: .env file not found.")

import logging
import json
import traceback
import inspect
import importlib
from typing import AsyncGenerator, Optional, Dict, Any, List

from fastapi import FastAPI, APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.utils.message_utils import sanitize_text, normalize_content
from app.utils.db import create_session, get_sessions, delete_session, add_message, get_messages
from app.utils.context import AgentContext

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMOps_Server")

def load_config(file_path: str, default: dict) -> dict:
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config {file_path}: {e}")
    return default

# --- Schemas ---
class UserInput(BaseModel):
    message: str
    thread_id: Optional[str] = None

class StreamInput(UserInput):
    stream_tokens: bool = Field(default=True)

class ChatMessage(BaseModel):
    type: str
    content: str

class SessionCreate(BaseModel):
    session_id: str
    agent_name: str
    title: str

# --- Router Factory ---
from contextlib import asynccontextmanager
import asyncio

# --- Lifespan Context Manager (비동기 루프 기동 후 에이전트 보관소 셋업 및 셧다운 리소스 클린업) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 FastAPI Lifespan Startup: Initializing agent storage and lock...")
    app.state.agents = {}
    app.state.load_lock = asyncio.Lock()
    yield
    logger.info("🛑 FastAPI Lifespan Shutdown: Cleaning up agent database connections...")
    for name, agent in list(app.state.agents.items()):
        try:
            # LangGraph checkpointer 커넥션 종료 처리
            checkpointer = getattr(agent, "checkpointer", None)
            if checkpointer:
                conn = getattr(checkpointer, "conn", None)
                if conn:
                    logger.info(f"🔌 Closing database connection for agent: {name}")
                    await conn.close()
        except Exception as e:
            logger.error(f"⚠️ Error cleaning up database connection for agent '{name}': {e}")

# --- App Initialization ---
app = FastAPI(
    title="LLMOps Class Agent Server", 
    version="1.0",
    description="Unified Server for Multiple Agents",
    lifespan=lifespan
)

# --- Dynamic Agent Loader ---
async def get_or_load_agent(agent_name: str, app: FastAPI) -> Any:
    """
    런타임 비동기 이벤트 루프 하에서 에이전트 모듈을 지연 로딩하고 캐싱합니다.
    """
    if agent_name in app.state.agents:
        return app.state.agents[agent_name]

    # 에이전트 파일 물리적 존재 유무 체크
    agents_dir = os.path.join(project_root, "app", "agents")
    agent_file = os.path.join(agents_dir, f"{agent_name}.py")
    if not os.path.exists(agent_file):
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found on server.")

    async with app.state.load_lock:
        # Double check locking pattern
        if agent_name in app.state.agents:
            return app.state.agents[agent_name]

        try:
            module_path = f"app.agents.{agent_name}"
            # 모듈 리로드를 지원하여 파일 수정 시 서버 기동 없이 즉시 반영
            if module_path in sys.modules:
                module = importlib.reload(sys.modules[module_path])
                logger.info(f"🔄 Reloaded agent module: {module_path}")
            else:
                module = importlib.import_module(module_path)
                logger.info(f"✅ Loaded agent module: {module_path}")

            # 팩토리 함수 검색
            factory = getattr(module, "create_agent_executor", None)
            if not factory:
                # 하위 호환성 폴백
                factory = getattr(module, f"create_{agent_name}_agent", None)
                if not factory:
                    factory = getattr(module, "get_agent_executor", None)

            if not factory:
                raise AttributeError(f"Module '{module_path}' must expose 'create_agent_executor' function.")

            # 동기/비동기 팩토리 모두 호환되도록 처리
            if inspect.iscoroutinefunction(factory):
                executor = await factory()
            else:
                executor = factory()

            app.state.agents[agent_name] = executor
            return executor

        except Exception as e:
            logger.error(f"❌ Failed to dynamically load agent '{agent_name}': {e}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to load agent '{agent_name}': {str(e)}")

# --- Dynamic Agent Registry API ---
@app.get("/agents")
def api_list_agents():
    """
    app/agents/ 폴더 내 파이썬 파일들을 스캔하여 사용 가능한 에이전트 목록을 동적으로 리턴합니다.
    """
    agents_dir = os.path.join(project_root, "app", "agents")
    available_agents = []
    
    if not os.path.exists(agents_dir):
        return []
        
    for filename in os.listdir(agents_dir):
        # __init__.py 및 헬퍼 모듈 제외
        if filename.endswith(".py") and not filename.startswith("__") and filename != "utils.py":
            agent_name = filename[:-3]
            description = f"Runtime loaded {agent_name} agent"
            
            try:
                module_path = f"app.agents.{agent_name}"
                # 메타데이터 파싱을 위해 가볍게 임포트
                if module_path in sys.modules:
                    module = sys.modules[module_path]
                else:
                    module = importlib.import_module(module_path)
                
                metadata = getattr(module, "AGENT_METADATA", None)
                if metadata and isinstance(metadata, dict):
                    name = metadata.get("name", agent_name)
                    description = metadata.get("description", description)
                    available_agents.append({"name": name, "description": description})
                else:
                    available_agents.append({"name": agent_name, "description": description})
            except Exception as scan_err:
                logger.warning(f"⚠️ Failed to parse metadata for {agent_name}: {scan_err}")
                available_agents.append({"name": agent_name, "description": description})
                
    return available_agents

# --- Unified Dynamic Routing ---
@app.post("/agents/{agent_name}/invoke", response_model=ChatMessage)
async def invoke_agent(agent_name: str, input_data: UserInput, request: Request):
    try:
        agent_executor = await get_or_load_agent(agent_name, request.app)
        config = {"configurable": {"thread_id": input_data.thread_id}, "recursion_limit": 100} if input_data.thread_id else {"recursion_limit": 100}
        
        logging_cfg = load_config("./configs/logging.config", {"logging_enabled": True, "log_path": "./artifacts/agent_audit_trail.json"})
        hitl_cfg = load_config("./configs/hitl.config", {"hitl_enabled": False})
        memory_cfg = load_config("./configs/memory.config", {
            "episodic_memory_enabled": True,
            "semantic_memory_enabled": True,
            "memory_learning_enabled": True,
            "memory_dir": "./artifacts/memory",
            "episodic_db_path": "./app/database/episodic.db",
        })
        
        context_obj = AgentContext(
            session_id=input_data.thread_id or "unknown",
            logging_enabled=logging_cfg.get("logging_enabled", True),
            log_path=logging_cfg.get("log_path", "./artifacts/agent_audit_trail.json"),
            response_mode="chat",
            hitl_enabled=hitl_cfg.get("hitl_enabled", False),
            debug_mode=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
            episodic_memory_enabled=memory_cfg.get("episodic_memory_enabled", True),
            semantic_memory_enabled=memory_cfg.get("semantic_memory_enabled", True),
            memory_learning_enabled=memory_cfg.get("memory_learning_enabled", True),
            memory_dir=memory_cfg.get("memory_dir", "./artifacts/memory"),
            episodic_db_path=memory_cfg.get("episodic_db_path", "./app/database/episodic.db"),
        )
        
        if input_data.thread_id:
            add_message(input_data.thread_id, "user", input_data.message)
            
        result = await agent_executor.ainvoke(
            {"messages": [("user", input_data.message)]},
            config=config,
            context=context_obj
        )
        last_message = result["messages"][-1]
        normalized = sanitize_text(normalize_content(last_message.content))
        
        if input_data.thread_id:
            add_message(input_data.thread_id, "assistant", normalized)
            
        return ChatMessage(type="ai", content=normalized)
    except Exception as e:
        logger.error(f"Dynamic invocation error in agent '{agent_name}': {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/{agent_name}/stream")
async def stream_agent(agent_name: str, input_data: StreamInput, request: Request):
    try:
        agent_executor = await get_or_load_agent(agent_name, request.app)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    async def _dynamic_stream_generator() -> AsyncGenerator[str, None]:
        try:
            config = {"configurable": {"thread_id": input_data.thread_id}, "recursion_limit": 100} if input_data.thread_id else {"recursion_limit": 100}
            
            logging_cfg = load_config("./configs/logging.config", {"logging_enabled": True, "log_path": "./artifacts/agent_audit_trail.json"})
            hitl_cfg = load_config("./configs/hitl.config", {"hitl_enabled": False})
            memory_cfg = load_config("./configs/memory.config", {
                "episodic_memory_enabled": True,
                "semantic_memory_enabled": True,
                "memory_learning_enabled": True,
                "memory_dir": "./artifacts/memory",
                "episodic_db_path": "./app/database/episodic.db",
            })
            
            context_obj = AgentContext(
                session_id=input_data.thread_id or "unknown",
                logging_enabled=logging_cfg.get("logging_enabled", True),
                log_path=logging_cfg.get("log_path", "./artifacts/agent_audit_trail.json"),
                response_mode="chat",
                hitl_enabled=hitl_cfg.get("hitl_enabled", False),
                debug_mode=os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true",
                episodic_memory_enabled=memory_cfg.get("episodic_memory_enabled", True),
                semantic_memory_enabled=memory_cfg.get("semantic_memory_enabled", True),
                memory_learning_enabled=memory_cfg.get("memory_learning_enabled", True),
                memory_dir=memory_cfg.get("memory_dir", "./artifacts/memory"),
                episodic_db_path=memory_cfg.get("episodic_db_path", "./app/database/episodic.db"),
            )
            
            if input_data.thread_id:
                add_message(input_data.thread_id, "user", input_data.message)
            
            full_response = ""
            
            async for event in agent_executor.astream_events(
                {"messages": [("user", input_data.message)]}, 
                config=config,
                context=context_obj,
                version="v2"
            ):
                kind = event["event"]
                
                if kind == "on_tool_start":
                    tool_input = sanitize_text(str(event['data'].get('input', '')))
                    yield f"data: {json.dumps({'type': 'tool_start', 'name': event['name'], 'input': tool_input})}\n\n"
                
                elif kind == "on_tool_end":
                    tool_output = str(event["data"].get("output", ""))
                    truncated = tool_output[:500] + "..." if len(tool_output) > 500 else tool_output
                    yield f"data: {json.dumps({'type': 'tool_end', 'name': event['name'], 'output': sanitize_text(truncated)})}\n\n"
                
                elif kind == "on_chat_model_stream":
                    tags = event.get("tags", [])
                    if "exclude_from_stream" in tags:
                        continue

                    chunk = event["data"]["chunk"]
                    if chunk and chunk.content:
                        normalized = sanitize_text(normalize_content(chunk.content))
                        if normalized:
                            full_response += normalized
                            yield f"data: {json.dumps({'type': 'token', 'content': normalized})}\n\n"

            if not full_response and config:
                try:
                    state = await agent_executor.aget_state(config)
                    messages = state.values.get("messages", []) if state else []
                    if messages:
                        last_msg = messages[-1]
                        from langchain_core.messages import AIMessage
                        if isinstance(last_msg, AIMessage) or getattr(last_msg, "type", None) == "ai":
                            full_response = sanitize_text(normalize_content(last_msg.content))
                            if full_response:
                                yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                except Exception as get_state_err:
                    logger.error(f"Error getting final state: {get_state_err}")

            if input_data.thread_id and full_response:
                add_message(input_data.thread_id, "assistant", full_response)

        except Exception as e:
            logger.error(f"Dynamic stream error in agent '{agent_name}': {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        yield "event: end\ndata: \n\n"

    return StreamingResponse(
        _dynamic_stream_generator(), 
        media_type="text/event-stream"
    )

# --- Chat Session API Endpoints ---
@app.post("/sessions")
def api_create_session(session: SessionCreate):
    try:
        return create_session(session.session_id, session.agent_name, session.title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions")
def api_get_sessions(agent_name: Optional[str] = None):
    try:
        return get_sessions(agent_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/sessions/{session_id}")
def api_delete_session(session_id: str):
    try:
        delete_session(session_id)
        return {"status": "success", "deleted_session": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/messages")
def api_get_messages(session_id: str):
    try:
        return get_messages(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Dynamic Agent Registration (Registry 패턴) ---
# 기존 기동 시점 고정 라우터 마운트 루프는 동적 지연 로딩(Unified Router)으로 대체되어 제거합니다.

@app.get("/health")
def health():
    loaded = list(app.state.agents.keys()) if hasattr(app.state, "agents") else []
    return {"status": "ok", "loaded_agents": loaded}

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Server Port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server Host")
    args = parser.parse_args()
    
    print(f"🚀 Server starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
