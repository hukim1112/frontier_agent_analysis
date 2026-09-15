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
import uuid
from datetime import datetime
import traceback
import inspect
import importlib
from typing import AsyncGenerator, Optional, Dict, Any, List

from fastapi import FastAPI, APIRouter, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.utils.message_utils import sanitize_text, normalize_content
from app.utils.database import create_session, get_sessions, delete_session, add_message, get_messages
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

def _create_agent_context(session_id: Optional[str] = None) -> AgentContext:
    logging_cfg = load_config("./configs/logging.config", {"logging_enabled": False})
    hitl_cfg = load_config("./configs/hitl.config", {"hitl_enabled": False})
    memory_cfg = load_config("./configs/memory.config", {})
    sem_cfg = memory_cfg.get("semantic_memory", {})
    epi_cfg = memory_cfg.get("episodic_memory", {})

    return AgentContext(
        session_id=session_id or "unknown",
        logging_enabled=logging_cfg.get("logging_enabled", False),
        hitl_enabled=hitl_cfg.get("hitl_enabled", False),
        semantic_memory_enabled=sem_cfg.get("enabled", False),
        episodic_memory_enabled=epi_cfg.get("enabled", False),
        memory_learning_enabled=sem_cfg.get("auto_review", False) or epi_cfg.get("auto_finalize", False),
    )

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

class ResumeInput(BaseModel):
    thread_id: str
    decisions: List[Dict[str, Any]]

class JobSubmitInput(BaseModel):
    message: str
    thread_id: Optional[str] = None
    callback_agent: Optional[str] = "main_agent"
    callback_thread_id: Optional[str] = None

# --- In-Memory Job Store for Long-Running Background Tasks ---
job_store: Dict[str, dict] = {}

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
    logger.info("🛑 FastAPI Lifespan Shutdown: Cleaning up resources...")
    # 1. LangGraph checkpointer DB 커넥션 종료
    for name, agent in list(app.state.agents.items()):
        try:
            checkpointer = getattr(agent, "checkpointer", None)
            if checkpointer:
                conn = getattr(checkpointer, "conn", None)
                if conn:
                    logger.info(f"🔌 Closing database connection for agent: {name}")
                    await conn.close()
        except Exception as e:
            logger.error(f"⚠️ Error cleaning up database connection for agent '{name}': {e}")
    # 2. Playwright 브라우저 & Chrome subprocess 종료
    try:
        from app.tools.navigator import PlaywrightManager
        if PlaywrightManager._instance is not None:
            logger.info("🌐 Closing Playwright browser & Chrome subprocess...")
            await PlaywrightManager._instance.close()
            logger.info("✅ Browser cleanup complete")
    except Exception as e:
        logger.error(f"⚠️ Error cleaning up browser: {e}")

# --- App Initialization ---
app = FastAPI(
    title="LLMOps Class Agent Server", 
    version="1.0",
    description="Unified Server for Multiple Agents",
    lifespan=lifespan
)

# --- Static File Serving (artifacts/ 디렉토리를 /artifacts URL로 서빙) ---
artifacts_dir = os.path.join(project_root, "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=artifacts_dir, html=True), name="artifacts")

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
        
        context_obj = _create_agent_context(session_id=input_data.thread_id)
        
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

# --- Long-Running Background Job Worker (Event-Driven Reactive Wakeup) ---
async def _run_agent_job(job_id: str, agent_name: str, input_data: JobSubmitInput, app: FastAPI):
    job = job_store.get(job_id)
    if not job:
        return
    job["status"] = "RUNNING"
    logger.info(f"⚙️ [Job {job_id}] Started background execution for agent '{agent_name}'")

    try:
        # 1. 서브에이전트 로드 & 실행
        agent_executor = await get_or_load_agent(agent_name, app)
        config = {"configurable": {"thread_id": input_data.thread_id}, "recursion_limit": 100} if input_data.thread_id else {"recursion_limit": 100}

        context_obj = _create_agent_context(session_id=input_data.thread_id)

        if input_data.thread_id:
            add_message(input_data.thread_id, "user", input_data.message)

        result = await agent_executor.ainvoke(
            {"messages": [("user", input_data.message)]},
            config=config,
            context=context_obj
        )
        last_message = result["messages"][-1]
        task_report = sanitize_text(normalize_content(last_message.content))

        if input_data.thread_id:
            add_message(input_data.thread_id, "assistant", task_report)

        job["status"] = "SUCCESS"
        job["completed_at"] = datetime.utcnow().isoformat() + "Z"
        job["result"] = task_report
        logger.info(f"✅ [Job {job_id}] Sub-agent '{agent_name}' completed successfully")

        # 2. 🌟 Reactive Agent Wakeup: 서브에이전트 완료 즉시 Supervisor 자동 호출 & 후속 작업 수행
        if input_data.callback_agent:
            try:
                cb_agent = input_data.callback_agent
                cb_thread_id = input_data.callback_thread_id or f"session_{job_id}"
                logger.info(f"🚀 [Job {job_id}] Triggering reactive wakeup for '{cb_agent}' (Thread: {cb_thread_id})...")

                cb_executor = await get_or_load_agent(cb_agent, app)

                trigger_prompt = (
                    f"[SYSTEM NOTIFICATION: BACKGROUND TASK COMPLETED]\n"
                    f"- Job ID: {job_id}\n"
                    f"- Sub-Agent: {agent_name}\n"
                    f"- Report Content:\n{task_report}\n\n"
                    f"[INSTRUCTION FOR SUPERVISOR]\n"
                    f"위 서브에이전트의 완료 보고서와 생성된 산출물을 바탕으로 다음 작업을 수행하거나 유저에게 답변합니다 "
                    f"생성된 차트 이미지나 HTML 대시보드가 있다면 UI 렌더링 태그(<Render_HTML>, <Render_Image>, <Render_File>)를 통해 출력 가능합니다."
                )

                cb_config = {"configurable": {"thread_id": cb_thread_id}, "recursion_limit": 100}
                add_message(cb_thread_id, "user", trigger_prompt)

                sup_result = await cb_executor.ainvoke(
                    {"messages": [("user", trigger_prompt)]},
                    config=cb_config,
                    context=context_obj
                )
                sup_last = sup_result["messages"][-1]
                sup_response = sanitize_text(normalize_content(sup_last.content))
                add_message(cb_thread_id, "assistant", sup_response)
                job["supervisor_response"] = sup_response
                logger.info(f"🎉 [Job {job_id}] Reactive Wakeup completed! Supervisor produced final response.")
            except Exception as cb_err:
                logger.error(f"⚠️ [Job {job_id}] Error during reactive wakeup of '{input_data.callback_agent}': {cb_err}")
                traceback.print_exc()

    except Exception as e:
        logger.error(f"❌ [Job {job_id}] Background execution failed: {e}")
        traceback.print_exc()
        job["status"] = "FAILED"
        job["completed_at"] = datetime.utcnow().isoformat() + "Z"
        job["error"] = str(e)


# --- Job Management Endpoints ---
@app.post("/agents/{agent_name}/jobs")
async def submit_agent_job(
    agent_name: str,
    input_data: JobSubmitInput,
    background_tasks: BackgroundTasks,
    request: Request
):
    """비동기 백그라운드 작업을 등록하고 job_id를 즉시 반환합니다."""
    # 에이전트 등록 여부 확인
    available = [a["name"] for a in api_list_agents() if isinstance(a, dict)]
    if agent_name not in available:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found. Available: {available}")

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    created_at = datetime.utcnow().isoformat() + "Z"
    job_store[job_id] = {
        "job_id": job_id,
        "agent_name": agent_name,
        "status": "SUBMITTED",
        "created_at": created_at,
        "completed_at": None,
        "result": None,
        "error": None,
        "supervisor_response": None,
        "callback_agent": input_data.callback_agent,
        "callback_thread_id": input_data.callback_thread_id,
    }

    background_tasks.add_task(_run_agent_job, job_id, agent_name, input_data, request.app)
    return {"job_id": job_id, "status": "SUBMITTED", "agent_name": agent_name}


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """특정 작업의 진행 상태와 결과(및 Supervisor 후속 보고서)를 조회합니다."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.get("/jobs")
async def list_jobs():
    """모든 백그라운드 작업 목록을 조회합니다."""
    return list(job_store.values())


@app.get("/sessions/{thread_id}/jobs")
async def list_session_jobs(thread_id: str):
    """특정 세션(thread)에 연관된 모든 백그라운드 작업 목록을 반환합니다.
    callback_thread_id로 필터링하여 해당 대화 세션에서 생성된 Job만 조회합니다."""
    return [
        job for job in job_store.values()
        if job.get("callback_thread_id") == thread_id
    ]

@app.post("/agents/{agent_name}/stream")
async def stream_agent(agent_name: str, input_data: StreamInput, request: Request):
    try:
        agent_executor = await get_or_load_agent(agent_name, request.app)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    async def _dynamic_stream_generator() -> AsyncGenerator[str, None]:
        try:
            config = {"configurable": {"thread_id": input_data.thread_id}, "recursion_limit": 100} if input_data.thread_id else {"recursion_limit": 100}
            
            context_obj = _create_agent_context(session_id=input_data.thread_id)
            
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
                    yield f"data: {json.dumps({'type': 'tool_start', 'name': event['name'], 'input': tool_input, 'run_id': event.get('run_id', '')})}\n\n"
                
                elif kind == "on_tool_end":
                    tool_output = str(event["data"].get("output", ""))
                    truncated = tool_output[:500] + "..." if len(tool_output) > 500 else tool_output
                    yield f"data: {json.dumps({'type': 'tool_end', 'name': event['name'], 'output': sanitize_text(truncated), 'run_id': event.get('run_id', '')})}\n\n"
                
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
                    # HITL: interrupt 상태 확인 및 전달
                    if state and state.tasks:
                        for task in state.tasks:
                            if task.interrupts:
                                interrupt_data = []
                                for intr in task.interrupts:
                                    # intr.value는 HITLRequest dict (action_requests, review_configs)
                                    value = intr.value
                                    # JSON 직렬화 가능하도록 변환
                                    if hasattr(value, '__dict__'):
                                        value = dict(value)
                                    interrupt_data.append({
                                        "value": value,
                                        "id": str(intr.id) if hasattr(intr, 'id') else ""
                                    })
                                yield f"data: {json.dumps({'type': 'interrupt', 'interrupts': interrupt_data}, default=str)}\n\n"
                    
                    # 기존: 빈 응답 시 마지막 AI 메시지 추출
                    if not full_response:
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

# --- HITL Resume Endpoint ---
@app.post("/agents/{agent_name}/resume")
async def resume_agent(agent_name: str, input_data: ResumeInput, request: Request):
    """interrupt된 에이전트를 사용자 결정과 함께 재개하고 SSE로 스트리밍합니다."""
    try:
        agent_executor = await get_or_load_agent(agent_name, request.app)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    from langgraph.types import Command
    
    async def _resume_stream_generator() -> AsyncGenerator[str, None]:
        try:
            config = {"configurable": {"thread_id": input_data.thread_id}, "recursion_limit": 100}
            resume_command = Command(resume={"decisions": input_data.decisions})
            
            full_response = ""
            
            async for event in agent_executor.astream_events(
                resume_command,
                config=config,
                version="v2"
            ):
                kind = event["event"]
                
                if kind == "on_tool_start":
                    tool_input = sanitize_text(str(event['data'].get('input', '')))
                    yield f"data: {json.dumps({'type': 'tool_start', 'name': event['name'], 'input': tool_input, 'run_id': event.get('run_id', '')})}\n\n"
                
                elif kind == "on_tool_end":
                    tool_output = str(event["data"].get("output", ""))
                    truncated = tool_output[:500] + "..." if len(tool_output) > 500 else tool_output
                    yield f"data: {json.dumps({'type': 'tool_end', 'name': event['name'], 'output': sanitize_text(truncated), 'run_id': event.get('run_id', '')})}\n\n"
                
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
            
            # Resume 후에도 다시 interrupt가 발생할 수 있음 (연쇄 승인)
            if config:
                try:
                    state = await agent_executor.aget_state(config)
                    if state and state.tasks:
                        for task in state.tasks:
                            if task.interrupts:
                                interrupt_data = []
                                for intr in task.interrupts:
                                    value = intr.value
                                    if hasattr(value, '__dict__'):
                                        value = dict(value)
                                    interrupt_data.append({
                                        "value": value,
                                        "id": str(intr.id) if hasattr(intr, 'id') else ""
                                    })
                                yield f"data: {json.dumps({'type': 'interrupt', 'interrupts': interrupt_data}, default=str)}\n\n"
                    
                    if not full_response:
                        messages = state.values.get("messages", []) if state else []
                        if messages:
                            last_msg = messages[-1]
                            from langchain_core.messages import AIMessage
                            if isinstance(last_msg, AIMessage) or getattr(last_msg, "type", None) == "ai":
                                full_response = sanitize_text(normalize_content(last_msg.content))
                                if full_response:
                                    yield f"data: {json.dumps({'type': 'token', 'content': full_response})}\n\n"
                except Exception as get_state_err:
                    logger.error(f"Error getting state after resume: {get_state_err}")
            
            if input_data.thread_id and full_response:
                add_message(input_data.thread_id, "assistant", full_response)
            
        except Exception as e:
            logger.error(f"Resume stream error in agent '{agent_name}': {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        yield "event: end\ndata: \n\n"
    
    return StreamingResponse(
        _resume_stream_generator(),
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
