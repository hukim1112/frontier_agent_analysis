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
else:
    print("Warning: .env file not found.")

# LangSmith Tracing 상태 확인 (.env 및 실행 환경 변수를 덮어쓰지 않고 LangSmith SDK 규칙 그대로 판별)
def _inspect_langsmith_status() -> tuple[bool, str, str, bool]:
    """실제 LangSmith SDK(langsmith.utils)와 동일한 우선순위로 트레이싱 활성화 여부를 판별합니다."""
    try:
        from langsmith.utils import get_env_var, get_tracer_project, tracing_is_enabled

        # load_dotenv() 직후 호출되므로 lru_cache를 비워 최신 환경변수를 반영
        if hasattr(get_env_var, "cache_clear"):
            get_env_var.cache_clear()
        if hasattr(get_tracer_project, "cache_clear"):
            get_tracer_project.cache_clear()

        enabled = bool(tracing_is_enabled())
        project = get_tracer_project() or "default"
        api_key = (get_env_var("API_KEY") or "").strip()
    except ImportError:
        def _env_first(*keys: str, default: str = "") -> str:
            for key in keys:
                val = os.environ.get(key)
                if val is not None and val.strip() != "":
                    return val
            return default

        tracing_val = _env_first(
            "LANGSMITH_TRACING_V2",
            "LANGCHAIN_TRACING_V2",
            "LANGSMITH_TRACING",
            "LANGCHAIN_TRACING",
        )
        enabled = tracing_val == "true"
        project = _env_first(
            "HOSTED_LANGSERVE_PROJECT_NAME",
            "LANGSMITH_PROJECT",
            "LANGCHAIN_PROJECT",
            "LANGSMITH_SESSION",
            "LANGCHAIN_SESSION",
            default="default",
        )
        api_key = _env_first("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY").strip()

    # LANGSMITH_TRACING=true로 설정했지만 *_TRACING_V2=false가 우선해 꺼진 흔한 실수 감지
    v2_val = os.environ.get("LANGSMITH_TRACING_V2")
    if v2_val is None or v2_val.strip() == "":
        v2_val = os.environ.get("LANGCHAIN_TRACING_V2")
    v1_val = os.environ.get("LANGSMITH_TRACING")
    if v1_val is None or v1_val.strip() == "":
        v1_val = os.environ.get("LANGCHAIN_TRACING")
    overridden_by_v2 = (
        not enabled
        and v1_val == "true"
        and v2_val is not None
        and v2_val.strip() != ""
        and v2_val != "true"
    )
    return enabled, project, api_key, overridden_by_v2


_ls_enabled, _ls_project, _ls_api_key, _ls_v2_override = _inspect_langsmith_status()
if _ls_enabled:
    if not _ls_api_key or _ls_api_key in ("your_langchain_api_key_here", "your-langsmith-key"):
        print(f"⚠️ LangSmith Tracing is enabled (Project: {_ls_project}), but LANGCHAIN_API_KEY/LANGSMITH_API_KEY is missing or placeholder.")
    else:
        print(f"📈 LangSmith Tracing Enabled. Project: {_ls_project}")
elif _ls_v2_override:
    print("ℹ️ LangSmith Tracing Disabled (LANGCHAIN_TRACING_V2/LANGSMITH_TRACING_V2 takes precedence over LANGSMITH_TRACING; set LANGCHAIN_TRACING_V2=true in .env to enable).")
else:
    print("ℹ️ LangSmith Tracing Disabled (set LANGCHAIN_TRACING_V2=true in .env to enable).")

import logging
import json
import time
import uuid
from datetime import datetime, timezone
import traceback
import inspect
import importlib
from typing import AsyncGenerator, Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.utils.message_utils import sanitize_text, normalize_content
from app.utils.database import (
    init_db,
    create_session, get_sessions, delete_session, get_messages,
    register_subagent_thread,
    get_shared_checkpointer, close_shared_checkpointer,
)
from app.context import AgentContext
from app.middleware.observability.agent_log_tracer import AgentLogTracer, _render_snapshot_report
from app.tools.plan import plan_session_scope
from app.utils.workspace import resolve_session_workspace

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMOps_Server")

DEFAULT_RECURSION_LIMIT = 250


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_config(file_path: str, default: dict) -> dict:
    try:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading config {file_path}: {e}")
    return default


def _create_agent_context(
    session_id: Optional[str] = None,
    agent_name: str = "",
    user_id: Optional[str] = None,
    parent_thread_id: Optional[str] = None,
    force_disable_hitl: bool = False,
) -> AgentContext:
    hitl_cfg = load_config("./configs/hitl.config", {"hitl_enabled": False})
    memory_cfg = load_config("./configs/memory.config", {})
    sem_cfg = memory_cfg.get("semantic_memory", {})
    epi_cfg = memory_cfg.get("episodic_memory", {})
    tr_cfg = memory_cfg.get("transcript", {})

    resolved_user_id = (user_id or "default").strip() or "default"
    resolved_parent_tid = (parent_thread_id or "").strip()
    ws = resolve_session_workspace(
        user_id=resolved_user_id,
        session_id=session_id or "default",
        parent_session_id=resolved_parent_tid or None,
        subagent_role=agent_name if resolved_parent_tid else None,
    )

    hitl_enabled = False if (force_disable_hitl or resolved_parent_tid) else hitl_cfg.get("hitl_enabled", False)
    sem_enabled = sem_cfg.get("enabled", True)
    epi_enabled = epi_cfg.get("enabled", True)
    sem_learning = sem_cfg.get("auto_review", True)
    epi_learning = epi_cfg.get("auto_finalize", True)
    tr_enabled = tr_cfg.get("enabled", True)

    return AgentContext(
        agent_name=agent_name,
        session_id=session_id or "unknown",
        parent_session_id=resolved_parent_tid,
        user_id=resolved_user_id,
        workspace_dir=ws.rel_root,
        hitl_enabled=hitl_enabled,
        semantic_memory_enabled=sem_enabled,
        semantic_learning_enabled=sem_learning,
        episodic_memory_enabled=epi_enabled,
        episodic_learning_enabled=epi_learning,
        transcript_enabled=tr_enabled,
    )


async def _resolve_run(
    agent_name: str,
    thread_id: Optional[str],
    user_id: Optional[str] = None,
    parent_thread_id: Optional[str] = None,
    force_disable_hitl: bool = False,
) -> tuple[AgentContext, dict]:
    """요청 1건의 실행 설정(AgentContext 및 LangGraph run config)을 해석하고 부모-자식 스레드 관계를 기록합니다."""
    if thread_id and parent_thread_id:
        await register_subagent_thread(thread_id, parent_thread_id)

    context_obj = _create_agent_context(
        session_id=thread_id,
        agent_name=agent_name,
        user_id=user_id,
        parent_thread_id=parent_thread_id,
        force_disable_hitl=force_disable_hitl,
    )

    configurable: Dict[str, Any] = {
        "user_id": context_obj.user_id,
        "workspace_dir": context_obj.workspace_dir,
    }
    if thread_id:
        configurable["thread_id"] = thread_id
    if context_obj.parent_session_id:
        configurable["parent_thread_id"] = context_obj.parent_session_id

    run_config: dict = {
        "recursion_limit": DEFAULT_RECURSION_LIMIT,
        "configurable": configurable,
    }
    return context_obj, run_config


# --- Schemas ---
class UserInput(BaseModel):
    message: str
    thread_id: Optional[str] = None
    parent_thread_id: Optional[str] = None
    user_id: Optional[str] = None


class StreamInput(UserInput):
    stream_tokens: bool = Field(default=True)


class ChatMessage(BaseModel):
    type: str
    content: str


class SessionCreate(BaseModel):
    session_id: str
    agent_name: str
    title: str
    user_id: Optional[str] = "default"


class ResumeInput(BaseModel):
    thread_id: str
    decisions: List[Dict[str, Any]]
    user_id: Optional[str] = None


class JobSubmitInput(BaseModel):
    message: str
    thread_id: Optional[str] = None
    parent_thread_id: Optional[str] = None
    callback_agent: Optional[str] = "main_agent"
    callback_thread_id: Optional[str] = None
    user_id: Optional[str] = None


class JobCancelInput(BaseModel):
    reason: Optional[str] = None
    caller_thread_id: Optional[str] = None


# --- Router Factory & Async Lock Store ---
from contextlib import asynccontextmanager
import asyncio

# --- In-Memory Job Store & Unified Thread-Level Execution Lock ---
job_store: Dict[str, dict] = {}
job_tasks: Dict[str, asyncio.Task] = {}
MAX_CONSECUTIVE_FAILED_WAKEUPS = 3
thread_locks: Dict[str, asyncio.Lock] = {}
thread_inflight_counts: Dict[str, int] = {}
wakeup_locks = thread_locks


def get_thread_lock(thread_id: Optional[str]) -> asyncio.Lock:
    """동일 thread_id에 대한 동시 실행(Double-Texting 및 Reactive Wakeup 충돌)을 직렬화하는 통합 Lock을 반환합니다."""
    key = thread_id or "__anon__"
    lock = thread_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        thread_locks[key] = lock
    return lock


@asynccontextmanager
async def acquire_thread_slot(thread_id: Optional[str]):
    """thread_locks 획득과 thread_inflight_counts 증감을 원자적으로 관리하는 컨텍스트 매니저."""
    if not thread_id:
        yield
        return
    lock = get_thread_lock(thread_id)
    thread_inflight_counts[thread_id] = thread_inflight_counts.get(thread_id, 0) + 1
    try:
        async with lock:
            yield
    finally:
        remaining = thread_inflight_counts.get(thread_id, 1) - 1
        if remaining <= 0:
            thread_inflight_counts.pop(thread_id, None)
        else:
            thread_inflight_counts[thread_id] = remaining


# --- Lifespan Context Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 FastAPI Lifespan Startup: Initializing database, shared checkpointer, and agent storage...")
    init_db()
    app.state.agents = {}
    app.state.load_lock = asyncio.Lock()
    app.state.checkpointer = await get_shared_checkpointer()
    yield
    logger.info("🛑 FastAPI Lifespan Shutdown: Cleaning up resources...")
    # 0. 실행 중인 백그라운드 작업 태스크 취소 및 대기
    pending_tasks = [t for t in list(job_tasks.values()) if not t.done()]
    for t in pending_tasks:
        t.cancel()
    if pending_tasks:
        await asyncio.wait(pending_tasks, timeout=3.0)
    job_tasks.clear()
    # 1. 공유 LangGraph checkpointer, MemoryMiddleware 백그라운드 리뷰/스토어, 개별 커넥션 종료
    await close_shared_checkpointer()
    try:
        from app.middleware.memory import MemoryMiddleware
        await MemoryMiddleware.aclose_all_instances(review_timeout=5.0)
    except Exception as e:
        logger.error(f"⚠️ Error closing MemoryMiddleware instances: {e}")

    for name, agent in list(app.state.agents.items()):
        try:
            checkpointer = getattr(agent, "checkpointer", None)
            conn = getattr(checkpointer, "conn", None) if checkpointer else None
            if conn is not None and getattr(conn, "_running", False):
                logger.info(f"🔌 Closing database connection for agent: {name}")
                await conn.close()
        except Exception as e:
            logger.error(f"⚠️ Error cleaning up database connection for agent '{name}': {e}")
    # 2. 공유 AsyncAgentClient 커넥션 풀 종료
    try:
        from app.tools.supervisor_tools import _shared_client
        await _shared_client.aclose()
    except Exception as e:
        logger.error(f"⚠️ Error closing shared AsyncAgentClient: {e}")
    # 3. Playwright 브라우저 & Chrome subprocess 종료
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
    lifespan=lifespan,
)

# --- Static File Serving (artifacts/ 디렉토리를 /artifacts URL로 서빙하되 memory/logs/plan/db 내부 파일은 차단) ---
_BLOCKED_ARTIFACT_SEGMENTS = {"memory", "logs", "plan"}
_BLOCKED_ARTIFACT_EXTS = (".db", ".db-wal", ".db-shm", ".jsonl")


@app.middleware("http")
async def _guard_sensitive_artifacts(request: Request, call_next):
    path = request.url.path or ""
    if path.startswith("/artifacts/"):
        parts = [seg.lower() for seg in path.split("/") if seg]
        if any(seg in _BLOCKED_ARTIFACT_SEGMENTS for seg in parts) or path.lower().endswith(_BLOCKED_ARTIFACT_EXTS):
            return JSONResponse(
                status_code=403,
                content={"detail": "Access to internal memory, plan, or audit log files via /artifacts is forbidden."},
            )
    return await call_next(request)


artifacts_dir = os.path.join(project_root, "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)
app.mount("/artifacts", StaticFiles(directory=artifacts_dir, html=True), name="artifacts")


# --- Dynamic Agent Loader ---
def _find_agent_module_path(agent_name: str, agents_dir: str) -> Optional[str]:
    """에이전트 이름에 대해 단일 .py 파일 또는 디렉터리 패키지 모듈 경로를 탐색합니다."""
    # 1. 단일 파일 에이전트: app/agents/{agent_name}.py
    file_path = os.path.join(agents_dir, f"{agent_name}.py")
    if os.path.isfile(file_path):
        return f"app.agents.{agent_name}"

    # 2. 디렉터리형 에이전트 패키지: app/agents/{agent_name}/
    pkg_dir = os.path.join(agents_dir, agent_name)
    if os.path.isdir(pkg_dir):
        if os.path.isfile(os.path.join(pkg_dir, "agent.py")):
            return f"app.agents.{agent_name}.agent"
        elif os.path.isfile(os.path.join(pkg_dir, "__init__.py")):
            return f"app.agents.{agent_name}"
    return None


def _is_hot_reload_enabled() -> bool:
    """개발 모드(ENV=dev 또는 AGENT_HOT_RELOAD=true)에서만 importlib.reload를 활성화합니다."""
    env = os.getenv("ENV", "").lower()
    hot_reload = os.getenv("AGENT_HOT_RELOAD", "").lower()
    return env in ("dev", "development", "local") or hot_reload in ("1", "true", "yes")


async def get_or_load_agent(agent_name: str, app: FastAPI) -> Any:
    """
    런타임 비동기 이벤트 루프 하에서 에이전트 모듈을 지연 로딩하고 캐싱합니다.
    단일 .py 파일 에이전트와 디렉터리형 에이전트 패키지(agent.py / __init__.py)를 모두 지원합니다.
    """
    if not hasattr(app.state, "agents"):
        app.state.agents = {}
    if not hasattr(app.state, "load_lock"):
        app.state.load_lock = asyncio.Lock()

    if agent_name in app.state.agents:
        return app.state.agents[agent_name]

    agents_dir = os.path.join(project_root, "app", "agents")
    module_path = _find_agent_module_path(agent_name, agents_dir)
    if not module_path:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found on server.")

    async with app.state.load_lock:
        if agent_name in app.state.agents:
            return app.state.agents[agent_name]

        try:
            if module_path in sys.modules:
                if _is_hot_reload_enabled():
                    module = importlib.reload(sys.modules[module_path])
                    logger.info(f"🔄 Reloaded agent module (dev mode): {module_path}")
                else:
                    module = sys.modules[module_path]
            else:
                module = importlib.import_module(module_path)
                logger.info(f"✅ Loaded agent module: {module_path}")

            factory = getattr(module, "create_agent_executor", None)
            if not factory:
                factory = getattr(module, f"create_{agent_name}_agent", None)
                if not factory:
                    factory = getattr(module, "get_agent_executor", None)

            if not factory:
                raise AttributeError(f"Module '{module_path}' must expose 'create_agent_executor' function.")

            shared_cp = getattr(app.state, "checkpointer", None)
            sig = inspect.signature(factory)
            accepts_cp = "checkpointer" in sig.parameters

            if inspect.iscoroutinefunction(factory):
                executor = await (factory(checkpointer=shared_cp) if accepts_cp else factory())
            else:
                executor = factory(checkpointer=shared_cp) if accepts_cp else factory()

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
    app/agents/ 폴더 내 단일 파이썬 파일 및 디렉터리 패키지를 스캔하여 사용 가능한 에이전트 목록을 동적으로 리턴합니다.
    """
    agents_dir = os.path.join(project_root, "app", "agents")
    available_agents = []

    if not os.path.exists(agents_dir):
        return []

    for item in sorted(os.listdir(agents_dir)):
        if item.startswith("__") or item.startswith(".") or item in ("utils.py", "common"):
            continue

        if not (item.endswith(".py") or os.path.isdir(os.path.join(agents_dir, item))):
            continue

        agent_name = item[:-3] if item.endswith(".py") else item
        module_path = _find_agent_module_path(agent_name, agents_dir)
        if not module_path:
            continue

        description = f"Runtime loaded {agent_name} agent"
        # roles 미표기 에이전트는 UI 대화 전용(main)으로 간주 — 서브에이전트 위임 대상에 실수로 노출되지 않도록
        default_roles = ["main"]
        try:
            if module_path in sys.modules:
                module = sys.modules[module_path]
            else:
                module = importlib.import_module(module_path)

            metadata = getattr(module, "AGENT_METADATA", None)
            if metadata and isinstance(metadata, dict):
                name = metadata.get("name", agent_name)
                description = metadata.get("description", description)
                roles = metadata.get("roles") or default_roles
                if isinstance(roles, str):
                    roles = [roles]
                available_agents.append({"name": name, "description": description, "roles": list(roles)})
            else:
                available_agents.append({"name": agent_name, "description": description, "roles": default_roles})
        except Exception as scan_err:
            logger.warning(f"⚠️ Failed to parse metadata for {agent_name}: {scan_err}")
            available_agents.append({"name": agent_name, "description": description, "roles": default_roles})

    return available_agents


# --- Unified Dynamic Routing ---
@app.post("/agents/{agent_name}/invoke", response_model=ChatMessage)
async def invoke_agent(agent_name: str, input_data: UserInput, request: Request):
    agent_executor = await get_or_load_agent(agent_name, request.app)
    context_obj, config = await _resolve_run(
        agent_name,
        input_data.thread_id,
        user_id=input_data.user_id,
        parent_thread_id=input_data.parent_thread_id,
    )
    try:
        async def _do_invoke() -> ChatMessage:
            with plan_session_scope(
                input_data.thread_id or "default",
                user_id=context_obj.user_id,
                parent_session_id=context_obj.parent_session_id,
            ):
                result = await agent_executor.ainvoke(
                    {"messages": [("user", input_data.message)]},
                    config=config,
                    context=context_obj,
                )
            last_message = result["messages"][-1]
            normalized = sanitize_text(normalize_content(last_message.content))
            return ChatMessage(type="ai", content=normalized)

        async with acquire_thread_slot(input_data.thread_id):
            return await _do_invoke()
    except Exception as e:
        logger.error(f"Dynamic invocation error in agent '{agent_name}': {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# --- Long-Running Background Job Worker (Event-Driven Reactive Wakeup) ---
def _verify_job_caller(job: dict, caller_thread_id: Optional[str]) -> None:
    """caller_thread_id가 전달된 경우 대상 Job의 부모 세션 소유권을 검증합니다 (불일치 시 403)."""
    caller_tid = (caller_thread_id or "").strip()
    if not caller_tid:
        return
    owners = {
        tid
        for tid in (job.get("callback_thread_id"), job.get("parent_thread_id"))
        if tid
    }
    if owners and caller_tid not in owners:
        raise HTTPException(
            status_code=403,
            detail=f"Job '{job.get('job_id')}' does not belong to session '{caller_tid}'.",
        )


def _effective_job_start_ts(job: dict) -> Optional[float]:
    if "execution_started_at_ts" in job:
        ts = job.get("execution_started_at_ts")
        return float(ts) if ts is not None else None
    ts = job.get("created_at_ts")
    return float(ts) if ts is not None else None


def _snapshot_belongs_to_job(job: dict, snap: dict) -> bool:
    """현재 sub_thread_id의 스냅샷이 주어진 job에 속하는지 시작 시각 기준으로 판별합니다."""
    job_id = job.get("job_id")
    sub_tid = job.get("sub_thread_id")
    if not job_id or not sub_tid:
        return True

    if "execution_started_at_ts" in job and job.get("execution_started_at_ts") is None:
        return False

    same_thread_jobs = [
        (idx, j)
        for idx, j in enumerate(job_store.values())
        if j.get("sub_thread_id") == sub_tid
    ]
    if not same_thread_jobs:
        return True

    snap_started_at = snap.get("started_at")
    if snap_started_at is not None:
        snap_ts = float(snap_started_at)
        candidates = []
        for idx, j in same_thread_jobs:
            eff_ts = _effective_job_start_ts(j)
            if eff_ts is not None and eff_ts <= snap_ts + 0.005:
                candidates.append((eff_ts, idx, j.get("job_id")))
        if candidates:
            _, _, owner_job_id = max(candidates, key=lambda item: (item[0], item[1]))
            return owner_job_id == job_id
        if any(_effective_job_start_ts(j) is not None for _, j in same_thread_jobs):
            return False

    started_jobs = [
        (idx, j)
        for idx, j in same_thread_jobs
        if not ("execution_started_at_ts" in j and j.get("execution_started_at_ts") is None)
    ]
    if not started_jobs:
        return False
    return started_jobs[-1][1].get("job_id") == job_id


def _enrich_job_with_progress(job: dict, include_terminal: bool = True) -> dict:
    """job_store 레코드에 AgentLogTracer 실시간/디스크 진행 스냅샷을 결합한 사본을 반환합니다."""
    enriched = dict(job)
    sub_tid = job.get("sub_thread_id")
    if not sub_tid:
        return enriched

    status = job.get("status", "")
    is_active = status in ("SUBMITTED", "RUNNING")
    if not include_terminal and not is_active:
        return enriched
    if "execution_started_at_ts" in job and job.get("execution_started_at_ts") is None:
        return enriched

    parent_sid = job.get("parent_thread_id") or job.get("callback_thread_id")
    snap = AgentLogTracer.get_session_snapshot(
        session_id=sub_tid,
        user_id=job.get("user_id"),
        parent_session_id=parent_sid,
        workspace_dir=job.get("workspace_dir"),
        recent_limit=5,
        allow_disk_fallback=include_terminal,
    )
    if not snap.get("found"):
        return enriched
    if not _snapshot_belongs_to_job(job, snap):
        return enriched
    if not is_active and snap.get("status") in ("RUNNING", "INTERRUPTED"):
        return enriched
    if status == "SUBMITTED" and snap.get("status") in ("SUCCESS", "FAILED", "CANCELLED"):
        return enriched

    enriched["progress"] = snap
    enriched["progress_report"] = _render_snapshot_report(snap)
    return enriched


def _count_consecutive_failures(callback_thread_id: Optional[str], agent_name: str) -> int:
    """동일 부모 세션의 특정 서브에이전트 최근 연속 실패 횟수를 완료 시각 역순으로 계산합니다."""
    matched = [
        (idx, j)
        for idx, j in enumerate(job_store.values())
        if j.get("callback_thread_id") == callback_thread_id
        and j.get("agent_name") == agent_name
        and j.get("status") in ("SUCCESS", "FAILED")
    ]
    matched.sort(
        key=lambda item: (item[1].get("completed_at") or item[1].get("created_at") or "", item[0]),
        reverse=True,
    )
    count = 0
    for _, j in matched:
        if j.get("status") == "FAILED":
            count += 1
        elif j.get("status") == "SUCCESS":
            break
    return count


async def _cleanup_cancelled_subagent_resources(sub_thread_id: Optional[str]) -> int:
    """취소된 서브에이전트 세션이 점유한 브라우저 탭 등 외부 리소스를 안전하게 정리합니다."""
    if not sub_thread_id:
        return 0
    try:
        from app.tools.navigator import PlaywrightManager

        if PlaywrightManager._instance is not None:
            return await PlaywrightManager._instance.close_session_tabs(sub_thread_id)
    except Exception as err:
        logger.warning(f"⚠️ Failed to clean up browser tabs for '{sub_thread_id}': {err}")
    return 0


async def _trigger_reactive_wakeup(
    job_id: str,
    agent_name: str,
    input_data: JobSubmitInput,
    context_obj: AgentContext,
    app: FastAPI,
    outcome: str,
    task_report: str = "",
    error_text: str = "",
) -> None:
    """서브에이전트 SUCCESS 또는 FAILED 직후 Supervisor를 비동기 재호출하고 wakeup_status를 추적합니다."""
    job = job_store.get(job_id)
    if not input_data.callback_agent:
        if job is not None:
            job["wakeup_status"] = "SKIPPED"
        return

    cb_agent = input_data.callback_agent
    cb_thread_id = input_data.callback_thread_id or f"session_{job_id}"
    fail_count = 0

    if outcome == "FAILED":
        fail_count = _count_consecutive_failures(cb_thread_id, agent_name)
        if fail_count > MAX_CONSECUTIVE_FAILED_WAKEUPS:
            msg = (
                f"Suppressed reactive wakeup after {fail_count} consecutive failures "
                f"for '{agent_name}'."
            )
            if job is not None:
                job["wakeup_status"] = "SKIPPED"
                job["wakeup_error"] = msg
            logger.warning(f"🧯 [Job {job_id}] {msg}")
            return

        telemetry = ""
        if input_data.thread_id:
            telemetry = AgentLogTracer.format_session_report(
                session_id=input_data.thread_id,
                user_id=getattr(context_obj, "user_id", None) or input_data.user_id,
                parent_session_id=input_data.parent_thread_id or input_data.callback_thread_id,
                workspace_dir=getattr(context_obj, "workspace_dir", None),
            )
        telemetry_section = f"- Execution Telemetry:\n{telemetry}\n" if telemetry else ""
        trigger_prompt = (
            f"[SYSTEM NOTIFICATION: BACKGROUND TASK FAILED]\n"
            f"- Job ID: {job_id}\n"
            f"- Sub-Agent: {agent_name}\n"
            f"- Sub-Thread ID: {input_data.thread_id} (use sub_thread_id='{input_data.thread_id}' to inspect/continue if appropriate)\n"
            f"- Consecutive Failures for '{agent_name}': {max(1, fail_count)}/{MAX_CONSECUTIVE_FAILED_WAKEUPS}\n"
            f"- Error Detail:\n{error_text}\n"
            f"{telemetry_section}\n"
            f"[INSTRUCTION FOR SUPERVISOR]\n"
            f"서브에이전트 백그라운드 작업이 실패했습니다. 태스크 상태를 `task_update(..., 'BLOCKED')`로 갱신하고 오류 원인과 텔레메트리를 점검하세요. "
            f"근거 있는 대안이나 수정 전략이 있다면 계획을 수정(`enter_plan(intent='revise')`)한 뒤 재위임하고, "
            f"동일 실패를 근거 없이 반복하지 마세요. 복구가 불가능하거나 사용자 조치가 필요하면 원인과 현재 상태를 명확히 보고하세요."
        )
    else:
        trigger_prompt = (
            f"[SYSTEM NOTIFICATION: BACKGROUND TASK COMPLETED]\n"
            f"- Job ID: {job_id}\n"
            f"- Sub-Agent: {agent_name}\n"
            f"- Sub-Thread ID: {input_data.thread_id} (use sub_thread_id='{input_data.thread_id}' to continue this agent)\n"
            f"- Report Content:\n{task_report}\n\n"
            f"[INSTRUCTION FOR SUPERVISOR]\n"
            f"위 서브에이전트의 완료 보고서와 생성된 산출물을 바탕으로 다음 작업을 수행하거나 유저에게 답변합니다 "
            f"생성된 차트 이미지나 HTML 대시보드가 있다면 UI 렌더링 태그(<Render_HTML>, <Render_Image>, <Render_File>)를 통해 출력 가능합니다."
        )

    if job is not None:
        job["wakeup_status"] = "RUNNING"

    try:
        logger.info(
            f"🚀 [Job {job_id}] Triggering reactive wakeup ({outcome}) for '{cb_agent}' (Thread: {cb_thread_id})..."
        )
        cb_executor = await get_or_load_agent(cb_agent, app)
        cb_user_id = getattr(context_obj, "user_id", None) or input_data.user_id
        cb_context_obj, cb_config = await _resolve_run(
            cb_agent, cb_thread_id, user_id=cb_user_id, force_disable_hitl=True
        )
        async with acquire_thread_slot(cb_thread_id):
            with plan_session_scope(cb_thread_id, user_id=cb_context_obj.user_id):
                sup_result = await cb_executor.ainvoke(
                    {"messages": [("user", trigger_prompt)]},
                    config=cb_config,
                    context=cb_context_obj,
                )
            sup_last = sup_result["messages"][-1]
            sup_response = sanitize_text(normalize_content(sup_last.content))
            if job is not None:
                job["supervisor_response"] = sup_response
                job["wakeup_status"] = "COMPLETED"
        logger.info(f"🎉 [Job {job_id}] Reactive Wakeup ({outcome}) completed! Supervisor produced final response.")
    except Exception as cb_err:
        if job is not None:
            job["wakeup_status"] = "FAILED"
            job["wakeup_error"] = str(cb_err)
        logger.error(f"⚠️ [Job {job_id}] Error during reactive wakeup of '{cb_agent}': {cb_err}")
        traceback.print_exc()


async def _run_agent_job(
    job_id: str,
    agent_name: str,
    input_data: JobSubmitInput,
    context_obj: AgentContext,
    config: dict,
    app: FastAPI,
):
    job = job_store.get(job_id)
    if not job:
        return

    if input_data.callback_thread_id and not job.get("callback_thread_id"):
        job["callback_thread_id"] = input_data.callback_thread_id
    if input_data.parent_thread_id and not job.get("parent_thread_id"):
        job["parent_thread_id"] = input_data.parent_thread_id
    if getattr(context_obj, "user_id", None) and not job.get("user_id"):
        job["user_id"] = context_obj.user_id
    if getattr(context_obj, "workspace_dir", None) and not job.get("workspace_dir"):
        job["workspace_dir"] = context_obj.workspace_dir
    if not job.get("sub_thread_id") and input_data.thread_id:
        job["sub_thread_id"] = input_data.thread_id

    if job.get("cancel_requested"):
        job["status"] = "CANCELLED"
        job["completed_at"] = _utc_now_iso()
        job["wakeup_status"] = "SKIPPED"
        job.setdefault("cancel_reason", "Cancelled before execution started")
        await _cleanup_cancelled_subagent_resources(input_data.thread_id)
        return

    job["status"] = "RUNNING"
    job.setdefault("created_at_ts", time.time())
    job.setdefault("execution_started_at_ts", None)
    job.setdefault("wakeup_status", "PENDING" if input_data.callback_agent else "SKIPPED")
    logger.info(f"⚙️ [Job {job_id}] Started background execution for agent '{agent_name}'")

    try:
        agent_executor = await get_or_load_agent(agent_name, app)

        async def _execute_subagent() -> str:
            with plan_session_scope(
                input_data.thread_id or "default",
                user_id=context_obj.user_id,
                parent_session_id=context_obj.parent_session_id,
            ):
                result = await agent_executor.ainvoke(
                    {"messages": [("user", input_data.message)]},
                    config=config,
                    context=context_obj,
                )
            last_message = result["messages"][-1]
            return sanitize_text(normalize_content(last_message.content))

        async with acquire_thread_slot(input_data.thread_id):
            job["execution_started_at_ts"] = time.time()
            task_report = await _execute_subagent()

        if job.get("cancel_requested"):
            raise asyncio.CancelledError()

        job["status"] = "SUCCESS"
        job["completed_at"] = _utc_now_iso()
        job["result"] = task_report
        logger.info(f"✅ [Job {job_id}] Sub-agent '{agent_name}' completed successfully")

    except asyncio.CancelledError:
        logger.info(f"🛑 [Job {job_id}] Background execution cancelled for agent '{agent_name}'")
        job["status"] = "CANCELLED"
        job["completed_at"] = _utc_now_iso()
        job["wakeup_status"] = "SKIPPED"
        job.setdefault("cancel_reason", "Cancelled")
        await _cleanup_cancelled_subagent_resources(input_data.thread_id)
        raise

    except Exception as e:
        if job.get("cancel_requested"):
            logger.info(
                f"🛑 [Job {job_id}] Ignored teardown exception after cancel request ({type(e).__name__}: {e})"
            )
            job["status"] = "CANCELLED"
            job["completed_at"] = job.get("completed_at") or _utc_now_iso()
            job["wakeup_status"] = "SKIPPED"
            job.setdefault("cancel_reason", "Cancelled")
            await _cleanup_cancelled_subagent_resources(input_data.thread_id)
            return
        logger.error(f"❌ [Job {job_id}] Background execution failed: {e}")
        traceback.print_exc()
        err_text = str(e)
        job["status"] = "FAILED"
        job["completed_at"] = _utc_now_iso()
        job["error"] = err_text
        await _trigger_reactive_wakeup(
            job_id=job_id,
            agent_name=agent_name,
            input_data=input_data,
            context_obj=context_obj,
            app=app,
            outcome="FAILED",
            error_text=err_text,
        )
        return

    # 2. 🌟 Reactive Agent Wakeup: 서브에이전트 완료(SUCCESS) 후 Supervisor 자동 호출 & 후속 작업 수행
    await _trigger_reactive_wakeup(
        job_id=job_id,
        agent_name=agent_name,
        input_data=input_data,
        context_obj=context_obj,
        app=app,
        outcome="SUCCESS",
        task_report=task_report,
    )


# --- Job Management Endpoints ---
@app.post("/agents/{agent_name}/jobs")
async def submit_agent_job(
    agent_name: str,
    input_data: JobSubmitInput,
    request: Request,
):
    """비동기 백그라운드 작업을 등록하고 job_id를 즉시 반환합니다."""
    available = [a["name"] for a in api_list_agents() if isinstance(a, dict)]
    if agent_name not in available:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found. Available: {available}")

    effective_parent_tid = input_data.parent_thread_id or input_data.callback_thread_id
    resolved_tid = (input_data.thread_id or "").strip() or (
        f"{effective_parent_tid or 'bg'}_{agent_name}_{uuid.uuid4().hex[:8]}"
    )
    if input_data.thread_id != resolved_tid:
        input_data = input_data.model_copy(update={"thread_id": resolved_tid})

    if any(
        j.get("sub_thread_id") == resolved_tid and j.get("status") == "CANCELLED"
        for j in job_store.values()
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot reuse cancelled sub_thread_id '{resolved_tid}'. Start a fresh session instead.",
        )

    context_obj, config = await _resolve_run(
        agent_name,
        resolved_tid,
        user_id=input_data.user_id,
        parent_thread_id=effective_parent_tid,
        force_disable_hitl=True,
    )

    job_id = f"job_{uuid.uuid4().hex[:8]}"
    created_at_ts = time.time()
    created_at = _utc_now_iso()
    job_store[job_id] = {
        "job_id": job_id,
        "agent_name": agent_name,
        "sub_thread_id": resolved_tid,
        "parent_thread_id": effective_parent_tid,
        "user_id": context_obj.user_id,
        "workspace_dir": context_obj.workspace_dir,
        "status": "SUBMITTED",
        "created_at": created_at,
        "created_at_ts": created_at_ts,
        "execution_started_at_ts": None,
        "completed_at": None,
        "result": None,
        "error": None,
        "cancel_requested": False,
        "cancel_reason": None,
        "supervisor_response": None,
        "wakeup_status": "PENDING" if input_data.callback_agent else "SKIPPED",
        "wakeup_error": None,
        "callback_agent": input_data.callback_agent,
        "callback_thread_id": input_data.callback_thread_id,
    }

    task = asyncio.create_task(
        _run_agent_job(job_id, agent_name, input_data, context_obj, config, request.app),
        name=f"agent-job-{job_id}",
    )
    job_tasks[job_id] = task
    task.add_done_callback(lambda _t, jid=job_id: job_tasks.pop(jid, None))
    return {
        "job_id": job_id,
        "status": "SUBMITTED",
        "agent_name": agent_name,
        "sub_thread_id": resolved_tid,
    }


@app.post("/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, body: Optional[JobCancelInput] = None):
    """실행 중이거나 대기 중인 백그라운드 작업을 즉시 취소하고 리소스를 정리합니다."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    _verify_job_caller(job, body.caller_thread_id if body else None)

    if job.get("status") == "CANCELLED":
        return {
            "job_id": job_id,
            "agent_name": job.get("agent_name"),
            "sub_thread_id": job.get("sub_thread_id"),
            "status": "CANCELLED",
            "cancelled": True,
            "already_finished": True,
            "cancel_reason": job.get("cancel_reason"),
            "message": f"Job '{job_id}' was already cancelled.",
        }

    if job.get("status") not in ("SUBMITTED", "RUNNING"):
        return {
            "job_id": job_id,
            "agent_name": job.get("agent_name"),
            "sub_thread_id": job.get("sub_thread_id"),
            "status": job.get("status"),
            "cancelled": False,
            "already_finished": True,
            "message": f"Job '{job_id}' already finished with status '{job.get('status')}'.",
        }

    reason = (body.reason.strip() if (body and body.reason) else "") or "Cancelled by request"
    job["cancel_requested"] = True
    job["cancel_reason"] = reason

    task = job_tasks.get(job_id)
    if task is not None and not task.done():
        task.cancel()
        await asyncio.wait({task}, timeout=3.0)

    if job.get("status") in ("SUBMITTED", "RUNNING"):
        job["status"] = "CANCELLED"
        job["completed_at"] = _utc_now_iso()
        job["wakeup_status"] = "SKIPPED"
        await _cleanup_cancelled_subagent_resources(job.get("sub_thread_id"))

    return {
        "job_id": job_id,
        "agent_name": job.get("agent_name"),
        "sub_thread_id": job.get("sub_thread_id"),
        "status": job.get("status"),
        "cancelled": job.get("status") == "CANCELLED",
        "already_finished": False,
        "cancel_reason": job.get("cancel_reason"),
        "wakeup_status": job.get("wakeup_status"),
    }


@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str, caller_thread_id: Optional[str] = None):
    """특정 작업의 진행 상태와 결과(및 실시간 실행 텔레메트리/Supervisor 후속 보고서)를 조회합니다."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    _verify_job_caller(job, caller_thread_id)
    return _enrich_job_with_progress(job, include_terminal=True)


@app.get("/jobs")
async def list_jobs():
    """모든 백그라운드 작업 목록을 조회합니다."""
    return [_enrich_job_with_progress(job, include_terminal=False) for job in job_store.values()]


@app.get("/sessions/{thread_id}/jobs")
async def list_session_jobs(thread_id: str):
    """특정 세션(thread)에 연관된 모든 백그라운드 작업 목록과 실시간 텔레메트리를 반환합니다."""
    return [
        _enrich_job_with_progress(job, include_terminal=False)
        for job in job_store.values()
        if job.get("callback_thread_id") == thread_id or job.get("parent_thread_id") == thread_id
    ]


# --- Unified SSE Stream Generator (Shared by /stream and /resume) ---
async def _stream_agent_events(
    agent_name: str,
    agent_executor: Any,
    stream_input: Any,
    config: dict,
    context_obj: AgentContext,
    thread_id: Optional[str],
    user_message_to_record: Optional[str] = None,
    check_state_always: bool = False,
) -> AsyncGenerator[str, None]:
    """/stream 과 /resume 엔드포인트가 공통으로 사용하는 통합 SSE 스트리밍 제너레이터입니다.
    동일 thread_id에 대해서는 get_thread_lock(thread_id)를 획득하여 Double-Texting 및 Wakeup 경합을 방지합니다.
    (대화 메시지 영속화는 LangGraph Checkpointer SSOT가 전담합니다.)
    """
    lock = get_thread_lock(thread_id) if thread_id else None

    async def _produce() -> AsyncGenerator[str, None]:
        try:
            full_response = ""

            with plan_session_scope(
                thread_id or "default",
                user_id=context_obj.user_id,
                parent_session_id=context_obj.parent_session_id,
            ):
                async for event in agent_executor.astream_events(
                    stream_input,
                    config=config,
                    context=context_obj,
                    version="v2",
                ):
                    kind = event["event"]

                    if kind == "on_tool_start":
                        tool_input = sanitize_text(str(event["data"].get("input", "")))
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

            if (check_state_always or not full_response) and config:
                try:
                    state = await agent_executor.aget_state(config)
                    if state and state.tasks:
                        for task in state.tasks:
                            if task.interrupts:
                                interrupt_data = []
                                for intr in task.interrupts:
                                    value = intr.value
                                    if hasattr(value, "__dict__"):
                                        value = dict(value)
                                    interrupt_data.append({
                                        "value": value,
                                        "id": str(intr.id) if hasattr(intr, "id") else "",
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
                    logger.error(f"Error getting final state: {get_state_err}")

        except Exception as e:
            logger.error(f"SSE stream error in agent '{agent_name}': {e}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

        yield "event: end\ndata: \n\n"

    if lock is not None:
        key = thread_id or "__anon__"
        already_busy = lock.locked() or (thread_inflight_counts.get(key, 0) > 0)
        thread_inflight_counts[key] = thread_inflight_counts.get(key, 0) + 1
        try:
            if already_busy:
                yield f"data: {json.dumps({'type': 'queued', 'thread_id': thread_id})}\n\n"
            async with lock:
                async for chunk in _produce():
                    yield chunk
        finally:
            remaining = thread_inflight_counts.get(key, 1) - 1
            if remaining <= 0:
                thread_inflight_counts.pop(key, None)
            else:
                thread_inflight_counts[key] = remaining
    else:
        async for chunk in _produce():
            yield chunk


@app.post("/agents/{agent_name}/stream")
async def stream_agent(agent_name: str, input_data: StreamInput, request: Request):
    try:
        agent_executor = await get_or_load_agent(agent_name, request.app)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    context_obj, config = await _resolve_run(
        agent_name,
        input_data.thread_id,
        user_id=input_data.user_id,
        parent_thread_id=input_data.parent_thread_id,
    )

    return StreamingResponse(
        _stream_agent_events(
            agent_name=agent_name,
            agent_executor=agent_executor,
            stream_input={"messages": [("user", input_data.message)]},
            config=config,
            context_obj=context_obj,
            thread_id=input_data.thread_id,
            user_message_to_record=input_data.message,
            check_state_always=False,
        ),
        media_type="text/event-stream",
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

    context_obj, config = await _resolve_run(
        agent_name, input_data.thread_id, user_id=input_data.user_id
    )
    resume_command = Command(resume={"decisions": input_data.decisions})

    return StreamingResponse(
        _stream_agent_events(
            agent_name=agent_name,
            agent_executor=agent_executor,
            stream_input=resume_command,
            config=config,
            context_obj=context_obj,
            thread_id=input_data.thread_id,
            user_message_to_record=None,
            check_state_always=True,
        ),
        media_type="text/event-stream",
    )


# --- Chat Session API Endpoints ---
@app.post("/sessions")
async def api_create_session(session: SessionCreate):
    try:
        return await create_session(
            session.session_id,
            session.agent_name,
            session.title,
            user_id=session.user_id or "default",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions")
async def api_get_sessions(agent_name: Optional[str] = None, user_id: Optional[str] = None):
    try:
        return await get_sessions(agent_name, user_id=user_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/sessions/{session_id}")
async def api_delete_session(session_id: str):
    try:
        await delete_session(session_id)
        return {"status": "success", "deleted_session": session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sessions/{session_id}/messages")
async def api_get_messages(session_id: str):
    try:
        return await get_messages(session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
