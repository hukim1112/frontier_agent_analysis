"""
===============================================================================
[AAWS Tool Module] Planning & Task Management Tool Engine (plan.py)
===============================================================================
Supervisor 및 오케스트레이션 에이전트를 위한 5대 Planning & Task Board 도구 엔진:
1. enter_plan: 거시적 실행 계획 수립 및 Planning 모드 진입
2. exit_plan: Planning 모드 종료 및 상태 완료 처리
3. task_create: Task Board에 세부 하위 과제 등록 (ID 자동 발급)
4. task_list: 현재 Task Board의 전체 작업 목록 및 상태 조회
5. task_update: 특정 과제 상태 갱신 (PENDING, IN_PROGRESS, COMPLETED, BLOCKED)
===============================================================================
"""

import os
import re
import time
import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.utils.workspace import resolve_session_workspace

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PLANS_BASE_DIR = PROJECT_ROOT / "artifacts" / "plans"
PLANS_BASE_DIR = _DEFAULT_PLANS_BASE_DIR

_DEFAULT_PLAN_SESSION = "default"
_DEFAULT_PLAN_USER = "default"
_plan_session_var: ContextVar[Optional[str]] = ContextVar("_plan_session_var", default=None)
_plan_user_var: ContextVar[Optional[str]] = ContextVar("_plan_user_var", default=None)
_plan_parent_session_var: ContextVar[Optional[str]] = ContextVar("_plan_parent_session_var", default=None)
_plan_lock = threading.RLock()


def _sanitize_session_id(session_id: str) -> str:
    """파일시스템 경로로 안전하게 사용할 수 있도록 session_id를 정제합니다."""
    cleaned = re.sub(r"[^a-zA-Z0-9._\-]", "_", str(session_id).strip())
    if not cleaned or cleaned in {".", ".."} or set(cleaned) == {"."}:
        return _DEFAULT_PLAN_SESSION
    return cleaned


def _current_plan_session_id(session_id: Optional[str] = None) -> str:
    """현재 실행 중인 에이전트 세션(thread_id)을 반환합니다."""
    if session_id and str(session_id).strip():
        return _sanitize_session_id(session_id)
    ctx_sid = _plan_session_var.get()
    if ctx_sid:
        return _sanitize_session_id(ctx_sid)
    try:
        cfg = var_child_runnable_config.get()
        if isinstance(cfg, dict):
            tid = (cfg.get("configurable") or {}).get("thread_id")
            if tid and str(tid).strip():
                return _sanitize_session_id(str(tid))
    except Exception:
        pass
    try:
        from langchain_core.runnables.config import get_config_from_context
        cfg = get_config_from_context()
        if isinstance(cfg, dict):
            tid = (cfg.get("configurable") or {}).get("thread_id")
            if tid and str(tid).strip():
                return _sanitize_session_id(str(tid))
    except Exception:
        pass
    return _DEFAULT_PLAN_SESSION


def _current_plan_user_id(user_id: Optional[str] = None) -> str:
    """현재 실행 중인 사용자(user_id)를 반환합니다."""
    if user_id and str(user_id).strip():
        return _sanitize_session_id(user_id)
    ctx_uid = _plan_user_var.get()
    if ctx_uid:
        return _sanitize_session_id(ctx_uid)
    try:
        cfg = var_child_runnable_config.get()
        if isinstance(cfg, dict):
            uid = (cfg.get("configurable") or {}).get("user_id")
            if uid and str(uid).strip():
                return _sanitize_session_id(str(uid))
    except Exception:
        pass
    try:
        from langchain_core.runnables.config import get_config_from_context
        cfg = get_config_from_context()
        if isinstance(cfg, dict):
            uid = (cfg.get("configurable") or {}).get("user_id")
            if uid and str(uid).strip():
                return _sanitize_session_id(str(uid))
    except Exception:
        pass
    return _DEFAULT_PLAN_USER


def _current_plan_parent_session_id(parent_session_id: Optional[str] = None) -> Optional[str]:
    """현재 서브에이전트 실행의 상위 세션(parent_thread_id)이 명시적으로 전달된 경우 반환합니다."""
    if parent_session_id and str(parent_session_id).strip():
        return _sanitize_session_id(parent_session_id)
    ctx_psid = _plan_parent_session_var.get()
    if ctx_psid:
        return _sanitize_session_id(ctx_psid)
    try:
        cfg = var_child_runnable_config.get()
        if isinstance(cfg, dict):
            ptid = (cfg.get("configurable") or {}).get("parent_thread_id")
            if ptid and str(ptid).strip():
                return _sanitize_session_id(str(ptid))
    except Exception:
        pass
    try:
        from langchain_core.runnables.config import get_config_from_context
        cfg = get_config_from_context()
        if isinstance(cfg, dict):
            ptid = (cfg.get("configurable") or {}).get("parent_thread_id")
            if ptid and str(ptid).strip():
                return _sanitize_session_id(str(ptid))
    except Exception:
        pass
    return None


@contextmanager
def plan_session_scope(
    session_id: str,
    user_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
):
    """명시적으로 Planning/TaskBoard 세션(thread_id), 사용자(user_id), 상위 세션(parent_session_id)을 지정하는 컨텍스트 매니저."""
    tok_sid = _plan_session_var.set(_current_plan_session_id(session_id))
    tok_uid = _plan_user_var.set(_current_plan_user_id(user_id))
    tok_psid = _plan_parent_session_var.set(_current_plan_parent_session_id(parent_session_id))
    try:
        yield
    finally:
        _plan_session_var.reset(tok_sid)
        _plan_user_var.reset(tok_uid)
        _plan_parent_session_var.reset(tok_psid)


def _session_plan_dir(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
) -> Path:
    sid = _current_plan_session_id(session_id)
    uid = _current_plan_user_id(user_id)
    psid = _current_plan_parent_session_id(parent_session_id)
    # 단위 테스트에서 PLANS_BASE_DIR를 임시 폴더로 직접 패치한 경우 하위 호환 유지
    if PLANS_BASE_DIR != _DEFAULT_PLANS_BASE_DIR:
        d = PLANS_BASE_DIR / sid
        d.mkdir(parents=True, exist_ok=True)
        return d
    ws = resolve_session_workspace(user_id=uid, session_id=sid, parent_session_id=psid)
    ws.plan_dir.mkdir(parents=True, exist_ok=True)
    return ws.plan_dir


# =============================================================================
# Category 1: Planning Tools (enter_plan, exit_plan) — 세션별 격리
# =============================================================================
# 하위 호환용 최근 상태 스냅샷 (실제 진실 공급원은 세션별 plan_state.json)
plan_session_state: Dict[str, Any] = {"active": False, "plan_name": "", "steps": []}


def get_plan_state(session_id: Optional[str] = None) -> Dict[str, Any]:
    """특정 세션(또는 현재 세션)의 계획 상태를 디스크에서 읽어 반환합니다."""
    plan_file = _session_plan_dir(session_id) / "plan_state.json"
    with _plan_lock:
        if plan_file.exists():
            try:
                return json.loads(plan_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"active": False, "plan_name": "", "steps": []}


def _save_plan_state(state: Dict[str, Any], session_id: Optional[str] = None) -> None:
    global plan_session_state
    plan_file = _session_plan_dir(session_id) / "plan_state.json"
    with _plan_lock:
        plan_session_state = dict(state)
        try:
            plan_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def clear_plan_state(session_id: Optional[str] = None) -> None:
    """현재 세션(또는 지정 세션)의 plan_state.json 및 메모리 스냅샷을 초기화합니다."""
    global plan_session_state
    plan_file = _session_plan_dir(session_id) / "plan_state.json"
    with _plan_lock:
        plan_session_state = {"active": False, "plan_name": "", "steps": []}
        try:
            if plan_file.exists():
                plan_file.unlink()
        except Exception:
            pass


class EnterPlanInput(BaseModel):
    plan_name: str = Field(description="Title or concise name of the execution plan.")
    steps: List[str] = Field(description="Sequential step-by-step list of plan actions.")


@tool(args_schema=EnterPlanInput)
def enter_plan(plan_name: str, steps: List[str]) -> str:
    """Enters Planning Mode and registers a step-by-step execution plan.

    Args:
        plan_name: Title or concise name of the plan.
        steps: List of string steps detailing the sequence of actions.

    Returns:
        Status message confirming planning mode entry and step count.
    """
    sid = _current_plan_session_id()
    state = {"active": True, "plan_name": plan_name, "steps": steps, "session_id": sid}
    _save_plan_state(state, session_id=sid)
    return f"[PLANNING MODE ENTERED] Registered '{plan_name}' with {len(steps)} sequential steps."


class ExitPlanInput(BaseModel):
    plan_name: str = Field(description="Title or concise name of the plan to exit.")
    status: str = Field(default="COMPLETED", description="Final status of the plan (e.g. COMPLETED, ABORTED). Defaults to COMPLETED.")


@tool(args_schema=ExitPlanInput)
def exit_plan(plan_name: str, status: str = "COMPLETED") -> str:
    """Exits Planning Mode and resumes standard execution mode.

    Args:
        plan_name: Title or name of the plan to conclude.
        status: Plan outcome status string. Defaults to 'COMPLETED'.

    Returns:
        Status message confirming plan termination.
    """
    sid = _current_plan_session_id()
    state = get_plan_state(session_id=sid)
    state["active"] = False
    state["plan_name"] = plan_name or state.get("plan_name", "")
    state["status"] = status
    state["session_id"] = sid
    _save_plan_state(state, session_id=sid)
    return f"[PLANNING MODE EXITED] Plan '{plan_name}' terminated with status '{status}'."


# =============================================================================
# Category 2: Task Board Tools (task_create, task_list, task_update) — 세션별 격리
# =============================================================================
class _TaskListProxy(list):
    """list 하위 호환 프록시 — task_board.tasks.clear() 호출 시 디스크의 세션 상태까지 함께 초기화합니다."""
    def __init__(self, items: List[Dict[str, Any]], board: "TaskBoardState", session_id: Optional[str] = None):
        super().__init__(items)
        self._board = board
        self._session_id = session_id

    def clear(self) -> None:
        super().clear()
        self._board.clear(session_id=self._session_id)


class TaskBoardState:
    """Manages persistent task items on the agent task board, isolated per session (thread_id)."""

    def __init__(self, filename: Optional[str] = None):
        # filename이 명시되면 고정 경로 사용(단위 테스트 호환), None이면 현재 세션별 경로 동적 해석
        self._explicit_filename = filename
        self._lock = threading.RLock()

    def _resolve_file(self, session_id: Optional[str] = None) -> Path:
        if self._explicit_filename:
            return Path(self._explicit_filename)
        return _session_plan_dir(session_id) / "task_state.json"

    def _load(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        fpath = self._resolve_file(session_id)
        if fpath.exists():
            try:
                return json.loads(fpath.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save(self, tasks: List[Dict[str, Any]], session_id: Optional[str] = None) -> None:
        fpath = self._resolve_file(session_id)
        try:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    def clear(self, session_id: Optional[str] = None) -> None:
        """현재 세션(또는 지정 세션)의 task_state.json 및 plan_state.json을 초기화합니다."""
        with self._lock:
            fpath = self._resolve_file(session_id)
            try:
                if fpath.exists():
                    fpath.unlink()
            except Exception:
                pass
            if not self._explicit_filename:
                clear_plan_state(session_id=session_id)

    @property
    def tasks(self) -> List[Dict[str, Any]]:
        with self._lock:
            return _TaskListProxy(self._load(), self)

    def create_task(self, description: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            tasks = self._load(session_id)
            t_id = len(tasks) + 1
            t_item = {
                "id": t_id,
                "description": description,
                "status": "PENDING",
                "created_at": time.time(),
            }
            tasks.append(t_item)
            self._save(tasks, session_id)
            return t_item

    def list_tasks(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        with self._lock:
            return self._load(session_id)

    def update_task(self, task_id: int, status: str, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with self._lock:
            tasks = self._load(session_id)
            for t in tasks:
                if t["id"] == task_id:
                    t["status"] = status
                    self._save(tasks, session_id)
                    return t
            return None


task_board = TaskBoardState()


class TaskCreateInput(BaseModel):
    task_description: str = Field(description="Clear description of the sub-task item to track.")


@tool(args_schema=TaskCreateInput)
def task_create(task_description: str) -> str:
    """Creates a new tracked sub-task item on the task board.

    Args:
        task_description: Clear description of the task item.

    Returns:
        Status message with generated task ID and initial PENDING status.
    """
    t = task_board.create_task(task_description)
    return f"[TASK CREATED] Task #{t['id']}: '{task_description}' (Status: PENDING)"


@tool
def task_list() -> str:
    """Lists all active and completed tasks on the task board.

    Returns:
        JSON string listing all tracked tasks with their IDs, descriptions, and statuses.
    """
    tasks = task_board.list_tasks()
    if not tasks:
        return "No tasks currently registered on task board."
    return json.dumps(tasks, indent=2, ensure_ascii=False)


class TaskUpdateInput(BaseModel):
    task_id: int = Field(description="Numeric ID of the target task to update.")
    status: str = Field(description="New status string: 'PENDING', 'IN_PROGRESS', 'COMPLETED', or 'BLOCKED'.")


@tool(args_schema=TaskUpdateInput)
def task_update(task_id: int, status: str) -> str:
    """Updates the status of a specific task on the task board.

    Args:
        task_id: Numeric ID of the task to modify.
        status: New status string ('PENDING', 'IN_PROGRESS', 'COMPLETED', 'BLOCKED').

    Returns:
        Confirmation message of status update or error if ID not found.
    """
    t = task_board.update_task(task_id, status)
    if not t:
        return f"Task #{task_id} not found."
    return f"[TASK UPDATED] Task #{task_id} status changed to '{status}'."


# 📋 5대 Planning & Task 도구 묶음
tools_planning = [enter_plan, exit_plan, task_create, task_list, task_update]
PLANNING_AND_TASK_TOOLS = tools_planning
