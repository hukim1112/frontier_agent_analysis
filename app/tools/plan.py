"""
===============================================================================
[AAWS Tool Module] Planning & Task Management Tool Engine (plan.py)
===============================================================================
Claude Code Plan Mode 기반의 5대 Planning & Task Board 도구 엔진 (Blackboard SSOT):
1. enter_plan: 읽기 전용 Planning Mode 진입 (사전 조사 또는 [BLOCKER] 백트래킹 재설계)
2. exit_plan: 확정된 실행 계획을 plan_state.json에 게시하고 Execution Mode 잠금 해제
3. task_create: Task Board에 세부 하위 과제 등록 (ID 자동 발급, Planning Mode 중 차단)
4. task_list: 현재 활성 계획 요약(plan_state.json)과 태스크 보드(task_state.json) 현황 조회
5. task_update: 특정 과제 상태 갱신 (PENDING, IN_PROGRESS, COMPLETED, BLOCKED)
===============================================================================
"""

import json
import inspect
import logging
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

from langchain_core.runnables.config import ensure_config, var_child_runnable_config
from langchain_core.tools import ToolException, tool
from pydantic import BaseModel, Field, field_validator

from app.utils.workspace import resolve_session_workspace, sanitize_path_segment

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PLANS_BASE_DIR = PROJECT_ROOT / "artifacts" / "plans"
PLANS_BASE_DIR = _DEFAULT_PLANS_BASE_DIR

_DEFAULT_PLAN_SESSION = "default"
_DEFAULT_PLAN_USER = "default"
_plan_session_var: ContextVar[Optional[str]] = ContextVar("_plan_session_var", default=None)
_plan_user_var: ContextVar[Optional[str]] = ContextVar("_plan_user_var", default=None)
_plan_parent_session_var: ContextVar[Optional[str]] = ContextVar("_plan_parent_session_var", default=None)
_plan_lock = threading.RLock()

VALID_TASK_STATUSES = ("PENDING", "IN_PROGRESS", "COMPLETED", "BLOCKED")
TaskStatus = Literal["PENDING", "IN_PROGRESS", "COMPLETED", "BLOCKED"]

PLANNING_ALLOWED_TOOLS = frozenset({
    "enter_plan",
    "exit_plan",
    "task_list",
    "task_update",
    "list_sub_agents",
    "get_sub_agent_job_status",
    "file_read",
    "grep_search",
    "glob_search",
    "web_search",
    "web_fetch",
})


def _sanitize_session_id(session_id: str) -> str:
    """파일시스템 경로로 안전하게 사용할 수 있도록 session_id를 정제합니다."""
    return sanitize_path_segment(session_id, fallback=_DEFAULT_PLAN_SESSION)


def _get_runnable_configurable() -> Dict[str, Any]:
    """현재 LangChain Runnable 컨텍스트의 configurable 딕셔너리를 반환합니다."""
    try:
        cfg = var_child_runnable_config.get()
        if isinstance(cfg, dict) and isinstance(cfg.get("configurable"), dict):
            return cfg["configurable"]
    except Exception:
        pass
    try:
        cfg = ensure_config()
        if isinstance(cfg, dict) and isinstance(cfg.get("configurable"), dict):
            return cfg["configurable"]
    except Exception:
        pass
    return {}


def _current_plan_session_id(session_id: Optional[str] = None) -> str:
    """현재 실행 중인 에이전트 세션(thread_id)을 반환합니다."""
    if session_id and str(session_id).strip():
        return _sanitize_session_id(session_id)
    ctx_sid = _plan_session_var.get()
    if ctx_sid:
        return _sanitize_session_id(ctx_sid)
    tid = _get_runnable_configurable().get("thread_id")
    if tid and str(tid).strip():
        return _sanitize_session_id(str(tid))
    return _DEFAULT_PLAN_SESSION


def _current_plan_user_id(user_id: Optional[str] = None) -> str:
    """현재 실행 중인 사용자(user_id)를 반환합니다."""
    if user_id and str(user_id).strip():
        return sanitize_path_segment(user_id, fallback=_DEFAULT_PLAN_USER)
    ctx_uid = _plan_user_var.get()
    if ctx_uid:
        return sanitize_path_segment(ctx_uid, fallback=_DEFAULT_PLAN_USER)
    uid = _get_runnable_configurable().get("user_id")
    if uid and str(uid).strip():
        return sanitize_path_segment(str(uid), fallback=_DEFAULT_PLAN_USER)
    return _DEFAULT_PLAN_USER


def _current_plan_parent_session_id(parent_session_id: Optional[str] = None) -> Optional[str]:
    """현재 서브에이전트 실행의 상위 세션(parent_thread_id)이 명시적으로 전달된 경우 반환합니다."""
    if parent_session_id and str(parent_session_id).strip():
        return sanitize_path_segment(parent_session_id, fallback=_DEFAULT_PLAN_SESSION)
    ctx_psid = _plan_parent_session_var.get()
    if ctx_psid is not None:
        return ctx_psid if ctx_psid != "" else None
    ptid = _get_runnable_configurable().get("parent_thread_id")
    if ptid and str(ptid).strip():
        return sanitize_path_segment(str(ptid), fallback=_DEFAULT_PLAN_SESSION)
    return None


@contextmanager
def plan_session_scope(
    session_id: str,
    user_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
):
    """명시적으로 Planning/TaskBoard 세션(thread_id), 사용자(user_id), 상위 세션(parent_session_id)을 지정하는 컨텍스트 매니저."""
    tok_sid = _plan_session_var.set(sanitize_path_segment(session_id, fallback=_DEFAULT_PLAN_SESSION))
    tok_uid = _plan_user_var.set(sanitize_path_segment(user_id, fallback=_DEFAULT_PLAN_USER))
    if parent_session_id and str(parent_session_id).strip():
        psid_val = sanitize_path_segment(parent_session_id, fallback=_DEFAULT_PLAN_SESSION)
    else:
        # 빈 문자열 센티넬: 이 스코프에서는 명시적으로 상위 세션이 없음을 뜻함 (외부 RunnableConfig 폴백 차단)
        psid_val = ""
    tok_psid = _plan_parent_session_var.set(psid_val)
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
    ensure_exists: bool = False,
) -> Path:
    sid = _current_plan_session_id(session_id)
    uid = _current_plan_user_id(user_id)
    psid = _current_plan_parent_session_id(parent_session_id)
    # 단위 테스트에서 PLANS_BASE_DIR를 임시 폴더로 직접 패치한 경우 하위 호환 유지
    if PLANS_BASE_DIR != _DEFAULT_PLANS_BASE_DIR:
        d = PLANS_BASE_DIR / sid
        if ensure_exists:
            d.mkdir(parents=True, exist_ok=True)
        return d
    ws = resolve_session_workspace(
        user_id=uid,
        session_id=sid,
        parent_session_id=psid,
        ensure_exists=False,
    )
    if ensure_exists:
        ws.plan_dir.mkdir(parents=True, exist_ok=True)
    return ws.plan_dir


def _rel_path_str(path: Path) -> str:
    """프로젝트 루트 기준 상대 POSIX 경로를 반환합니다 (외부 경로인 경우 POSIX 전체 경로 반환)."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


# =============================================================================
# Category 1: Planning Tools (enter_plan, exit_plan) — Claude Code Plan Mode
# =============================================================================
def _default_plan_state() -> Dict[str, Any]:
    return {
        "active": False,
        "phase": "IDLE",
        "status": "IDLE",
        "plan_name": "",
        "goal": "",
        "steps": [],
    }


# 하위 호환용 최근 상태 스냅샷 (실제 진실 공급원은 세션별 plan_state.json)
plan_session_state: Dict[str, Any] = _default_plan_state()


def get_plan_state(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """특정 세션(또는 현재 세션)의 계획 상태를 디스크에서 읽어 반환합니다."""
    plan_file = _session_plan_dir(session_id, user_id, parent_session_id, ensure_exists=False) / "plan_state.json"
    with _plan_lock:
        if plan_file.exists():
            try:
                data = json.loads(plan_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception as err:
                logger.warning("Failed to read plan_state.json at '%s': %s", plan_file, err)
        return _default_plan_state()


def _save_plan_state(
    state: Dict[str, Any],
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
) -> Path:
    global plan_session_state
    plan_file = _session_plan_dir(session_id, user_id, parent_session_id, ensure_exists=True) / "plan_state.json"
    with _plan_lock:
        plan_session_state = dict(state)
        try:
            plan_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as err:
            logger.warning("Failed to write plan_state.json at '%s': %s", plan_file, err)
    return plan_file


def get_plan_history(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """이전 계획 및 완료/중단된 태스크 아카이브(plan_history.json)를 읽어 반환합니다."""
    history_file = _session_plan_dir(session_id, user_id, parent_session_id, ensure_exists=False) / "plan_history.json"
    with _plan_lock:
        if history_file.exists():
            try:
                data = json.loads(history_file.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except Exception as err:
                logger.warning("Failed to read plan_history.json at '%s': %s", history_file, err)
        return []


def _append_plan_history(
    entry: Dict[str, Any],
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
) -> Path:
    history_file = _session_plan_dir(session_id, user_id, parent_session_id, ensure_exists=True) / "plan_history.json"
    with _plan_lock:
        history = get_plan_history(session_id, user_id, parent_session_id)
        history.append(entry)
        try:
            history_file.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as err:
            logger.warning("Failed to write plan_history.json at '%s': %s", history_file, err)
    return history_file


def clear_plan_state(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
    clear_history: bool = False,
) -> None:
    """현재 세션(또는 지정 세션)의 plan_state.json 및 메모리 스냅샷을 초기화합니다."""
    global plan_session_state
    plan_dir = _session_plan_dir(session_id, user_id, parent_session_id, ensure_exists=False)
    plan_file = plan_dir / "plan_state.json"
    history_file = plan_dir / "plan_history.json"
    with _plan_lock:
        plan_session_state = _default_plan_state()
        for target in (plan_file, history_file) if clear_history else (plan_file,):
            try:
                if target.exists():
                    target.unlink()
            except Exception as err:
                logger.warning("Failed to remove plan file '%s': %s", target, err)


def _has_unfinished_or_blocked_work(tasks: List[Dict[str, Any]]) -> bool:
    """현재 태스크 목록이 진행 중이거나 마지막 태스크가 BLOCKED 상태(백트래킹 대상)인지 판별합니다."""
    if not tasks:
        return False
    if any(t.get("status") in {"PENDING", "IN_PROGRESS"} for t in tasks):
        return True
    return tasks[-1].get("status") == "BLOCKED"


def _is_plan_unfinished(plan_state: Dict[str, Any], tasks: List[Dict[str, Any]]) -> bool:
    """이전 계획이 완료되지 않은 채 진행 중이거나 중단된 상태인지 판별합니다."""
    phase = plan_state.get("phase", "IDLE")
    if phase == "PLANNING":
        return True
    if tasks:
        return _has_unfinished_or_blocked_work(tasks)
    return phase == "EXECUTING"


def is_planning_mode_active(
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
) -> bool:
    """현재 세션이 읽기 전용 Planning Mode(exit_plan 호출 전)인지 여부를 반환합니다."""
    return bool(get_plan_state(session_id, user_id, parent_session_id).get("active", False))


def check_plan_mode_tool_allowed(
    tool_name: str,
    session_id: Optional[str] = None,
    user_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
    allowed_tools: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Planning Mode 활성 중 비허용 도구 호출을 차단하고 안내 메시지를 반환합니다 (허용 시 None)."""
    permitted = frozenset(allowed_tools) if allowed_tools is not None else PLANNING_ALLOWED_TOOLS
    if tool_name not in permitted and is_planning_mode_active(session_id, user_id, parent_session_id):
        return (
            f"[PLANNING MODE GUARDRAIL] Tool '{tool_name}' is blocked while in Planning Mode "
            "(read-only discovery phase). Finish your inspection and call 'exit_plan(plan_name, steps)' "
            "first to publish your finalized plan to the Blackboard and unlock task creation and sub-agent delegation."
        )
    return None


class EnterPlanInput(BaseModel):
    intent: Literal["new", "revise"] = Field(
        description=(
            "'new': start a plan for a new request. Any previous plan and its tasks (finished, unfinished, "
            "or BLOCKED) are archived to plan_history.json and the task board is reset. "
            "'revise': backtrack and redesign the CURRENT plan (e.g. after a [BLOCKER]). "
            "Existing tasks, including BLOCKED ones, are preserved so failure context stays visible."
        ),
    )
    goal: str = Field(
        default="",
        description="Concise objective or problem statement to investigate before finalizing the plan.",
    )
    plan_name: Optional[str] = Field(
        default=None,
        description="Optional draft plan title.",
    )


@tool(args_schema=EnterPlanInput)
def enter_plan(
    intent: Literal["new", "revise"],
    goal: str = "",
    plan_name: Optional[str] = None,
) -> str:
    """Enters read-only Planning Mode and returns the Shared Plan File path with next-step guidance.

    While in Planning Mode (active=True, phase='PLANNING'), perform read-only discovery only
    (e.g. list_sub_agents, glob_search, grep_search, file_read, web_search, task_list) and do not
    modify files. Task creation (task_create) and sub-agent delegation (invoke_sub_agent) are
    blocked until exit_plan(plan_name, steps) is called.

    Choose intent explicitly:
      - intent='new'    : the user made a new request (even if the previous plan was left unfinished/BLOCKED).
      - intent='revise' : redesign the current plan mid-execution (backtracking after a [BLOCKER]).
    """
    sid = _current_plan_session_id()
    with _plan_lock:
        prev_plan = get_plan_state()
        prev_tasks = task_board.list_tasks()
        prev_phase = prev_plan.get("phase", "IDLE")
        has_prior_content = bool(
            prev_plan.get("steps") or prev_plan.get("plan_name") or prev_plan.get("goal") or prev_tasks
        )

        if intent == "revise":
            if prev_phase == "IDLE" and not prev_tasks:
                raise ToolException(
                    "[ERROR] There is no current plan to revise. "
                    "Call enter_plan(intent='new', goal=...) to start a new plan."
                )
            if not _is_plan_unfinished(prev_plan, prev_tasks):
                raise ToolException(
                    "[ERROR] The previous plan is already completed. "
                    "Call enter_plan(intent='new', goal=...) to start a new plan "
                    "(the completed plan will be archived automatically)."
                )
            if prev_phase != "PLANNING":
                _append_plan_history({
                    "archived_at": time.time(),
                    "reason": "backtrack_replan",
                    "plan": prev_plan,
                    "tasks": list(prev_tasks),
                })
            board_note = f"Revising current plan; preserved {len(prev_tasks)} existing task(s) on the board."
            state = {
                "active": True,
                "phase": "PLANNING",
                "status": "PLANNING",
                "plan_name": (plan_name or prev_plan.get("plan_name") or "").strip(),
                "goal": goal.strip() or prev_plan.get("goal", ""),
                "steps": list(prev_plan.get("steps", [])),
                "session_id": sid,
                "updated_at": time.time(),
            }
        else:
            if has_prior_content:
                reason = "abandoned" if _is_plan_unfinished(prev_plan, prev_tasks) else "new_plan"
                _append_plan_history({
                    "archived_at": time.time(),
                    "reason": reason,
                    "plan": prev_plan,
                    "tasks": prev_tasks,
                })
                board_note = f"Previous plan archived to plan_history.json (reason: {reason}); active task board reset."
            else:
                board_note = "Active task board initialized."
            task_board.clear_tasks_only()
            state = {
                "active": True,
                "phase": "PLANNING",
                "status": "PLANNING",
                "plan_name": (plan_name or "").strip(),
                "goal": goal.strip(),
                "steps": [],
                "session_id": sid,
                "updated_at": time.time(),
            }
        plan_file = _save_plan_state(state)

    rel_plan_file = _rel_path_str(plan_file)
    return (
        f"[PLANNING MODE ENTERED] Session '{sid}' is now in read-only discovery mode.\n"
        f"- Shared Plan File: {rel_plan_file}\n"
        f"- Task Board: {board_note}\n"
        f"- Blocked until exit_plan: task_create, invoke_sub_agent.\n"
        f"- Next: Perform any needed read-only discovery, then call exit_plan(plan_name, steps) to publish the finalized plan and unlock execution."
    )


class ExitPlanInput(BaseModel):
    plan_name: str = Field(description="Title or concise name of the finalized execution plan.")
    steps: List[str] = Field(
        description="Finalized sequential step-by-step action list based on discovery findings."
    )


@tool(args_schema=ExitPlanInput)
def exit_plan(plan_name: str, steps: List[str]) -> str:
    """Exits Planning Mode, publishes the finalized plan to the Shared Blackboard, and unlocks Execution Mode.

    Can only be called while in Planning Mode: call enter_plan(intent=...) first.
    Once called, task creation (task_create) and sub-agent delegation (invoke_sub_agent) are unlocked,
    returning the Shared Plan File and Shared Task File paths to share via target_file_list.
    """
    sid = _current_plan_session_id()
    clean_steps = [str(s).strip() for s in (steps or []) if str(s).strip()]
    with _plan_lock:
        prev_state = get_plan_state()
        if prev_state.get("phase") != "PLANNING":
            raise ToolException(
                "[ERROR] Not in Planning Mode. Call enter_plan(intent='new' or 'revise', goal=...) first, "
                "inspect the context, then call exit_plan(plan_name, steps)."
            )
        if not clean_steps:
            raise ToolException(
                "[ERROR] Cannot exit Planning Mode with an empty 'steps' list. "
                "Provide at least one actionable step in exit_plan(plan_name, steps)."
            )
        resolved_name = (plan_name or "").strip() or prev_state.get("plan_name", "") or "Execution Plan"
        state = {
            "active": False,
            "phase": "EXECUTING",
            "status": "EXECUTING",
            "plan_name": resolved_name,
            "goal": prev_state.get("goal", ""),
            "steps": clean_steps,
            "session_id": sid,
            "updated_at": time.time(),
        }
        plan_file = _save_plan_state(state)

    rel_plan_file = _rel_path_str(plan_file)
    rel_task_file = _rel_path_str(task_board._resolve_file())
    return (
        f"[PLANNING MODE EXITED] Finalized plan '{resolved_name}' with {len(clean_steps)} steps "
        f"(Execution Mode unlocked).\n"
        f"- Shared Plan File: {rel_plan_file}\n"
        f"- Shared Task File: {rel_task_file}\n"
        f"- Next: Create tracked tasks with task_create() and include '{rel_plan_file}' "
        f"in target_file_list when delegating to sub-agents."
    )


# =============================================================================
# Category 2: Task Board Tools (task_create, task_list, task_update) — 세션별 격리
# =============================================================================
def _normalize_task_status(status: str) -> str:
    norm = str(status).strip().upper()
    if norm == "DONE":
        norm = "COMPLETED"
    if norm not in VALID_TASK_STATUSES:
        raise ValueError(
            f"Invalid task status '{status}'. Must be one of: {', '.join(VALID_TASK_STATUSES)}."
        )
    return norm


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
        # plan_state와 task_state를 함께 갱신하는 경로가 있으므로 모듈 공용 잠금 하나만 사용 (잠금 순서 역전/교착 방지)
        self._lock = _plan_lock

    def _resolve_file(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        ensure_exists: bool = False,
    ) -> Path:
        if self._explicit_filename:
            fpath = Path(self._explicit_filename)
            if ensure_exists:
                fpath.parent.mkdir(parents=True, exist_ok=True)
            return fpath
        return _session_plan_dir(
            session_id=session_id,
            user_id=user_id,
            parent_session_id=parent_session_id,
            ensure_exists=ensure_exists,
        ) / "task_state.json"

    def _load(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        fpath = self._resolve_file(session_id, user_id, parent_session_id, ensure_exists=False)
        if fpath.exists():
            try:
                data = json.loads(fpath.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except Exception as err:
                logger.warning("Failed to read task_state.json at '%s': %s", fpath, err)
                return []
        return []

    def _save(
        self,
        tasks: List[Dict[str, Any]],
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
    ) -> None:
        fpath = self._resolve_file(session_id, user_id, parent_session_id, ensure_exists=True)
        try:
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.write_text(json.dumps(tasks, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as err:
            logger.warning("Failed to write task_state.json at '%s': %s", fpath, err)

    def clear_tasks_only(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
    ) -> None:
        """현재 세션(또는 지정 세션)의 task_state.json만 초기화합니다 (plan_state/plan_history 보존)."""
        with self._lock:
            fpath = self._resolve_file(session_id, user_id, parent_session_id, ensure_exists=False)
            try:
                if fpath.exists():
                    fpath.unlink()
            except Exception as err:
                logger.warning("Failed to remove task_state.json at '%s': %s", fpath, err)

    def clear(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        clear_history: bool = False,
    ) -> None:
        """현재 세션(또는 지정 세션)의 task_state.json 및 plan_state.json을 초기화합니다."""
        with self._lock:
            self.clear_tasks_only(session_id, user_id, parent_session_id)
            if not self._explicit_filename:
                clear_plan_state(
                    session_id=session_id,
                    user_id=user_id,
                    parent_session_id=parent_session_id,
                    clear_history=clear_history,
                )

    @property
    def tasks(self) -> List[Dict[str, Any]]:
        with self._lock:
            return _TaskListProxy(self._load(), self)

    def create_task(
        self,
        description: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            tasks = self._load(session_id, user_id, parent_session_id)
            t_id = len(tasks) + 1
            t_item = {
                "id": t_id,
                "description": description,
                "status": "PENDING",
                "created_at": time.time(),
            }
            tasks.append(t_item)
            self._save(tasks, session_id, user_id, parent_session_id)
            return t_item

    def list_tasks(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            return self._load(session_id, user_id, parent_session_id)

    def update_task(
        self,
        task_id: int,
        status: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        norm_status = _normalize_task_status(status)
        with self._lock:
            tasks = self._load(session_id, user_id, parent_session_id)
            updated_item: Optional[Dict[str, Any]] = None
            for t in tasks:
                if t["id"] == task_id:
                    t["status"] = norm_status
                    t["updated_at"] = time.time()
                    updated_item = t
                    break
            if updated_item is None:
                return None
            self._save(tasks, session_id, user_id, parent_session_id)

            if not self._explicit_filename:
                plan_state = get_plan_state(session_id, user_id, parent_session_id)
                plan_completed = bool(tasks) and not _has_unfinished_or_blocked_work(tasks)
                if plan_completed and plan_state.get("phase") == "EXECUTING":
                    plan_state["phase"] = "COMPLETED"
                    plan_state["status"] = "COMPLETED"
                    plan_state["updated_at"] = time.time()
                    _save_plan_state(plan_state, session_id, user_id, parent_session_id)
                elif not plan_completed and plan_state.get("phase") == "COMPLETED":
                    plan_state["phase"] = "EXECUTING"
                    plan_state["status"] = "EXECUTING"
                    plan_state["updated_at"] = time.time()
                    _save_plan_state(plan_state, session_id, user_id, parent_session_id)

            return updated_item


task_board = TaskBoardState()


class TaskCreateInput(BaseModel):
    task_description: str = Field(description="Clear description of the sub-task item to track.")


@tool(args_schema=TaskCreateInput)
def task_create(task_description: str) -> str:
    """Creates a new tracked sub-task item on the task board and returns its generated ID and Shared Task File path.

    Requires Execution Mode (if currently in Planning Mode, call exit_plan first).
    """
    with _plan_lock:
        block_msg = check_plan_mode_tool_allowed("task_create")
        if block_msg:
            raise ToolException(block_msg)
        if not (task_description or "").strip():
            raise ToolException("[ERROR] 'task_description' cannot be empty.")

        t = task_board.create_task(task_description)
        plan_state = get_plan_state()
        if plan_state.get("phase") == "COMPLETED":
            plan_state["phase"] = "EXECUTING"
            plan_state["status"] = "EXECUTING"
            plan_state["updated_at"] = time.time()
            _save_plan_state(plan_state)

        rel_task_file = _rel_path_str(task_board._resolve_file())
    return (
        f"[TASK CREATED] Task #{t['id']}: '{task_description}' (Status: PENDING) "
        f"| Shared Task File: {rel_task_file}"
    )


@tool
def task_list() -> str:
    """Lists the current plan summary, shared file paths, and all tracked tasks with their IDs and statuses as JSON."""
    plan_dir = _session_plan_dir(ensure_exists=False)
    plan_state = get_plan_state()
    tasks = task_board.list_tasks()
    if not tasks and plan_state.get("phase", "IDLE") == "IDLE" and not plan_state.get("steps"):
        return f"No active plan or tasks currently registered on task board. (Plan Dir: {_rel_path_str(plan_dir)})"

    payload = {
        "plan_file": _rel_path_str(plan_dir / "plan_state.json"),
        "task_file": _rel_path_str(task_board._resolve_file()),
        "plan": {
            "active": bool(plan_state.get("active", False)),
            "phase": plan_state.get("phase", "IDLE"),
            "status": plan_state.get("status", "IDLE"),
            "plan_name": plan_state.get("plan_name", ""),
            "goal": plan_state.get("goal", ""),
            "steps": plan_state.get("steps", []),
        },
        "tasks": tasks,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


class TaskUpdateInput(BaseModel):
    task_id: int = Field(description="Numeric ID of the target task to update.")
    status: TaskStatus = Field(
        description="New status string: 'PENDING', 'IN_PROGRESS', 'COMPLETED', or 'BLOCKED'."
    )

    @field_validator("status", mode="before")
    @classmethod
    def _validate_status(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("status must be a string.")
        return _normalize_task_status(value)


@tool(args_schema=TaskUpdateInput)
def task_update(task_id: int, status: TaskStatus) -> str:
    """Updates a task status on the task board and returns the updated confirmation with Shared Task File path."""
    with _plan_lock:
        t = task_board.update_task(task_id, status)
        if not t:
            existing_ids = [item["id"] for item in task_board.list_tasks()]
            valid_hint = existing_ids if existing_ids else "none (call task_create first)"
            raise ToolException(
                f"[ERROR] Task #{task_id} not found. Valid task IDs on board: {valid_hint}."
            )
    rel_task_file = _rel_path_str(task_board._resolve_file())
    return (
        f"[TASK UPDATED] Task #{task_id} status changed to '{t['status']}' "
        f"| Shared Task File: {rel_task_file}"
    )


def _finalize_plan_tool(t):
    """Planning/Task 도구 공통 메타데이터(docstring 정제, 에러 핸들링)를 일관 적용합니다."""
    if getattr(t, "description", None):
        t.description = inspect.cleandoc(t.description)
    t.handle_tool_error = True
    return t


# 📋 5대 Planning & Task 도구 묶음
tools_planning = [_finalize_plan_tool(t) for t in (enter_plan, exit_plan, task_create, task_list, task_update)]
PLANNING_AND_TASK_TOOLS = tools_planning
