"""
===============================================================================
[H-02] Amnesia Guard Middleware (Advanced Edition)
===============================================================================
Source: frontier_agent_analysis/modules/claude_code/amnesia_guard.py

컨텍스트 컴팩션 후 최근 작업 컨텍스트(파일, 계획) 소실을 방지하는 복원 미들웨어.
- 최근 접근한 파일 경로 추적 (LRU, max_restore_files 제한)
- 활성 계획(Active Plan) 텍스트 보존 (str 및 list[str] 타입 모두 지원)
- 컴팩션 시 create_recovery_attachments()로 복원 블록 생성
  (HumanMessage + harness_origin="amnesia_restore" 표식 + 첫 줄 머리말 — 사용자 턴으로 세지 않음)
- 파일당 max_file_chars 제한 + Head/Tail 트리밍으로 복원 블록 크기 폭발 방지
- 세션(thread_id)별 상태 격리: 서버가 에이전트 인스턴스 1개를 여러 사용자/세션이
  공유하더라도, A 세션의 파일·계획이 B 세션 복원 블록에 섞이지 않습니다.
  (그래프 밖에서 직접 호출하면 'default' 세션 버킷을 사용합니다.)

H-01 (Compactor)의 AutoCompactor/ReactiveCompactor와 쌍으로 동작합니다.
도구 호출 인터셉터(wrap_tool_call / awrap_tool_call)를 통해 자동 추적합니다.
===============================================================================
"""

import os
import threading
from collections import OrderedDict
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain.agents.middleware import AgentMiddleware

from .message_markers import (
    AMNESIA_RESTORE_HEADER,
    AUTO_GENERATED_NOTE,
    ORIGIN_AMNESIA_RESTORE,
    origin_kwargs,
)


DEFAULT_SESSION_KEY = "default"

# 도구 인자 중 파일 경로로 간주할 키 (file_read/file_writer/file_edit → file_path, notebook_edit → notebook_path)
FILE_PATH_ARG_KEYS = ("file_path", "path", "filename", "file", "notebook_path")
# 활성 계획을 등록/갱신하는 도구 (프로젝트 실제 도구: enter_plan(plan_name, steps))
PLAN_SET_TOOLS = {"enter_plan", "update_plan", "create_plan", "set_plan"}
# 활성 계획을 종료하는 도구 (프로젝트 실제 도구: exit_plan(plan_name, status))
PLAN_CLEAR_TOOLS = {"exit_plan"}
PLAN_ARG_KEYS = ("steps", "plan", "content", "text")


def resolve_session_key() -> str:
    """현재 실행 중인 LangGraph 런의 thread_id를 세션 키로 반환합니다.

    create_agent 그래프 안(도구 노드/모델 노드)에서는 config["configurable"]["thread_id"]를,
    그래프 밖(노트북에서 직접 호출 등)이거나 thread_id가 없으면 'default'를 반환합니다.
    """
    try:
        from langgraph.config import get_config
        config = get_config()
    except Exception:
        return DEFAULT_SESSION_KEY
    configurable = (config or {}).get("configurable") or {}
    thread_id = configurable.get("thread_id")
    return str(thread_id) if thread_id else DEFAULT_SESSION_KEY


class _SessionState:
    """세션 1개 분량의 추적 상태."""
    __slots__ = ("recent_files", "active_plan")

    def __init__(self):
        self.recent_files: list[str] = []
        self.active_plan: Optional[str] = None


class AmnesiaGuardMiddleware(AgentMiddleware):
    """Tracks recently accessed files and active plans to restore them after context compaction.
    Limits per-file content to max_file_chars with Head/Tail trimming to prevent token explosion.
    Supports both synchronous (wrap_tool_call) and asynchronous (awrap_tool_call) execution.
    State is isolated per session (LangGraph thread_id) so a shared agent instance never leaks
    one user's files/plan into another session's recovery attachments."""

    def __init__(self, max_restore_files: int = 5, max_file_chars: int = 3000, max_sessions: int = 256):
        self.max_restore_files = max_restore_files
        self.max_file_chars = max_file_chars
        self.max_sessions = max_sessions  # 메모리 보호: 오래된 세션 버킷부터 LRU 제거
        self._sessions: "OrderedDict[str, _SessionState]" = OrderedDict()
        self._lock = threading.RLock()

    # ── Session bucket helpers ──

    def _state(self, session_key: Optional[str] = None) -> _SessionState:
        key = session_key or resolve_session_key()
        with self._lock:
            state = self._sessions.get(key)
            if state is None:
                state = _SessionState()
                self._sessions[key] = state
                while len(self._sessions) > self.max_sessions:
                    self._sessions.popitem(last=False)
            else:
                self._sessions.move_to_end(key)
            return state

    @property
    def recent_files(self) -> list:
        """현재 세션의 최근 파일 목록 (읽기 전용 사본)."""
        return list(self._state().recent_files)

    @property
    def active_plan(self) -> Optional[str]:
        """현재 세션의 활성 계획."""
        return self._state().active_plan

    def clear_session(self, session_key: Optional[str] = None) -> None:
        """세션 버킷을 삭제합니다 (세션 종료 시 정리용)."""
        key = session_key or resolve_session_key()
        with self._lock:
            self._sessions.pop(key, None)

    # ── Tracking API ──

    def track_file_access(self, file_path: str, session_key: Optional[str] = None):
        if not file_path:
            return

        # Normalize path
        normalized = os.path.normpath(file_path)
        state = self._state(session_key)

        with self._lock:
            # Remove if already exists to move it to the end (most recent)
            if normalized in state.recent_files:
                state.recent_files.remove(normalized)

            state.recent_files.append(normalized)

            # Keep only the last max_restore_files
            if len(state.recent_files) > self.max_restore_files:
                state.recent_files.pop(0)

    def set_active_plan(self, plan, session_key: Optional[str] = None):
        """Set active plan. Accepts both str and list[str] types."""
        state = self._state(session_key)
        if isinstance(plan, list):
            state.active_plan = "\n".join(str(item) for item in plan)
        elif isinstance(plan, str):
            state.active_plan = plan

    def clear_active_plan(self, session_key: Optional[str] = None):
        """활성 계획을 해제합니다 (exit_plan 호출 시)."""
        self._state(session_key).active_plan = None

    def _trim_file_content(self, content: str, file_path: str = "") -> str:
        """Trim large file content to max_file_chars using Head(40 lines)/Tail(10 lines) strategy.
        Includes Actionable Hint for full content retrieval."""
        if len(content) <= self.max_file_chars:
            return content

        lines = content.split("\n")
        head_lines = 40
        tail_lines = 10

        if len(lines) <= head_lines + tail_lines:
            # File has few lines but large content per line — truncate by chars
            truncated = content[:self.max_file_chars]
            hint = f"\n... [{len(content) - self.max_file_chars} chars omitted. Use `file_read(file_path='{file_path}')` for full content]"
            return truncated + hint

        head = "\n".join(lines[:head_lines])
        tail = "\n".join(lines[-tail_lines:])
        omitted_count = len(lines) - head_lines - tail_lines

        trimmed = (
            f"{head}\n"
            f"\n... [{omitted_count} lines omitted. Use `file_read(file_path='{file_path}')` for full content] ...\n\n"
            f"{tail}"
        )

        # Final safety check against max_file_chars
        if len(trimmed) > self.max_file_chars:
            trimmed = trimmed[:self.max_file_chars] + f"\n... [Trimmed to {self.max_file_chars} chars]"

        return trimmed

    def create_recovery_attachments(self, session_key: Optional[str] = None) -> list:
        state = self._state(session_key)
        with self._lock:
            active_plan = state.active_plan
            recent_files = list(state.recent_files)

        recovery_sections = []

        if active_plan:
            recovery_sections.append(f"=== Active Plan ===\n{active_plan}")

        if recent_files:
            file_snapshots = []
            for f in recent_files:
                if os.path.exists(f) and os.path.isfile(f):
                    try:
                        with open(f, "r", encoding="utf-8") as file:
                            content = file.read()
                        trimmed_content = self._trim_file_content(content, file_path=f)
                        file_snapshots.append(f"File: {f}\nContent:\n{trimmed_content}")
                    except Exception as e:
                        file_snapshots.append(f"File: {f}\nContent: (Error reading: {e})")
                else:
                    file_snapshots.append(f"File: {f}\nContent: (File does not exist on disk)")

            recovery_sections.append("=== Recently Modified/Accessed Files ===\n" + "\n\n".join(file_snapshots))

        if not recovery_sections:
            return []

        attachment_text = (
            f"{AMNESIA_RESTORE_HEADER}\n{AUTO_GENERATED_NOTE}\n"
            "The following context has been restored from your recent actions:\n\n"
            + "\n\n".join(recovery_sections)
        )
        return [HumanMessage(content=attachment_text, additional_kwargs=origin_kwargs(ORIGIN_AMNESIA_RESTORE))]

    # ── Tool call interception ──

    def _track_tool_request(self, request) -> None:
        tool_call = getattr(request, "tool_call", None) or {}
        tool_name = tool_call.get("name", "")
        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            return

        # 세션 키는 도구 노드 실행 컨텍스트에서 1회만 해석
        session_key = resolve_session_key()

        for key in FILE_PATH_ARG_KEYS:
            if key in args and isinstance(args[key], str):
                self.track_file_access(args[key], session_key=session_key)

        if tool_name in PLAN_SET_TOOLS:
            for key in PLAN_ARG_KEYS:
                val = args.get(key)
                if isinstance(val, (str, list)) and val:
                    plan_name = args.get("plan_name")
                    if isinstance(val, list):
                        body = "\n".join(f"{i}. {step}" for i, step in enumerate(val, 1))
                    else:
                        body = val
                    plan_text = f"[{plan_name}]\n{body}" if isinstance(plan_name, str) and plan_name else body
                    self.set_active_plan(plan_text, session_key=session_key)
                    break
        elif tool_name in PLAN_CLEAR_TOOLS:
            self.clear_active_plan(session_key=session_key)

    def wrap_tool_call(self, request, handler):
        self._track_tool_request(request)
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        self._track_tool_request(request)
        return await handler(request)


def create_amnesia_guard_middleware(amnesia_guard: AmnesiaGuardMiddleware) -> AmnesiaGuardMiddleware:
    """Returns the AmnesiaGuardMiddleware instance (AgentMiddleware subclass supporting both sync and async tool interception)."""
    return amnesia_guard
