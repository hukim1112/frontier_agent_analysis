"""
===============================================================================
[H-04] AgentLogTracer (Unified Async Logging & Observability Middleware)
===============================================================================
감사 로깅(Audit Logging), 실시간 실행 상태 추적(In-Flight Progress Snapshot),
심층 궤적 추적(Tracing & Profiling)을 단일화한 통합 미들웨어:
- 동시 다발 멀티 서브에이전트 환경에서 session_id(sub_thread_id) 단위 완전 격리
- 세션/턴(Turn) 수준의 완결된 계층형 상태 관리 및 진행 상태 스냅샷 조회 (Memory + Disk SSOT)
- LLM 호출 토큰 사용량(입력/출력/총합) 및 레이턴시 정밀 프로파일링 (wrap_model_call / awrap_model_call)
- 도구 격발 상세 및 에러/취소/비정상 종료 포착 (wrap_tool_call / awrap_tool_call)
- 백그라운드 스레드 큐 기반 논블로킹 비동기 파일 저장 (_AsyncLogWorker)
- 세션별 분할(.jsonl), 단일 통합 파일(log_path), 인메모리 전용(disk_enabled=False) 지원
- IPython/Jupyter 인터랙티브 HTML 시각화 대시보드 (Timeline, Tool Summary)
- LoggingMiddleware 하위 호환 Alias 제공
  (AgentTracer는 노트북용 경량 트레이서 agent_tracer.py의 클래스 이름입니다)
===============================================================================
"""

import asyncio
import json
import os
import queue
import threading
import time
import uuid
import weakref
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents.middleware import AgentMiddleware, ModelResponse
from app.middleware.observability.viz import render_timeline_html, render_tool_summary_html
from app.utils.message_markers import last_user_turn_start
from app.utils.message_utils import normalize_content, sanitize_text

try:
    from IPython.display import display, HTML
except ImportError:
    def display(x):
        print(x)

    def HTML(x):
        return x

try:
    from langgraph.errors import GraphBubbleUp as _GraphBubbleUp
except Exception:
    _GraphBubbleUp = ()


class _AsyncLogWorker:
    """
    백그라운드 스레드 기반의 논블로킹 파일 로깅 워커.
    에이전트 추론 루프가 디스크 I/O에 의해 지연되지 않도록 큐를 통해 비동기로 디스크에 적재합니다.
    """

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._run, daemon=True, name="AsyncLogWorker")
        self._worker_thread.start()

    def _run(self):
        while True:
            try:
                target_path, log_entry = self._queue.get()
                if target_path is None:
                    break
                parent_dir = os.path.dirname(target_path)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
                with open(target_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception:
                pass
            finally:
                self._queue.task_done()

    def enqueue(self, target_path: str, log_entry: Dict[str, Any]):
        self._queue.put((target_path, log_entry))

    def flush(self):
        """남아있는 모든 큐 작업을 파일에 쓸 때까지 대기"""
        self._queue.join()


# 프로세스 전역 비동기 로거 싱글턴
_GLOBAL_LOG_WORKER = _AsyncLogWorker()

# 살아있는 AgentLogTracer 인스턴스의 활성 메모리 스냅샷 전역 조회용 WeakSet
_TRACER_REGISTRY_LOCK = threading.RLock()
_TRACER_INSTANCES: "weakref.WeakSet[AgentLogTracer]" = weakref.WeakSet()


_DEFAULT_LOG_DIRS = {
    os.path.normpath("./artifacts/logs"),
    os.path.normpath("artifacts/logs"),
    os.path.normpath("./artifacts/users"),
    os.path.normpath("artifacts/users"),
}

_PRIORITY_ARG_KEYS = (
    "command",
    "url",
    "file_path",
    "path",
    "pattern",
    "query",
    "task_instruction",
    "job_id",
    "output_path",
    "excel_path",
    "html_path",
)

_PATH_ARG_KEYS = (
    "file_path",
    "path",
    "output_path",
    "save_path",
    "excel_path",
    "html_path",
    "chart_path",
    "data_path",
    "script_path",
    "target_file_list",
)

_ERROR_PREFIXES = (
    "[ERROR:",
    "[BLOCKER:",
    "[Tool Error",
    "Error executing tool",
    "ToolException",
)


def _format_scalar_preview(value: Any, max_len: int = 80) -> str:
    if isinstance(value, str):
        clean = " ".join(sanitize_text(value).split())
    else:
        try:
            clean = sanitize_text(json.dumps(value, ensure_ascii=False))
        except Exception:
            clean = sanitize_text(str(value))
    if len(clean) > max_len:
        return clean[: max_len - 3] + "..."
    return clean


def _summarize_tool_args(tool_name: str, tool_args: Any, max_len: int = 160) -> str:
    """도구 인자에서 핵심 파라미터 위주의 간결한 한 줄 요약을 생성합니다."""
    if tool_args is None or tool_args == {}:
        return ""
    if not isinstance(tool_args, dict):
        return _format_scalar_preview(tool_args, max_len=max_len)

    ordered_keys = [k for k in _PRIORITY_ARG_KEYS if k in tool_args]
    ordered_keys.extend(k for k in tool_args.keys() if k not in ordered_keys)

    parts: List[str] = []
    for k in ordered_keys:
        val = tool_args.get(k)
        if val is None or val == "" or val == []:
            continue
        if isinstance(val, str):
            rendered = f'"{_format_scalar_preview(val, max_len=70)}"'
        else:
            rendered = _format_scalar_preview(val, max_len=70)
        parts.append(f"{k}={rendered}")

    summary = ", ".join(parts)
    if len(summary) > max_len:
        return summary[: max_len - 3] + "..."
    return summary


def _extract_file_paths(tool_args: Any) -> List[str]:
    """도구 인자에서 다룬 파일/아티팩트 경로를 추출합니다."""
    if not isinstance(tool_args, dict):
        return []
    found: List[str] = []
    seen = set()
    for key in _PATH_ARG_KEYS:
        val = tool_args.get(key)
        if isinstance(val, str):
            p = val.strip()
            if p and p not in seen:
                seen.add(p)
                found.append(p)
        elif isinstance(val, (list, tuple)):
            for item in val:
                if isinstance(item, str):
                    p = item.strip()
                    if p and p not in seen:
                        seen.add(p)
                        found.append(p)
    return found


def _extract_tool_result_and_status(response: Any) -> Tuple[str, str]:
    """ToolMessage / Command / 문자열 응답에서 텍스트와 실제 성패 상태(SUCCESS | FAILED)를 판별합니다."""
    target_msg = response
    update = getattr(response, "update", None)
    if isinstance(update, dict):
        msgs = update.get("messages")
        if isinstance(msgs, list) and msgs:
            target_msg = msgs[-1]

    if hasattr(target_msg, "content"):
        result_str = sanitize_text(normalize_content(target_msg.content))
    else:
        result_str = sanitize_text(str(target_msg))

    stripped = result_str.lstrip()
    is_error = getattr(target_msg, "status", None) == "error" or stripped.startswith(_ERROR_PREFIXES)
    return result_str, ("FAILED" if is_error else "SUCCESS")


def _extract_token_usage(ai_msg: Any) -> Tuple[int, int]:
    """AIMessage에서 입력/출력 토큰 수를 추출합니다."""
    usage = getattr(ai_msg, "usage_metadata", None) or {}
    if not usage and ai_msg and hasattr(ai_msg, "response_metadata"):
        usage = (ai_msg.response_metadata or {}).get("token_usage", {}) or {}
    input_tokens = int(usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0)
    return input_tokens, output_tokens


def _extract_planned_tools(ai_msg: Any) -> List[str]:
    """AIMessage의 tool_calls에서 호출 예정 도구 이름 목록을 추출합니다."""
    if not ai_msg:
        return []
    raw_calls = getattr(ai_msg, "tool_calls", None) or []
    names: List[str] = []
    for tc in raw_calls:
        if isinstance(tc, dict) and tc.get("name"):
            names.append(str(tc["name"]))
    return names


def _extract_turn_anchor_id(state: Any = None, request: Any = None) -> Optional[str]:
    """익명(thread_id 미지정) 실행에서 동일 턴의 훅들을 상관짓기 위해 최신 HumanMessage.id를 추출합니다."""
    messages = None
    if isinstance(state, dict):
        messages = state.get("messages")
    if messages is None and request is not None:
        req_state = getattr(request, "state", None)
        if isinstance(req_state, dict):
            messages = req_state.get("messages")
        if messages is None:
            messages = getattr(request, "messages", None)
    if not isinstance(messages, (list, tuple)) or not messages:
        return None

    turn_idx = last_user_turn_start(messages)
    target_msg = messages[turn_idx] if turn_idx is not None else messages[0]

    msg_id = target_msg.get("id") if isinstance(target_msg, dict) else getattr(target_msg, "id", None)
    if msg_id and str(msg_id).strip():
        return str(msg_id).strip()
    return None


def _resolve_unified_audit_file(
    session_id: str,
    user_id: Optional[str] = None,
    parent_session_id: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    agent_name: Optional[str] = None,
    ensure_dirs: bool = True,
) -> str:
    """통합 워크스페이스 규칙에 따른 audit.jsonl 경로를 계산합니다."""
    from app.utils.workspace import PROJECT_ROOT, resolve_session_workspace

    ws_dir = (workspace_dir or "").strip()
    if ws_dir:
        ws_path = (PROJECT_ROOT / ws_dir) if not os.path.isabs(ws_dir) else ws_dir
        logs_dir = os.path.join(str(ws_path), "logs")
        if ensure_dirs:
            os.makedirs(logs_dir, exist_ok=True)
        return os.path.join(logs_dir, "audit.jsonl")

    uid = (user_id or "").strip() or "default"
    psid = (parent_session_id or "").strip() or None
    role = (agent_name or "").strip() or None
    ws = resolve_session_workspace(
        user_id=uid,
        session_id=session_id,
        parent_session_id=psid,
        subagent_role=role if psid else None,
        ensure_exists=ensure_dirs,
    )
    return str(ws.logs_dir / "audit.jsonl")


def _render_snapshot_report(snap: Dict[str, Any]) -> str:
    """스냅샷 딕셔너리를 Supervisor/사용자용 다중 행 리포트로 렌더링합니다."""
    if not snap or not snap.get("found"):
        return ""

    elapsed_s = snap.get("elapsed_ms", 0) / 1000.0
    lines = [
        f"- Execution Phase: {snap.get('phase')} (Status: {snap.get('status')}, Turn #{snap.get('turn_id', 1)}, Source: {snap.get('source')})",
        (
            f"- Progress Counters: LLM Steps={snap.get('llm_step_count', 0)}, "
            f"Tool Calls={snap.get('total_tool_calls', 0)} (Failed={snap.get('failed_tool_calls', 0)}), "
            f"Elapsed={elapsed_s:.1f}s"
        ),
    ]

    active_tools = snap.get("active_tools") or []
    active_mc = snap.get("active_model_call")
    if active_tools:
        for t in active_tools:
            t_sec = t.get("elapsed_ms", 0) / 1000.0
            args_str = t.get("args_summary", "")
            lines.append(f"- Active Tool: {t.get('tool_name')}({args_str}) — running for {t_sec:.1f}s")
    elif active_mc:
        mc_sec = active_mc.get("elapsed_ms", 0) / 1000.0
        lines.append(f"- Active LLM Call: Step #{active_mc.get('step', 1)} — thinking for {mc_sec:.1f}s")

    latest_thought = (snap.get("latest_thought") or "").strip()
    if latest_thought:
        lines.append(f"- Latest Reasoning: {latest_thought}")

    recent_steps = snap.get("recent_steps") or []
    if recent_steps:
        lines.append("- Recent Completed Tool Steps:")
        for step in recent_steps:
            raw_status = step.get("status", "SUCCESS")
            tag = {"SUCCESS": "OK", "FAILED": "ERR", "CANCELLED": "CANCELLED"}.get(raw_status, raw_status)
            args_str = step.get("args_summary", "")
            lat = step.get("latency_ms", 0)
            line = f"  • [{tag}] {step.get('tool_name')}({args_str}) [{lat}ms]"
            if raw_status in ("FAILED", "CANCELLED") and step.get("result_snippet"):
                line += f" — {step['result_snippet'][:120]}"
            lines.append(line)

    if snap.get("last_error"):
        lines.append(f"- Last Error: {snap['last_error']}")

    touched = snap.get("touched_files") or []
    if touched:
        lines.append(f"- Touched Artifacts: {', '.join(touched)}")

    return "\n".join(lines)


class AgentLogTracer(AgentMiddleware):
    """
    에이전트 감사 로깅, 실시간 진행 스냅샷, 궤적 관측성 통합 미들웨어 (AgentLogTracer).
    세션 계층 트리, 실시간 작업 상태(In-flight LLM/Tool), 토큰 사용량, 도구 레이턴시를 추적하며
    비동기로 JSONL 감사 로그를 영속화하고 대화형 HTML 시각화 대시보드를 제공합니다.
    """

    def __init__(
        self,
        log_dir: str = "./artifacts/users",
        log_path: Optional[str] = None,
        verbose: bool = False,
        stream_events: bool = True,
        disk_enabled: bool = True,
        max_completed_runs: int = 128,
        stale_active_ttl_s: float = 86400.0,
    ):
        super().__init__()
        self.log_path = log_path
        self.log_dir = os.path.dirname(log_path) if log_path else log_dir
        self.verbose = verbose
        self.stream_events = stream_events  # 개별 tool_execution 이벤트도 실시간 기록할지 여부
        self.disk_enabled = bool(disk_enabled)
        self.max_completed_runs = max(1, int(max_completed_runs))
        self.stale_active_ttl_s = float(stale_active_ttl_s)
        self._use_unified_workspace = (
            not self.log_path
            and bool(self.log_dir)
            and os.path.normpath(self.log_dir) in _DEFAULT_LOG_DIRS
        )

        if self.disk_enabled and self.log_dir and not self._use_unified_workspace:
            os.makedirs(self.log_dir, exist_ok=True)

        # 스레드/코루틴 동시 접근 보호 락 및 세션별 런타임 상태 버퍼
        self._state_lock = threading.RLock()
        self._active_runs: Dict[str, Dict[str, Any]] = {}
        self._turn_counters: Dict[str, int] = {}
        self._last_session_id: Optional[str] = None
        self._anon_turn_sessions: Dict[str, str] = {}

        with _TRACER_REGISTRY_LOCK:
            _TRACER_INSTANCES.add(self)

    # =========================================================================
    # Context / Session / Target File Resolution
    # =========================================================================

    def _get_context_field(
        self,
        field: str,
        runtime: Any = None,
        request: Any = None,
        state: Any = None,
        session_id: Optional[str] = None,
        config_key: Optional[str] = None,
    ) -> Optional[str]:
        """런타임/요청/설정/활성 세션 상태에서 지정 컨텍스트 필드를 우선순위대로 추출합니다."""
        for obj in (
            getattr(runtime, "context", None) if runtime else None,
            getattr(getattr(request, "runtime", None), "context", None) if request else None,
        ):
            if obj is not None:
                val = getattr(obj, field, None)
                if val and str(val).strip():
                    return str(val).strip()

        cfg_field = config_key or field
        for cfg in (
            getattr(getattr(request, "runtime", None), "config", None) if request else None,
            getattr(runtime, "config", None) if runtime else None,
        ):
            if isinstance(cfg, dict):
                val = (cfg.get("configurable") or {}).get(cfg_field)
                if val and str(val).strip():
                    return str(val).strip()

        if isinstance(state, dict):
            cfg = state.get("configurable", {})
            if isinstance(cfg, dict) and cfg.get(cfg_field):
                val = str(cfg[cfg_field]).strip()
                if val:
                    return val

        if session_id:
            with self._state_lock:
                run = self._active_runs.get(session_id)
                if run:
                    val = run.get(field)
                    if val and str(val).strip():
                        return str(val).strip()
        return None

    def _get_user_id(
        self,
        runtime: Any = None,
        request: Any = None,
        state: Any = None,
        session_id: Optional[str] = None,
    ) -> str:
        return (
            self._get_context_field(
                "user_id",
                runtime=runtime,
                request=request,
                state=state,
                session_id=session_id,
            )
            or "default"
        )

    def _get_parent_session_id(
        self,
        runtime: Any = None,
        request: Any = None,
        state: Any = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        return self._get_context_field(
            "parent_session_id",
            runtime=runtime,
            request=request,
            state=state,
            session_id=session_id,
            config_key="parent_thread_id",
        )

    def _get_workspace_dir(
        self,
        runtime: Any = None,
        request: Any = None,
        state: Any = None,
        session_id: Optional[str] = None,
    ) -> str:
        return (
            self._get_context_field(
                "workspace_dir",
                runtime=runtime,
                request=request,
                state=state,
                session_id=session_id,
            )
            or ""
        )

    def _get_agent_name(
        self,
        runtime: Any = None,
        request: Any = None,
        state: Any = None,
        session_id: Optional[str] = None,
    ) -> str:
        return (
            self._get_context_field(
                "agent_name",
                runtime=runtime,
                request=request,
                state=state,
                session_id=session_id,
            )
            or ""
        )

    def _resolve_target_file(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        agent_name: Optional[str] = None,
        runtime: Any = None,
        request: Any = None,
        ensure_dirs: bool = True,
    ) -> str:
        """적재 대상 파일 경로 결정:
        - 단일 파일(log_path) 지정 시 log_path 우선
        - 기본 워크스페이스 모드 시 artifacts/users/{user_id}/sessions/{session_id}/(subagents/{role}_{id}/)logs/audit.jsonl
        - 커스텀 log_dir 지정 시 {log_dir}/{session_id}/audit.jsonl
        """
        if self.log_path:
            return self.log_path
        if self._use_unified_workspace:
            ws_dir = workspace_dir or self._get_workspace_dir(
                runtime=runtime, request=request, session_id=session_id
            )
            uid = user_id or self._get_user_id(runtime=runtime, request=request, session_id=session_id)
            psid = parent_session_id or self._get_parent_session_id(
                runtime=runtime, request=request, session_id=session_id
            )
            role = agent_name or self._get_agent_name(
                runtime=runtime, request=request, session_id=session_id
            )
            return _resolve_unified_audit_file(
                session_id=session_id,
                user_id=uid,
                parent_session_id=psid,
                workspace_dir=ws_dir,
                agent_name=role,
                ensure_dirs=ensure_dirs,
            )

        session_folder = os.path.join(self.log_dir, session_id)
        if ensure_dirs:
            os.makedirs(session_folder, exist_ok=True)
        return os.path.join(session_folder, "audit.jsonl")

    def _enqueue_if_enabled(self, target_file: str, log_entry: Dict[str, Any]) -> None:
        if self.disk_enabled and target_file:
            _GLOBAL_LOG_WORKER.enqueue(target_file, log_entry)

    def _get_session_id(
        self,
        runtime: Any = None,
        request: Any = None,
        state: Any = None,
        for_new_run: bool = False,
    ) -> str:
        """명시적 thread_id를 최우선으로, 없을 경우 context.session_id 및 턴 앵커(HumanMessage.id) 순으로 식별합니다."""
        # 1. LangGraph Runtime의 execution_info.thread_id 탐색
        for rt in (runtime, getattr(request, "runtime", None) if request else None):
            exec_info = getattr(rt, "execution_info", None) if rt else None
            if exec_info:
                tid = getattr(exec_info, "thread_id", None)
                if tid:
                    return str(tid)

        # 2. ToolRequest / ModelRequest의 runtime.config 탐색
        if request and hasattr(request, "runtime") and getattr(request.runtime, "config", None):
            cfg = request.runtime.config
            if isinstance(cfg, dict):
                tid = (cfg.get("configurable") or {}).get("thread_id")
                if tid:
                    return str(tid)

        # 3. LangChain RunnableConfig ContextVar 탐색
        try:
            from langchain_core.runnables.config import var_child_runnable_config

            cfg = var_child_runnable_config.get()
            if cfg and isinstance(cfg, dict):
                tid = (cfg.get("configurable") or {}).get("thread_id")
                if tid:
                    return str(tid)
        except Exception:
            pass

        # 4. AgentRuntime의 config 탐색
        if runtime and hasattr(runtime, "config") and isinstance(runtime.config, dict):
            tid = (runtime.config.get("configurable") or {}).get("thread_id")
            if tid:
                return str(tid)

        # 5. State / request.state 내부의 configurable 탐색
        for st in (state, getattr(request, "state", None) if request else None):
            if isinstance(st, dict):
                cfg = st.get("configurable", {})
                if isinstance(cfg, dict) and cfg.get("thread_id"):
                    return str(cfg["thread_id"])

        # 6. runtime.context / request.runtime.context의 session_id 탐색 (thread_id 미지정 시)
        for rt in (runtime, getattr(request, "runtime", None) if request else None):
            ctx = getattr(rt, "context", None) if rt else None
            if ctx is not None:
                sid = getattr(ctx, "session_id", None)
                if sid and sid != "unknown":
                    return str(sid)

        # 7. 익명 실행 턴 앵커(HumanMessage.id) 기반 상관 (직전 명명 세션으로 절대 폴백하지 않음)
        anchor_id = _extract_turn_anchor_id(state=state, request=request)
        with self._state_lock:
            if not for_new_run and anchor_id and anchor_id in self._anon_turn_sessions:
                return self._anon_turn_sessions[anchor_id]

            generated_id = f"session_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
            if anchor_id:
                if len(self._anon_turn_sessions) >= 256:
                    self._anon_turn_sessions.pop(next(iter(self._anon_turn_sessions)), None)
                self._anon_turn_sessions[anchor_id] = generated_id
            self._last_session_id = generated_id
            return generated_id

    def _sync_session_keys(self, current_session_id: str, migrate_temp: bool = True) -> None:
        """노트북/수동 조회용 최근 세션 포인터만 갱신하며 활성 실행 버퍼는 절대 이동·삭제하지 않습니다."""
        if not current_session_id or current_session_id == "unknown":
            return
        with self._state_lock:
            self._last_session_id = current_session_id

    def _evict_stale_active_runs_locked(self, now: Optional[float] = None) -> None:
        """stale_active_ttl_s 이상 갱신이 없는 RUNNING/INTERRUPTED 세션을 정리해 메모리 누수를 방지합니다."""
        if self.stale_active_ttl_s <= 0:
            return
        current_time = now if now is not None else time.time()
        cutoff = current_time - self.stale_active_ttl_s
        stale_sids = [
            sid
            for sid, r in self._active_runs.items()
            if r.get("status") in ("RUNNING", "INTERRUPTED")
            and float(r["updated_at"] if r.get("updated_at") is not None else (r.get("start_time") or 0.0)) < cutoff
        ]
        if not stale_sids:
            return
        stale_set = set(stale_sids)
        for sid in stale_sids:
            self._active_runs.pop(sid, None)
        stale_anchors = [k for k, v in self._anon_turn_sessions.items() if v in stale_set]
        for k in stale_anchors:
            self._anon_turn_sessions.pop(k, None)

    def _cleanup_finished_session_locked(self, finished_session_id: str) -> None:
        """완료된 세션의 익명 앵커 매핑을 해제하고 완료된 런 버퍼를 max_completed_runs 상한 내로 유지합니다."""
        stale_anchors = [k for k, v in self._anon_turn_sessions.items() if v == finished_session_id]
        for k in stale_anchors:
            self._anon_turn_sessions.pop(k, None)

        self._evict_stale_active_runs_locked()

        terminal = [
            (sid, r)
            for sid, r in self._active_runs.items()
            if r.get("status") in ("SUCCESS", "FAILED", "CANCELLED")
        ]
        excess = len(terminal) - self.max_completed_runs
        if excess > 0:
            terminal.sort(key=lambda item: float(item[1].get("end_time") or item[1].get("updated_at") or 0.0))
            for sid, _ in terminal[:excess]:
                self._active_runs.pop(sid, None)

        counter_limit = self.max_completed_runs * 4
        if len(self._turn_counters) > counter_limit:
            stale_counters = [sid for sid in self._turn_counters.keys() if sid not in self._active_runs]
            for sid in stale_counters[: len(self._turn_counters) - counter_limit]:
                self._turn_counters.pop(sid, None)

    def _new_run_entry(
        self,
        session_id: str,
        user_id: str = "default",
        parent_session_id: Optional[str] = None,
        workspace_dir: str = "",
        agent_name: str = "",
        turn_id: int = 1,
        user_query: str = "unknown",
        start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        now = start_time if start_time is not None else time.time()
        return {
            "session_id": session_id,
            "agent_name": agent_name,
            "user_id": user_id,
            "parent_session_id": parent_session_id,
            "workspace_dir": workspace_dir,
            "turn_id": turn_id,
            "user_query": user_query,
            "start_time": now,
            "updated_at": now,
            "phase": "starting",
            "llm_step_count": 0,
            "latest_thought": "",
            "active_model_call": None,
            "active_tools": {},
            "touched_files": [],
            "last_error": None,
            "events": [],
            "status": "RUNNING",
            "final_response": "",
            "total_latency_ms": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tool_calls": 0,
            "_summary_persisted": False,
        }

    def _ensure_run_info(
        self,
        session_id: str,
        runtime: Any = None,
        request: Any = None,
        state: Any = None,
    ) -> Dict[str, Any]:
        with self._state_lock:
            run_info = self._active_runs.get(session_id)
            if run_info is None:
                user_id = self._get_user_id(runtime=runtime, request=request, state=state, session_id=session_id)
                parent_sid = self._get_parent_session_id(
                    runtime=runtime, request=request, state=state, session_id=session_id
                )
                ws_dir = self._get_workspace_dir(
                    runtime=runtime, request=request, state=state, session_id=session_id
                )
                agent_name = self._get_agent_name(
                    runtime=runtime, request=request, state=state, session_id=session_id
                )
                turn_id = self._turn_counters.get(session_id, 1)
                self._turn_counters.setdefault(session_id, turn_id)
                run_info = self._new_run_entry(
                    session_id=session_id,
                    user_id=user_id,
                    parent_session_id=parent_sid,
                    workspace_dir=ws_dir,
                    agent_name=agent_name,
                    turn_id=turn_id,
                )
                self._active_runs[session_id] = run_info
            return run_info

    # =========================================================================
    # Shared Event Recording Helpers (Sync + Async Reuse)
    # =========================================================================

    def _on_model_start(self, session_id: str, msg_count: int, request: Any = None) -> float:
        start_time = time.time()
        with self._state_lock:
            run_info = self._ensure_run_info(session_id, request=request)
            if run_info.get("status") == "INTERRUPTED":
                run_info["status"] = "RUNNING"
            run_info["phase"] = "thinking"
            run_info["updated_at"] = start_time
            run_info["active_model_call"] = {
                "step": run_info.get("llm_step_count", 0) + 1,
                "started_at": start_time,
                "input_messages_count": msg_count,
            }
        return start_time

    def _on_model_success(
        self, session_id: str, msg_count: int, start_time: float, response: Any
    ) -> Tuple[int, int, int]:
        now = time.time()
        duration_ms = int((now - start_time) * 1000)
        ai_msg = response.result[0] if getattr(response, "result", None) else None
        raw_content = ai_msg.content if ai_msg else ""
        response_text = sanitize_text(normalize_content(raw_content)) if raw_content else ""
        planned_tools = _extract_planned_tools(ai_msg)
        input_tokens, output_tokens = _extract_token_usage(ai_msg)

        event = {
            "type": "model_call",
            "timestamp": now,
            "input_messages_count": msg_count,
            "response_snippet": response_text[:200],
            "planned_tools": planned_tools,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": duration_ms,
        }

        with self._state_lock:
            run_info = self._ensure_run_info(session_id)
            run_info["llm_step_count"] = run_info.get("llm_step_count", 0) + 1
            if response_text.strip():
                run_info["latest_thought"] = response_text[:300]
            elif planned_tools:
                run_info["latest_thought"] = f"Planned tool call(s): {', '.join(planned_tools)}"
            run_info["active_model_call"] = None
            if run_info.get("active_tools"):
                run_info["phase"] = "running_tool"
            elif planned_tools:
                run_info["phase"] = "awaiting_tools"
            else:
                run_info["phase"] = "idle"
            run_info["updated_at"] = now
            run_info["events"].append(event)
            run_info["total_input_tokens"] = run_info.get("total_input_tokens", 0) + input_tokens
            run_info["total_output_tokens"] = run_info.get("total_output_tokens", 0) + output_tokens

        return duration_ms, input_tokens, output_tokens

    def _on_model_error(
        self,
        session_id: str,
        msg_count: int,
        start_time: float,
        exc: BaseException,
        request: Any = None,
    ) -> None:
        if _GraphBubbleUp and isinstance(exc, _GraphBubbleUp):
            with self._state_lock:
                run_info = self._active_runs.get(session_id)
                if run_info:
                    run_info["active_model_call"] = None
                    run_info["status"] = "INTERRUPTED"
                    run_info["phase"] = "interrupted"
                    run_info["updated_at"] = time.time()
            return

        is_cancelled = isinstance(exc, asyncio.CancelledError)
        run_status = "CANCELLED" if is_cancelled else "FAILED"
        now = time.time()
        duration_ms = int((now - start_time) * 1000)
        err_text = f"{type(exc).__name__}: {exc}"[:250]

        event = {
            "type": "model_call",
            "timestamp": now,
            "input_messages_count": msg_count,
            "response_snippet": err_text,
            "planned_tools": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": duration_ms,
            "status": run_status,
        }

        with self._state_lock:
            run_info = self._ensure_run_info(session_id, request=request)
            run_info["active_model_call"] = None
            run_info["updated_at"] = now
            run_info["events"].append(event)
            if not is_cancelled:
                run_info["last_error"] = f"LLM: {err_text}"

        self._finalize_abnormal_run(
            session_id=session_id,
            status=run_status,
            error_msg=f"LLM {run_status.lower()}: {err_text}",
            request=request,
        )

    def _on_tool_start(
        self,
        session_id: str,
        tool_call: Any,
        request: Any = None,
    ) -> Tuple[str, str, Any, str, float]:
        tc = tool_call if isinstance(tool_call, dict) else {}
        tool_name = str(tc.get("name", "unknown_tool"))
        tool_args = tc.get("args", {})
        call_id = str(tc.get("id") or f"{tool_name}_{uuid.uuid4().hex[:6]}")
        args_summary = _summarize_tool_args(tool_name, tool_args)
        start_time = time.time()

        with self._state_lock:
            run_info = self._ensure_run_info(session_id, request=request)
            if run_info.get("status") == "INTERRUPTED":
                run_info["status"] = "RUNNING"
            run_info["active_tools"][call_id] = {
                "tool_call_id": call_id,
                "tool_name": tool_name,
                "args_summary": args_summary,
                "arguments": tool_args,
                "started_at": start_time,
            }
            run_info["phase"] = "running_tool"
            run_info["updated_at"] = start_time

        return call_id, tool_name, tool_args, args_summary, start_time

    def _on_tool_success(
        self,
        session_id: str,
        call_id: str,
        tool_name: str,
        tool_args: Any,
        args_summary: str,
        start_time: float,
        response: Any,
        request: Any = None,
    ) -> Tuple[int, str]:
        now = time.time()
        duration_ms = int((now - start_time) * 1000)
        result_str, tool_status = _extract_tool_result_and_status(response)

        event = {
            "type": "tool_call",
            "timestamp": now,
            "tool_name": tool_name,
            "arguments": tool_args,
            "args_summary": args_summary,
            "result_snippet": result_str[:250],
            "latency_ms": duration_ms,
            "status": tool_status,
        }

        with self._state_lock:
            run_info = self._ensure_run_info(session_id, request=request)
            run_info["active_tools"].pop(call_id, None)
            run_info["phase"] = "running_tool" if run_info["active_tools"] else "idle"
            run_info["updated_at"] = now
            run_info["events"].append(event)
            run_info["total_tool_calls"] = run_info.get("total_tool_calls", 0) + 1
            if tool_status == "FAILED":
                run_info["last_error"] = f"{tool_name}: {result_str[:250]}"
            else:
                touched = run_info.setdefault("touched_files", [])
                for fpath in _extract_file_paths(tool_args):
                    if fpath not in touched:
                        touched.append(fpath)

        if self.stream_events:
            uid = self._get_user_id(request=request, session_id=session_id)
            tool_log = {
                "event": "tool_execution",
                "session_id": session_id,
                "user_id": uid,
                "tool_name": tool_name,
                "arguments": tool_args,
                "args_summary": args_summary,
                "result": result_str,
                "latency_ms": duration_ms,
                "status": tool_status,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            target_file = self._resolve_target_file(
                session_id, user_id=uid, request=request, ensure_dirs=self.disk_enabled
            )
            self._enqueue_if_enabled(target_file, tool_log)

        return duration_ms, tool_status

    def _on_tool_error(
        self,
        session_id: str,
        call_id: str,
        tool_name: str,
        tool_args: Any,
        args_summary: str,
        start_time: float,
        exc: BaseException,
        request: Any = None,
    ) -> None:
        if _GraphBubbleUp and isinstance(exc, _GraphBubbleUp):
            with self._state_lock:
                run_info = self._active_runs.get(session_id)
                if run_info:
                    run_info["active_tools"].pop(call_id, None)
                    run_info["status"] = "INTERRUPTED"
                    run_info["phase"] = "interrupted"
                    run_info["updated_at"] = time.time()
            return

        is_cancelled = isinstance(exc, asyncio.CancelledError)
        run_status = "CANCELLED" if is_cancelled else "FAILED"
        now = time.time()
        duration_ms = int((now - start_time) * 1000)
        err_text = f"{type(exc).__name__}: {exc}"[:250]

        event = {
            "type": "tool_call",
            "timestamp": now,
            "tool_name": tool_name,
            "arguments": tool_args,
            "args_summary": args_summary,
            "result_snippet": err_text,
            "latency_ms": duration_ms,
            "status": run_status,
        }

        with self._state_lock:
            run_info = self._ensure_run_info(session_id, request=request)
            run_info["active_tools"].pop(call_id, None)
            run_info["updated_at"] = now
            run_info["events"].append(event)
            run_info["total_tool_calls"] = run_info.get("total_tool_calls", 0) + 1
            if not is_cancelled:
                run_info["last_error"] = f"{tool_name}: {err_text}"

        if self.stream_events:
            uid = self._get_user_id(request=request, session_id=session_id)
            tool_log = {
                "event": "tool_execution",
                "session_id": session_id,
                "user_id": uid,
                "tool_name": tool_name,
                "arguments": tool_args,
                "args_summary": args_summary,
                "result": err_text,
                "latency_ms": duration_ms,
                "status": run_status,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            target_file = self._resolve_target_file(
                session_id, user_id=uid, request=request, ensure_dirs=self.disk_enabled
            )
            self._enqueue_if_enabled(target_file, tool_log)

        self._finalize_abnormal_run(
            session_id=session_id,
            status=run_status,
            error_msg=f"Tool '{tool_name}' {run_status.lower()}: {err_text}",
            request=request,
        )

    def _build_turn_summary(
        self,
        run_info: Dict[str, Any],
        dialogue_history: Optional[List[Dict[str, str]]] = None,
        error: str = "",
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "event": "turn_summary",
            "session_id": run_info.get("session_id", "unknown"),
            "agent_name": run_info.get("agent_name", ""),
            "user_id": run_info.get("user_id", "default"),
            "parent_session_id": run_info.get("parent_session_id"),
            "turn_id": run_info.get("turn_id", 1),
            "status": run_info.get("status", "SUCCESS"),
            "started_at": run_info.get("start_time"),
            "user_query": run_info.get("user_query", ""),
            "final_response": run_info.get("final_response", ""),
            "total_latency_ms": run_info.get("total_latency_ms", 0),
            "token_usage": {
                "prompt_tokens": run_info.get("total_input_tokens", 0),
                "completion_tokens": run_info.get("total_output_tokens", 0),
                "total_tokens": (
                    run_info.get("total_input_tokens", 0) + run_info.get("total_output_tokens", 0)
                ),
            },
            "total_tool_calls": run_info.get("total_tool_calls", 0),
            "touched_files": list(run_info.get("touched_files", [])),
            "last_error": run_info.get("last_error"),
            "steps": list(run_info.get("events", [])),
            "dialogue_history": dialogue_history or [],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if error:
            summary["error"] = error
        return summary

    def _finalize_abnormal_run(
        self,
        session_id: str,
        status: str,
        error_msg: str = "",
        request: Any = None,
        runtime: Any = None,
    ) -> None:
        """예외(FAILED) 또는 태스크 취소(CANCELLED)로 after_agent 없이 종료될 때 런 상태 및 디스크 로그를 마감합니다."""
        audit_log: Optional[Dict[str, Any]] = None
        target_uid = "default"
        target_psid: Optional[str] = None
        target_ws = ""
        target_agent = ""

        with self._state_lock:
            run_info = self._active_runs.get(session_id)
            if not run_info:
                return
            now = time.time()
            run_info["status"] = status
            run_info["phase"] = status.lower()
            run_info["end_time"] = now
            run_info["updated_at"] = now
            run_info["total_latency_ms"] = int((now - run_info["start_time"]) * 1000)
            run_info["active_model_call"] = None
            run_info["active_tools"].clear()
            if error_msg and status != "CANCELLED" and not run_info.get("last_error"):
                run_info["last_error"] = error_msg[:250]

            if not run_info.get("_summary_persisted"):
                run_info["_summary_persisted"] = True
                audit_log = self._build_turn_summary(run_info, error=error_msg)
                target_uid = run_info.get("user_id", "default")
                target_psid = run_info.get("parent_session_id")
                target_ws = run_info.get("workspace_dir", "")
                target_agent = run_info.get("agent_name", "")

            self._cleanup_finished_session_locked(session_id)

        if audit_log is not None:
            target_file = self._resolve_target_file(
                session_id,
                user_id=target_uid,
                parent_session_id=target_psid,
                workspace_dir=target_ws,
                agent_name=target_agent,
                runtime=runtime,
                request=request,
                ensure_dirs=self.disk_enabled,
            )
            self._enqueue_if_enabled(target_file, audit_log)

    # =========================================================================
    # Agent Lifecycle Hooks
    # =========================================================================

    def before_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        """에이전트 전체 실행 시작 시 세션 상태 초기화"""
        session_id = self._get_session_id(runtime=runtime, state=state, for_new_run=True)
        user_id = self._get_user_id(runtime=runtime, state=state, session_id=session_id)
        parent_session_id = self._get_parent_session_id(runtime=runtime, state=state, session_id=session_id)
        workspace_dir = self._get_workspace_dir(runtime=runtime, state=state, session_id=session_id)
        agent_name = self._get_agent_name(runtime=runtime, state=state, session_id=session_id)
        self._sync_session_keys(session_id, migrate_temp=False)

        messages = state.get("messages", [])
        user_query = normalize_content(messages[-1].content) if messages else "unknown"

        start_time = time.time()
        with self._state_lock:
            self._evict_stale_active_runs_locked(now=start_time)
            self._turn_counters[session_id] = self._turn_counters.get(session_id, 0) + 1
            turn_id = self._turn_counters[session_id]
            self._active_runs[session_id] = self._new_run_entry(
                session_id=session_id,
                user_id=user_id,
                parent_session_id=parent_session_id,
                workspace_dir=workspace_dir,
                agent_name=agent_name,
                turn_id=turn_id,
                user_query=user_query,
                start_time=start_time,
            )

        if runtime and getattr(runtime, "context", None) is not None:
            runtime.context.start_time = start_time
            runtime.context.user_query = user_query
            runtime.context.turn_id = turn_id

        if self.verbose:
            print(f"\n🪵 [AgentLogTracer] === 에이전트 실행 시작 (Turn #{turn_id}) ===")
            print(f"📥 사용자 질문: {user_query}")
            print(f"🆔 세션 ID: {session_id} (user={user_id})")

        return None

    def after_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        """에이전트 전체 실행 완료 시 계층적 턴 감사 로그(turn_summary)를 비동기로 영속화"""
        session_id = self._get_session_id(runtime=runtime, state=state)
        user_id = self._get_user_id(runtime=runtime, state=state, session_id=session_id)
        self._sync_session_keys(session_id)

        messages = state.get("messages", [])
        final_resp = ""
        dialogue_history: List[Dict[str, str]] = []
        if messages:
            final_resp = normalize_content(messages[-1].content)
            for msg in messages:
                dialogue_history.append({
                    "role": getattr(msg, "type", "unknown"),
                    "content": sanitize_text(normalize_content(msg.content)),
                })

        with self._state_lock:
            run_info = self._ensure_run_info(session_id, runtime=runtime, state=state)
            now = time.time()
            duration_ms = int((now - run_info["start_time"]) * 1000)
            run_info["total_latency_ms"] = duration_ms
            run_info["end_time"] = now
            run_info["updated_at"] = now
            run_info["active_model_call"] = None
            run_info["active_tools"].clear()
            run_info["final_response"] = final_resp
            if messages:
                run_info["status"] = "SUCCESS"
                run_info["phase"] = "completed"
            else:
                run_info["status"] = "FAILED"
                run_info["phase"] = "failed"
            run_info["_summary_persisted"] = True

            audit_log = self._build_turn_summary(run_info, dialogue_history=dialogue_history)
            target_uid = run_info.get("user_id", user_id)
            target_psid = run_info.get("parent_session_id")
            target_ws = run_info.get("workspace_dir", "")
            target_agent = run_info.get("agent_name", "")
            final_status = run_info["status"]
            self._cleanup_finished_session_locked(session_id)

        target_file = self._resolve_target_file(
            session_id,
            user_id=target_uid,
            parent_session_id=target_psid,
            workspace_dir=target_ws,
            agent_name=target_agent,
            runtime=runtime,
            ensure_dirs=self.disk_enabled,
        )
        self._enqueue_if_enabled(target_file, audit_log)

        if self.verbose:
            print(f"📤 에이전트 실행 완료 (소요: {duration_ms}ms, 상태: {final_status})")

        return None

    def wrap_model_call(self, request: Any, handler: Any) -> ModelResponse:
        """LLM 호출 토큰 사용량과 레이턴시 정밀 프로파일링"""
        session_id = self._get_session_id(request=request)
        self._sync_session_keys(session_id)
        msg_count = len(request.messages) if hasattr(request, "messages") else 0
        start_time = self._on_model_start(session_id, msg_count, request=request)

        try:
            response = handler(request)
        except (asyncio.CancelledError, Exception) as exc:
            self._on_model_error(session_id, msg_count, start_time, exc, request=request)
            raise

        duration_ms, input_tokens, output_tokens = self._on_model_success(
            session_id, msg_count, start_time, response
        )

        if self.verbose:
            print(
                f"🧠 [AgentLogTracer] LLM 응답 완료 (Prompt: {input_tokens}, Completion: {output_tokens}, 소요: {duration_ms}ms)"
            )

        return response

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        """도구 격발 상세 추적 및 실시간 이벤트 스트리밍"""
        session_id = self._get_session_id(request=request)
        self._sync_session_keys(session_id)
        call_id, tool_name, tool_args, args_summary, start_time = self._on_tool_start(
            session_id, getattr(request, "tool_call", None), request=request
        )

        if self.verbose:
            print(f"🔧 [AgentLogTracer] ➡️ 도구 격발 시작: {tool_name}({tool_args})")

        try:
            response = handler(request)
        except (asyncio.CancelledError, Exception) as exc:
            self._on_tool_error(
                session_id, call_id, tool_name, tool_args, args_summary, start_time, exc, request=request
            )
            raise

        duration_ms, _ = self._on_tool_success(
            session_id, call_id, tool_name, tool_args, args_summary, start_time, response, request=request
        )

        if self.verbose:
            print(f"🔧 [AgentLogTracer] ⬅️ 도구 격발 완료: {tool_name} (소요: {duration_ms}ms)")

        return response

    # =========================================================================
    # Async Middleware Wrappers
    # =========================================================================

    async def abefore_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        return self.before_agent(state, runtime)

    async def aafter_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        return self.after_agent(state, runtime)

    async def awrap_model_call(self, request: Any, handler: Any) -> ModelResponse:
        session_id = self._get_session_id(request=request)
        self._sync_session_keys(session_id)
        msg_count = len(request.messages) if hasattr(request, "messages") else 0
        start_time = self._on_model_start(session_id, msg_count, request=request)

        try:
            response = await handler(request)
        except (asyncio.CancelledError, Exception) as exc:
            self._on_model_error(session_id, msg_count, start_time, exc, request=request)
            raise

        self._on_model_success(session_id, msg_count, start_time, response)
        return response

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        session_id = self._get_session_id(request=request)
        self._sync_session_keys(session_id)
        call_id, tool_name, tool_args, args_summary, start_time = self._on_tool_start(
            session_id, getattr(request, "tool_call", None), request=request
        )

        try:
            response = await handler(request)
        except (asyncio.CancelledError, Exception) as exc:
            self._on_tool_error(
                session_id, call_id, tool_name, tool_args, args_summary, start_time, exc, request=request
            )
            raise

        self._on_tool_success(
            session_id, call_id, tool_name, tool_args, args_summary, start_time, response, request=request
        )
        return response

    # =========================================================================
    # Real-Time Progress Snapshot & Report API (In-Memory First + Disk SSOT Fallback)
    # =========================================================================

    def _build_snapshot_from_run(self, run_info: Dict[str, Any], recent_limit: int = 5) -> Dict[str, Any]:
        now = time.time()
        status = run_info.get("status", "RUNNING")
        start_time = run_info.get("start_time", now)
        if status != "RUNNING" and run_info.get("end_time"):
            elapsed_ms = int(
                run_info.get("total_latency_ms") or ((run_info["end_time"] - start_time) * 1000)
            )
        else:
            elapsed_ms = max(0, int((now - start_time) * 1000))

        mc = run_info.get("active_model_call")
        active_model = None
        if isinstance(mc, dict):
            active_model = {
                "step": mc.get("step", 1),
                "started_at": mc.get("started_at", now),
                "input_messages_count": mc.get("input_messages_count", 0),
                "elapsed_ms": max(0, int((now - mc.get("started_at", now)) * 1000)),
            }

        active_tools: List[Dict[str, Any]] = []
        raw_active_tools = run_info.get("active_tools") or {}
        for item in sorted(raw_active_tools.values(), key=lambda x: x.get("started_at", now)):
            active_tools.append({
                "tool_call_id": item.get("tool_call_id", ""),
                "tool_name": item.get("tool_name", "unknown_tool"),
                "args_summary": item.get("args_summary", ""),
                "started_at": item.get("started_at", now),
                "elapsed_ms": max(0, int((now - item.get("started_at", now)) * 1000)),
            })

        tool_events = [e for e in run_info.get("events", []) if e.get("type") == "tool_call"]
        failed_tool_calls = sum(
            1 for e in tool_events if e.get("status") in ("FAILED", "CANCELLED")
        )
        recent_slice = tool_events[-recent_limit:] if recent_limit > 0 else []
        recent_steps = [
            {
                "tool_name": e.get("tool_name", "unknown_tool"),
                "args_summary": e.get("args_summary") or _summarize_tool_args(
                    str(e.get("tool_name", "")), e.get("arguments")
                ),
                "status": e.get("status", "SUCCESS"),
                "latency_ms": e.get("latency_ms", 0),
                "result_snippet": e.get("result_snippet", ""),
            }
            for e in recent_slice
        ]

        in_tokens = int(run_info.get("total_input_tokens", 0))
        out_tokens = int(run_info.get("total_output_tokens", 0))
        return {
            "found": True,
            "source": "memory",
            "session_id": run_info.get("session_id", ""),
            "agent_name": run_info.get("agent_name", ""),
            "turn_id": run_info.get("turn_id", 1),
            "status": status,
            "phase": run_info.get("phase", "idle"),
            "started_at": start_time,
            "elapsed_ms": elapsed_ms,
            "llm_step_count": run_info.get("llm_step_count", 0),
            "total_tool_calls": run_info.get("total_tool_calls", len(tool_events)),
            "failed_tool_calls": failed_tool_calls,
            "latest_thought": run_info.get("latest_thought", ""),
            "active_model_call": active_model,
            "active_tools": active_tools,
            "recent_steps": recent_steps,
            "touched_files": list(run_info.get("touched_files", [])),
            "last_error": run_info.get("last_error"),
            "token_usage": {
                "prompt_tokens": in_tokens,
                "completion_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens,
            },
        }

    @staticmethod
    def _build_snapshot_from_disk(
        session_id: str,
        target_file: str,
        recent_limit: int = 5,
    ) -> Dict[str, Any]:
        if not target_file or not os.path.exists(target_file):
            return {"found": False, "source": "none", "session_id": session_id}

        latest_turn_summary: Optional[Dict[str, Any]] = None
        tool_executions: List[Dict[str, Any]] = []

        try:
            with open(target_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if rec.get("session_id") != session_id:
                        continue
                    ev_type = rec.get("event")
                    if ev_type == "turn_summary":
                        latest_turn_summary = rec
                    elif ev_type == "tool_execution":
                        tool_executions.append(rec)
        except Exception:
            return {"found": False, "source": "none", "session_id": session_id}

        if latest_turn_summary is not None:
            steps = latest_turn_summary.get("steps") or []
            model_steps = [s for s in steps if s.get("type") == "model_call"]
            tool_steps = [s for s in steps if s.get("type") == "tool_call"]
            failed_tool_calls = sum(
                1 for s in tool_steps if s.get("status") in ("FAILED", "CANCELLED")
            )
            recent_slice = tool_steps[-recent_limit:] if recent_limit > 0 else []
            recent_steps = [
                {
                    "tool_name": s.get("tool_name", "unknown_tool"),
                    "args_summary": s.get("args_summary") or _summarize_tool_args(
                        str(s.get("tool_name", "")), s.get("arguments")
                    ),
                    "status": s.get("status", "SUCCESS"),
                    "latency_ms": s.get("latency_ms", 0),
                    "result_snippet": s.get("result_snippet", ""),
                }
                for s in recent_slice
            ]

            touched_files = list(latest_turn_summary.get("touched_files") or [])
            if not touched_files:
                seen = set()
                for s in tool_steps:
                    if s.get("status", "SUCCESS") == "SUCCESS":
                        for p in _extract_file_paths(s.get("arguments")):
                            if p not in seen:
                                seen.add(p)
                                touched_files.append(p)

            latest_thought = ""
            for ms in reversed(model_steps):
                snip = (ms.get("response_snippet") or "").strip()
                if snip:
                    latest_thought = snip
                    break
            if not latest_thought:
                latest_thought = (latest_turn_summary.get("final_response") or "")[:200]

            last_error = latest_turn_summary.get("last_error") or latest_turn_summary.get("error") or None
            if not last_error:
                for s in reversed(tool_steps):
                    if s.get("status") in ("FAILED", "CANCELLED"):
                        last_error = f"{s.get('tool_name', 'tool')}: {s.get('result_snippet', '')}"
                        break

            status = latest_turn_summary.get("status", "SUCCESS")
            phase = {
                "SUCCESS": "completed",
                "FAILED": "failed",
                "CANCELLED": "cancelled",
            }.get(status, str(status).lower())
            tok = latest_turn_summary.get("token_usage") or {}
            in_t = int(tok.get("prompt_tokens", 0) or 0)
            out_t = int(tok.get("completion_tokens", 0) or 0)

            return {
                "found": True,
                "source": "disk",
                "session_id": session_id,
                "agent_name": latest_turn_summary.get("agent_name", ""),
                "turn_id": latest_turn_summary.get("turn_id", 1),
                "status": status,
                "phase": phase,
                "started_at": latest_turn_summary.get("started_at"),
                "elapsed_ms": int(latest_turn_summary.get("total_latency_ms", 0) or 0),
                "llm_step_count": len(model_steps),
                "total_tool_calls": int(
                    latest_turn_summary.get("total_tool_calls", len(tool_steps)) or 0
                ),
                "failed_tool_calls": failed_tool_calls,
                "latest_thought": latest_thought,
                "active_model_call": None,
                "active_tools": [],
                "recent_steps": recent_steps,
                "touched_files": touched_files,
                "last_error": last_error,
                "token_usage": {
                    "prompt_tokens": in_t,
                    "completion_tokens": out_t,
                    "total_tokens": int(tok.get("total_tokens", in_t + out_t) or 0),
                },
            }

        if tool_executions:
            failed_tool_calls = sum(
                1 for e in tool_executions if e.get("status") in ("FAILED", "CANCELLED")
            )
            recent_slice = tool_executions[-recent_limit:] if recent_limit > 0 else []
            recent_steps = [
                {
                    "tool_name": e.get("tool_name", "unknown_tool"),
                    "args_summary": e.get("args_summary") or _summarize_tool_args(
                        str(e.get("tool_name", "")), e.get("arguments")
                    ),
                    "status": e.get("status", "SUCCESS"),
                    "latency_ms": e.get("latency_ms", 0),
                    "result_snippet": str(e.get("result", ""))[:250],
                }
                for e in recent_slice
            ]
            touched_files = []
            seen = set()
            last_error = None
            total_ms = 0
            for e in tool_executions:
                total_ms += int(e.get("latency_ms", 0) or 0)
                if e.get("status") in ("FAILED", "CANCELLED"):
                    last_error = f"{e.get('tool_name', 'tool')}: {str(e.get('result', ''))[:250]}"
                else:
                    for p in _extract_file_paths(e.get("arguments")):
                        if p not in seen:
                            seen.add(p)
                            touched_files.append(p)

            return {
                "found": True,
                "source": "disk",
                "session_id": session_id,
                "agent_name": "",
                "turn_id": 1,
                "status": "UNKNOWN",
                "phase": "partial_disk_log",
                "started_at": None,
                "elapsed_ms": total_ms,
                "llm_step_count": 0,
                "total_tool_calls": len(tool_executions),
                "failed_tool_calls": failed_tool_calls,
                "latest_thought": "",
                "active_model_call": None,
                "active_tools": [],
                "recent_steps": recent_steps,
                "touched_files": touched_files,
                "last_error": last_error,
                "token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }

        return {"found": False, "source": "none", "session_id": session_id}

    def get_progress_snapshot(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        recent_limit: int = 5,
    ) -> Dict[str, Any]:
        """특정 세션의 현재 진행 스냅샷(인메모리 우선, 없으면 디스크 audit.jsonl 폴백)을 반환합니다."""
        if not session_id:
            return {"found": False, "source": "none", "session_id": ""}

        with self._state_lock:
            run_info = self._active_runs.get(session_id)
            if run_info is not None:
                return self._build_snapshot_from_run(run_info, recent_limit=recent_limit)

        target_file = self._resolve_target_file(
            session_id,
            user_id=user_id,
            parent_session_id=parent_session_id,
            workspace_dir=workspace_dir,
            ensure_dirs=False,
        )
        return self._build_snapshot_from_disk(session_id, target_file, recent_limit=recent_limit)

    def format_progress_report(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        recent_limit: int = 5,
    ) -> str:
        """스냅샷 데이터를 Supervisor/사용자가 바로 읽을 수 있는 다중 행 리포트로 포맷팅합니다."""
        snap = self.get_progress_snapshot(
            session_id=session_id,
            user_id=user_id,
            parent_session_id=parent_session_id,
            workspace_dir=workspace_dir,
            recent_limit=recent_limit,
        )
        return _render_snapshot_report(snap)

    @classmethod
    def get_session_snapshot(
        cls,
        session_id: str,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        recent_limit: int = 5,
        allow_disk_fallback: bool = True,
    ) -> Dict[str, Any]:
        """WeakSet 레지스트리에 등록된 활성 Tracer 메모리 -> 디스크 SSOT 순으로 세션 진행 스냅샷을 전역 조회합니다."""
        if not session_id:
            return {"found": False, "source": "none", "session_id": ""}

        with _TRACER_REGISTRY_LOCK:
            tracers = list(_TRACER_INSTANCES)

        candidates: List[Tuple[bool, float, Dict[str, Any]]] = []
        for tracer in tracers:
            with tracer._state_lock:
                run_info = tracer._active_runs.get(session_id)
                if run_info is not None:
                    snap = tracer._build_snapshot_from_run(run_info, recent_limit=recent_limit)
                    is_active = run_info.get("status") in ("RUNNING", "INTERRUPTED")
                    updated_at = float(run_info.get("updated_at", 0) or 0)
                    candidates.append((is_active, updated_at, snap))

        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
            return candidates[0][2]

        if not allow_disk_fallback:
            return {"found": False, "source": "none", "session_id": session_id}

        # 인메모리 히트가 없으면 등록된 Tracer 대상 경로 및 통합 워크스페이스 디스크 audit.jsonl 폴백 조회
        candidate_paths: List[str] = []
        seen_paths = set()
        for tracer in tracers:
            if tracer.disk_enabled or tracer.log_path:
                path = tracer._resolve_target_file(
                    session_id,
                    user_id=user_id,
                    parent_session_id=parent_session_id,
                    workspace_dir=workspace_dir,
                    ensure_dirs=False,
                )
                if path and path not in seen_paths:
                    seen_paths.add(path)
                    candidate_paths.append(path)

        unified_path = _resolve_unified_audit_file(
            session_id=session_id,
            user_id=user_id,
            parent_session_id=parent_session_id,
            workspace_dir=workspace_dir,
            ensure_dirs=False,
        )
        if unified_path and unified_path not in seen_paths:
            seen_paths.add(unified_path)
            candidate_paths.append(unified_path)

        for path in candidate_paths:
            snap = cls._build_snapshot_from_disk(session_id, path, recent_limit=recent_limit)
            if snap.get("found"):
                return snap

        return {"found": False, "source": "none", "session_id": session_id}

    @classmethod
    def format_session_report(
        cls,
        session_id: str,
        user_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        recent_limit: int = 5,
    ) -> str:
        """전역 WeakSet 스냅샷 조회 결과를 다중 행 리포트로 포맷팅합니다."""
        snap = cls.get_session_snapshot(
            session_id=session_id,
            user_id=user_id,
            parent_session_id=parent_session_id,
            workspace_dir=workspace_dir,
            recent_limit=recent_limit,
        )
        return _render_snapshot_report(snap)

    # =========================================================================
    # Standalone Manual API (단위 테스트 및 스크립트 모드 지원)
    # =========================================================================

    def start_turn(self, session_id: str, user_query: str = ""):
        """수동으로 세션 턴을 개시할 때 호출"""
        with self._state_lock:
            self._evict_stale_active_runs_locked()
            self._last_session_id = session_id
            self._turn_counters[session_id] = self._turn_counters.get(session_id, 0) + 1
            self._active_runs[session_id] = self._new_run_entry(
                session_id=session_id,
                turn_id=self._turn_counters[session_id],
                user_query=user_query,
            )

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        duration_ms: float = 0.0,
        success: bool = True,
        session_id: Optional[str] = None,
    ):
        """수동으로 도구 호출 이벤트를 기록할 때 호출"""
        sid = session_id or self._last_session_id
        with self._state_lock:
            if not sid or sid not in self._active_runs:
                self.start_turn(sid or "default_session")
                sid = self._last_session_id

            args_summary = _summarize_tool_args(tool_name, arguments)
            tool_status = "SUCCESS" if success else "FAILED"
            event = {
                "type": "tool_call",
                "timestamp": time.time(),
                "tool_name": tool_name,
                "arguments": arguments,
                "args_summary": args_summary,
                "result_snippet": "success" if success else "failed",
                "latency_ms": int(duration_ms),
                "status": tool_status,
            }
            run_info = self._active_runs[sid]
            run_info["events"].append(event)
            run_info["total_tool_calls"] += 1
            run_info["updated_at"] = time.time()
            if success:
                touched = run_info.setdefault("touched_files", [])
                for fpath in _extract_file_paths(arguments):
                    if fpath not in touched:
                        touched.append(fpath)
            else:
                run_info["last_error"] = f"{tool_name}: failed"

    def end_turn(
        self,
        token_usage: Optional[Dict[str, int]] = None,
        final_response: str = "",
        session_id: Optional[str] = None,
    ):
        """수동으로 턴을 마감하고 요약본을 비동기 저장할 때 호출"""
        sid = session_id or self._last_session_id
        with self._state_lock:
            if not sid or sid not in self._active_runs:
                return
            run_info = self._active_runs[sid]
            now = time.time()
            run_info["end_time"] = now
            run_info["updated_at"] = now
            run_info["total_latency_ms"] = int((now - run_info["start_time"]) * 1000)
            run_info["status"] = "SUCCESS"
            run_info["phase"] = "completed"
            run_info["final_response"] = final_response
            run_info["active_model_call"] = None
            run_info["active_tools"].clear()
            run_info["_summary_persisted"] = True

            if token_usage:
                prompt_t = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
                comp_t = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)
                run_info["total_input_tokens"] += prompt_t
                run_info["total_output_tokens"] += comp_t

            audit_log = self._build_turn_summary(run_info)
            self._cleanup_finished_session_locked(sid)

        target_file = self._resolve_target_file(sid, ensure_dirs=self.disk_enabled)
        self._enqueue_if_enabled(target_file, audit_log)

    # =========================================================================
    # Inspection, Summary & Visualizer
    # =========================================================================

    def flush(self):
        """백그라운드 파일 쓰기가 완료될 때까지 동기 대기"""
        _GLOBAL_LOG_WORKER.flush()

    def get_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """세션의 종합 정량 통계 요약 반환"""
        with self._state_lock:
            sid = session_id or self._last_session_id
            if not sid or sid not in self._active_runs:
                return {}
            run_info = self._active_runs[sid]
            tool_calls = run_info.get("total_tool_calls", 0)
            total_latency = run_info.get("total_latency_ms", 0)
            events = list(run_info.get("events", []))
            tool_latencies = [e["latency_ms"] for e in events if e.get("type") == "tool_call"]
            avg_latency = (sum(tool_latencies) / len(tool_latencies)) if tool_latencies else 0
            in_tokens = run_info.get("total_input_tokens", 0)
            out_tokens = run_info.get("total_output_tokens", 0)

            return {
                "session_id": sid,
                "turn_id": run_info.get("turn_id", 1),
                "status": run_info.get("status", "SUCCESS"),
                "total_turns": self._turn_counters.get(sid, 1),
                "total_latency_ms": total_latency,
                "tool_call_count": tool_calls,
                "avg_tool_latency_ms": round(avg_latency, 1),
                "total_input_tokens": in_tokens,
                "total_output_tokens": out_tokens,
                "total_tokens": in_tokens + out_tokens,
            }

    def show_timeline(self, session_id: Optional[str] = None):
        """주피터 내장 다크모드 타임라인 렌더링"""
        with self._state_lock:
            sid = session_id or self._last_session_id
            if not sid or sid not in self._active_runs:
                print("No active run found for timeline rendering.")
                return
            run_snapshot = dict(self._active_runs[sid])
            run_snapshot["events"] = list(self._active_runs[sid].get("events", []))
        html_code = render_timeline_html(run_snapshot)
        display(HTML(html_code))

    def show_tool_summary(self, session_id: Optional[str] = None):
        """주피터 내장 도구 실행 통계 카드 렌더링"""
        with self._state_lock:
            sid = session_id or self._last_session_id
            if not sid or sid not in self._active_runs:
                print("No active run found for tool summary.")
                return
            events = list(self._active_runs[sid].get("events", []))

        tool_stats: Dict[str, Any] = {}
        for ev in events:
            if ev.get("type") == "tool_call":
                t_name = ev.get("tool_name", "unknown")
                t_latency = ev.get("latency_ms", 0)
                t_status = ev.get("status", "SUCCESS")

                if t_name not in tool_stats:
                    tool_stats[t_name] = {"calls": 0, "failures": 0, "total_latency": 0}

                tool_stats[t_name]["calls"] += 1
                tool_stats[t_name]["total_latency"] += t_latency
                if t_status != "SUCCESS":
                    tool_stats[t_name]["failures"] += 1

        html_code = render_tool_summary_html(tool_stats)
        display(HTML(html_code))

    def get_logs(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """수집된 세션의 모든 이벤트 목록 반환"""
        with self._state_lock:
            sid = session_id or self._last_session_id
            if not sid or sid not in self._active_runs:
                return []
            return list(self._active_runs[sid].get("events", []))


# =============================================================================
# 하위 호환 및 용어 표준화를 위한 Aliases
# =============================================================================
LoggingMiddleware = AgentLogTracer

__all__ = ["AgentLogTracer", "LoggingMiddleware"]
