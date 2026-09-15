"""
===============================================================================
[H-04] AgentLogTracer (Unified Async Logging & Observability Middleware)
===============================================================================
감사 로깅(Audit Logging)과 심층 궤적 추적(Tracing & Profiling)을 단일화한 통합 미들웨어:
- 세션/턴(Turn) 수준의 완결된 계층형 상태 관리 및 session_id 식별
- LLM 호출 토큰 사용량(입력/출력/총합) 및 레이턴시 정밀 프로파일링 (wrap_model_call)
- 도구 격발 상세 인터셉트 (wrap_tool_call)
- 백그라운드 스레드 큐 기반 논블로킹 비동기 파일 저장 (_AsyncLogWorker)
- 세션별 분할(.jsonl) 및 단일 통합 파일(log_path) 모드 지원
- IPython/Jupyter 인터랙티브 HTML 시각화 대시보드 (Timeline, Tool Summary)
- AgentTracer, LoggingMiddleware 하위 호환 Alias 완벽 제공
===============================================================================
"""

import os
import time
import json
import queue
import threading
from typing import Any, Dict, List, Optional
from langchain.agents.middleware import AgentMiddleware, ModelResponse
from app.middleware.observability.viz import render_timeline_html, render_tool_summary_html
from app.utils.message_utils import normalize_content, sanitize_text

try:
    from IPython.display import display, HTML
except ImportError:
    def display(x):
        print(x)
    def HTML(x):
        return x


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


class AgentLogTracer(AgentMiddleware):
    """
    에이전트 감사 로깅 및 궤적 관측성 통합 미들웨어 (AgentLogTracer).
    세션 계층 트리, 토큰 사용량, 도구 격발 레이턴시를 비동기로 영속화하며
    대화형 HTML 시각화 대시보드를 제공합니다.
    """
    def __init__(
        self,
        log_dir: str = "./artifacts/logs",
        log_path: Optional[str] = None,
        verbose: bool = False,
        stream_events: bool = True,
    ):
        super().__init__()
        self.log_path = log_path
        self.log_dir = os.path.dirname(log_path) if log_path else log_dir
        self.verbose = verbose
        self.stream_events = stream_events  # 개별 tool_execution 이벤트도 실시간 기록할지 여부
        
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)

        # 세션별 런타임 상태 버퍼: {session_id: {...}}
        self._active_runs: Dict[str, Dict[str, Any]] = {}
        self._last_session_id: Optional[str] = None
        self._turn_counters: Dict[str, int] = {}

    def _resolve_target_file(self, session_id: str) -> str:
        """적재 대상 파일 경로 결정 (단일 파일 지정 시 log_path 우선, 미지정 시 session_id 폴더별 audit.jsonl)"""
        if self.log_path:
            return self.log_path
        session_folder = os.path.join(self.log_dir, session_id)
        os.makedirs(session_folder, exist_ok=True)
        return os.path.join(session_folder, "audit.jsonl")

    def _get_session_id(self, runtime: Any = None, request: Any = None, state: Any = None) -> str:
        """스레드 컨텍스트, 런타임, 요청 객체 전반에서 최우선순위로 session_id(thread_id) 식별"""
        # 0. runtime.context의 session_id 직접 탐색
        if runtime and hasattr(runtime, "context") and getattr(runtime.context, "session_id", None):
            sid = runtime.context.session_id
            if sid and sid != "unknown":
                return str(sid)

        # 1. LangGraph Runtime의 execution_info 탐색 (가장 정확)
        if runtime and hasattr(runtime, "execution_info") and runtime.execution_info:
            tid = getattr(runtime.execution_info, "thread_id", None)
            if tid:
                return str(tid)

        # 2. ToolRequest / ModelRequest의 runtime.config 탐색
        if request and hasattr(request, "runtime") and hasattr(request.runtime, "config") and request.runtime.config:
            tid = request.runtime.config.get("configurable", {}).get("thread_id")
            if tid:
                return str(tid)

        # 3. LangChain RunnableConfig ContextVar 탐색
        try:
            from langchain_core.runnables.config import var_child_runnable_config
            cfg = var_child_runnable_config.get()
            if cfg and isinstance(cfg, dict):
                tid = cfg.get("configurable", {}).get("thread_id")
                if tid:
                    return str(tid)
        except Exception:
            pass

        # 4. AgentRuntime의 config 탐색
        if runtime and hasattr(runtime, "config") and runtime.config:
            tid = runtime.config.get("configurable", {}).get("thread_id")
            if tid:
                return str(tid)

        # 5. State 내부의 configurable 탐색
        if state and isinstance(state, dict):
            cfg = state.get("configurable", {})
            if isinstance(cfg, dict) and cfg.get("thread_id"):
                return str(cfg["thread_id"])

        # 6. 이전 턴 세션 ID 폴백
        if self._last_session_id and self._last_session_id != "unknown":
            return self._last_session_id

        # 7. 기본 고유 세션 ID 자동 발급
        generated_id = f"session_{int(time.time() * 1000)}"
        self._last_session_id = generated_id
        return generated_id

    def _sync_session_keys(self, current_session_id: str):
        """임시 생성된 세션 버퍼가 있을 경우 실제 session_id로 상태 마이그레이션"""
        if current_session_id and current_session_id != "unknown":
            if self._last_session_id and self._last_session_id != current_session_id:
                if self._last_session_id in self._active_runs:
                    self._active_runs[current_session_id] = self._active_runs.pop(self._last_session_id)
                    self._active_runs[current_session_id]["session_id"] = current_session_id
            self._last_session_id = current_session_id

    # =========================================================================
    # Agent Lifecycle Hooks
    # =========================================================================

    def before_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        """에이전트 전체 실행 시작 시 세션 상태 초기화"""
        session_id = self._get_session_id(runtime=runtime, state=state)
        self._sync_session_keys(session_id)

        messages = state.get("messages", [])
        user_query = normalize_content(messages[-1].content) if messages else "unknown"

        # 세션별 턴 번호 증가
        self._turn_counters[session_id] = self._turn_counters.get(session_id, 0) + 1
        turn_id = self._turn_counters[session_id]

        start_time = time.time()
        self._active_runs[session_id] = {
            "session_id": session_id,
            "turn_id": turn_id,
            "user_query": user_query,
            "start_time": start_time,
            "events": [],
            "status": "RUNNING",
            "final_response": "",
            "total_latency_ms": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tool_calls": 0,
        }

        if runtime and getattr(runtime, "context", None) is not None:
            runtime.context.start_time = start_time
            runtime.context.user_query = user_query
            runtime.context.session_id = session_id
            runtime.context.turn_id = turn_id

        if self.verbose:
            print(f"\n🪵 [AgentLogTracer] === 에이전트 실행 시작 (Turn #{turn_id}) ===")
            print(f"📥 사용자 질문: {user_query}")
            print(f"🆔 세션 ID: {session_id}")

        return None

    def after_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        """에이전트 전체 실행 완료 시 계층적 턴 감사 로그(turn_summary)를 비동기로 영속화"""
        session_id = self._get_session_id(runtime=runtime, state=state)
        self._sync_session_keys(session_id)

        run_info = self._active_runs.get(session_id)
        if not run_info:
            run_info = {
                "session_id": session_id,
                "turn_id": self._turn_counters.get(session_id, 1),
                "user_query": "unknown",
                "start_time": time.time(),
                "events": [],
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "total_tool_calls": 0,
            }
            self._active_runs[session_id] = run_info

        duration_ms = int((time.time() - run_info["start_time"]) * 1000)
        run_info["total_latency_ms"] = duration_ms
        run_info["end_time"] = time.time()

        messages = state.get("messages", [])
        final_resp = ""
        dialogue_history = []

        if messages:
            final_resp = normalize_content(messages[-1].content)
            run_info["final_response"] = final_resp
            run_info["status"] = "SUCCESS"
            for msg in messages:
                dialogue_history.append({
                    "role": getattr(msg, "type", "unknown"),
                    "content": sanitize_text(normalize_content(msg.content))
                })
        else:
            run_info["status"] = "FAILED"

        # 세션/턴 수준 완결 감사 로그 구조체 (Hierarchical Turn Summary)
        audit_log = {
            "event": "turn_summary",
            "session_id": session_id,
            "turn_id": run_info.get("turn_id", 1),
            "status": run_info["status"],
            "user_query": run_info["user_query"],
            "final_response": final_resp,
            "total_latency_ms": duration_ms,
            "token_usage": {
                "prompt_tokens": run_info["total_input_tokens"],
                "completion_tokens": run_info["total_output_tokens"],
                "total_tokens": run_info["total_input_tokens"] + run_info["total_output_tokens"]
            },
            "total_tool_calls": run_info["total_tool_calls"],
            "steps": run_info["events"],
            "dialogue_history": dialogue_history,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

        target_file = self._resolve_target_file(session_id)
        _GLOBAL_LOG_WORKER.enqueue(target_file, audit_log)

        if self.verbose:
            print(f"📤 에이전트 실행 완료 (소요: {duration_ms}ms, 상태: {run_info['status']})")

        return None

    def wrap_model_call(self, request: Any, handler: Any) -> ModelResponse:
        """LLM 호출 토큰 사용량과 레이턴시 정밀 프로파일링"""
        session_id = self._get_session_id(request=request)
        self._sync_session_keys(session_id)

        start_time = time.time()
        msg_count = len(request.messages) if hasattr(request, "messages") else 0

        response = handler(request)

        duration_ms = int((time.time() - start_time) * 1000)
        ai_msg = response.result[0] if getattr(response, "result", None) else None
        response_text = ai_msg.content if ai_msg else ""

        usage = getattr(ai_msg, "usage_metadata", {}) or {}
        if not usage and ai_msg and hasattr(ai_msg, "response_metadata"):
            usage = ai_msg.response_metadata.get("token_usage", {})

        input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0

        event = {
            "type": "model_call",
            "timestamp": time.time(),
            "input_messages_count": msg_count,
            "response_snippet": sanitize_text(normalize_content(response_text))[:200],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": duration_ms
        }

        if session_id in self._active_runs:
            self._active_runs[session_id]["events"].append(event)
            self._active_runs[session_id]["total_input_tokens"] += input_tokens
            self._active_runs[session_id]["total_output_tokens"] += output_tokens

        if self.verbose:
            print(f"🧠 [AgentLogTracer] LLM 응답 완료 (Prompt: {input_tokens}, Completion: {output_tokens}, 소요: {duration_ms}ms)")

        return response

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        """도구 격발 상세 추적 및 실시간 이벤트 스트리밍"""
        session_id = self._get_session_id(request=request)
        self._sync_session_keys(session_id)

        tool_name = request.tool_call.get("name", "unknown_tool")
        tool_args = request.tool_call.get("args", {})
        start_time = time.time()

        if self.verbose:
            print(f"🔧 [AgentLogTracer] ➡️ 도구 격발 시작: {tool_name}({tool_args})")

        response = handler(request)

        duration_ms = int((time.time() - start_time) * 1000)
        result_str = sanitize_text(str(response))

        if self.verbose:
            print(f"🔧 [AgentLogTracer] ⬅️ 도구 격발 완료: {tool_name} (소요: {duration_ms}ms)")

        event = {
            "type": "tool_call",
            "timestamp": time.time(),
            "tool_name": tool_name,
            "arguments": tool_args,
            "result_snippet": result_str[:250],
            "latency_ms": duration_ms,
            "status": "SUCCESS"
        }

        if session_id in self._active_runs:
            self._active_runs[session_id]["events"].append(event)
            self._active_runs[session_id]["total_tool_calls"] += 1

        if self.stream_events:
            tool_log = {
                "event": "tool_execution",
                "session_id": session_id,
                "tool_name": tool_name,
                "arguments": tool_args,
                "result": result_str,
                "latency_ms": duration_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            target_file = self._resolve_target_file(session_id)
            _GLOBAL_LOG_WORKER.enqueue(target_file, tool_log)

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

        start_time = time.time()
        msg_count = len(request.messages) if hasattr(request, "messages") else 0

        response = await handler(request)

        duration_ms = int((time.time() - start_time) * 1000)
        ai_msg = response.result[0] if getattr(response, "result", None) else None
        response_text = ai_msg.content if ai_msg else ""

        usage = getattr(ai_msg, "usage_metadata", {}) or {}
        if not usage and ai_msg and hasattr(ai_msg, "response_metadata"):
            usage = ai_msg.response_metadata.get("token_usage", {})

        input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0

        event = {
            "type": "model_call",
            "timestamp": time.time(),
            "input_messages_count": msg_count,
            "response_snippet": sanitize_text(normalize_content(response_text))[:200],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": duration_ms
        }

        if session_id in self._active_runs:
            self._active_runs[session_id]["events"].append(event)
            self._active_runs[session_id]["total_input_tokens"] += input_tokens
            self._active_runs[session_id]["total_output_tokens"] += output_tokens

        return response

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        session_id = self._get_session_id(request=request)
        self._sync_session_keys(session_id)

        tool_name = request.tool_call.get("name", "unknown_tool")
        tool_args = request.tool_call.get("args", {})
        start_time = time.time()

        response = await handler(request)

        duration_ms = int((time.time() - start_time) * 1000)
        result_str = sanitize_text(str(response))

        event = {
            "type": "tool_call",
            "timestamp": time.time(),
            "tool_name": tool_name,
            "arguments": tool_args,
            "result_snippet": result_str[:250],
            "latency_ms": duration_ms,
            "status": "SUCCESS"
        }

        if session_id in self._active_runs:
            self._active_runs[session_id]["events"].append(event)
            self._active_runs[session_id]["total_tool_calls"] += 1

        if self.stream_events:
            tool_log = {
                "event": "tool_execution",
                "session_id": session_id,
                "tool_name": tool_name,
                "arguments": tool_args,
                "result": result_str,
                "latency_ms": duration_ms,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            target_file = self._resolve_target_file(session_id)
            _GLOBAL_LOG_WORKER.enqueue(target_file, tool_log)

        return response

    # =========================================================================
    # Standalone Manual API (단위 테스트 및 스크립트 모드 지원)
    # =========================================================================

    def start_turn(self, session_id: str, user_query: str = ""):
        """수동으로 세션 턴을 개시할 때 호출"""
        self._last_session_id = session_id
        self._turn_counters[session_id] = self._turn_counters.get(session_id, 0) + 1
        self._active_runs[session_id] = {
            "session_id": session_id,
            "turn_id": self._turn_counters[session_id],
            "user_query": user_query,
            "start_time": time.time(),
            "events": [],
            "status": "RUNNING",
            "final_response": "",
            "total_latency_ms": 0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_tool_calls": 0,
        }

    def record_tool_call(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        duration_ms: float = 0.0,
        success: bool = True,
        session_id: Optional[str] = None
    ):
        """수동으로 도구 호출 이벤트를 기록할 때 호출"""
        sid = session_id or self._last_session_id
        if not sid or sid not in self._active_runs:
            self.start_turn(sid or "default_session")
            sid = self._last_session_id

        event = {
            "type": "tool_call",
            "timestamp": time.time(),
            "tool_name": tool_name,
            "arguments": arguments,
            "result_snippet": "success" if success else "failed",
            "latency_ms": int(duration_ms),
            "status": "SUCCESS" if success else "FAILED"
        }
        self._active_runs[sid]["events"].append(event)
        self._active_runs[sid]["total_tool_calls"] += 1

    def end_turn(
        self,
        token_usage: Optional[Dict[str, int]] = None,
        final_response: str = "",
        session_id: Optional[str] = None
    ):
        """수동으로 턴을 마감하고 요약본을 비동기 저장할 때 호출"""
        sid = session_id or self._last_session_id
        if not sid or sid not in self._active_runs:
            return
        run_info = self._active_runs[sid]
        run_info["end_time"] = time.time()
        run_info["total_latency_ms"] = int((time.time() - run_info["start_time"]) * 1000)
        run_info["status"] = "SUCCESS"
        run_info["final_response"] = final_response

        if token_usage:
            prompt_t = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
            comp_t = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)
            run_info["total_input_tokens"] += prompt_t
            run_info["total_output_tokens"] += comp_t

        audit_log = {
            "event": "turn_summary",
            "session_id": sid,
            "turn_id": run_info.get("turn_id", 1),
            "status": run_info["status"],
            "user_query": run_info["user_query"],
            "final_response": final_response,
            "total_latency_ms": run_info["total_latency_ms"],
            "token_usage": {
                "prompt_tokens": run_info["total_input_tokens"],
                "completion_tokens": run_info["total_output_tokens"],
                "total_tokens": run_info["total_input_tokens"] + run_info["total_output_tokens"]
            },
            "total_tool_calls": run_info["total_tool_calls"],
            "steps": run_info["events"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        target_file = self._resolve_target_file(sid)
        _GLOBAL_LOG_WORKER.enqueue(target_file, audit_log)

    # =========================================================================
    # Inspection, Summary & Visualizer
    # =========================================================================

    def flush(self):
        """백그라운드 파일 쓰기가 완료될 때까지 동기 대기"""
        _GLOBAL_LOG_WORKER.flush()

    def get_summary(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """세션의 종합 정량 통계 요약 반환"""
        sid = session_id or self._last_session_id
        if not sid or sid not in self._active_runs:
            return {}
        run_info = self._active_runs[sid]
        tool_calls = run_info.get("total_tool_calls", 0)
        total_latency = run_info.get("total_latency_ms", 0)
        events = run_info.get("events", [])
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
        sid = session_id or self._last_session_id
        if not sid or sid not in self._active_runs:
            print("No active run found for timeline rendering.")
            return
        run_info = self._active_runs[sid]
        html_code = render_timeline_html(run_info)
        display(HTML(html_code))

    def show_tool_summary(self, session_id: Optional[str] = None):
        """주피터 내장 도구 실행 통계 카드 렌더링"""
        sid = session_id or self._last_session_id
        if not sid or sid not in self._active_runs:
            print("No active run found for tool summary.")
            return
        run_info = self._active_runs[sid]

        tool_stats: Dict[str, Any] = {}
        for ev in run_info.get("events", []):
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
        sid = session_id or self._last_session_id
        if not sid or sid not in self._active_runs:
            return []
        return self._active_runs[sid].get("events", [])


# =============================================================================
# 하위 호환 및 용어 표준화를 위한 Aliases
# =============================================================================
AgentTracer = AgentLogTracer
LoggingMiddleware = AgentLogTracer

__all__ = ["AgentLogTracer", "AgentTracer", "LoggingMiddleware"]
