import os
import time
import json
from typing import Any, Dict, List, Callable
from langchain.agents.middleware import AgentMiddleware, ModelResponse
from modules.common.viz import render_timeline_html, render_tool_summary_html

try:
    from IPython.display import display, HTML
except ImportError:
    # Fallback if run outside Jupyter
    def display(x):
        print(x)
    def HTML(x):
        return x

def normalize_content(content: Any) -> str:
    """Normalize input message content to plain string."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)

class AgentTracer(AgentMiddleware):
    """
    Enhanced logging middleware for LangGraph / LangChain agents.
    Tracks step-level executions, LLM token usages, tool details, and latencies.
    Generates JSONL logs and supports IPython display of executions.
    """
    def __init__(self, log_dir="./artifacts/logs", verbose=False):
        self.log_dir = log_dir
        self.verbose = verbose
        os.makedirs(self.log_dir, exist_ok=True)
        # Session state storage
        self._active_runs = {}
        self._last_session_id = None

    def _get_session_id(self, runtime, request=None) -> str:
        """Identify the priority session_id (thread_id) based on thread and context."""
        # 1. From get_config_from_context
        try:
            from langchain_core.runnables.config import get_config_from_context
            config = get_config_from_context()
            if config:
                tid = config.get("configurable", {}).get("thread_id")
                if tid:
                    return str(tid)
        except Exception:
            pass

        # 2. ToolRequest config
        if request and hasattr(request, "runtime") and hasattr(request.runtime, "config") and request.runtime.config:
            tid = request.runtime.config.get("configurable", {}).get("thread_id")
            if tid:
                return str(tid)

        # 3. AgentRuntime config
        if runtime and hasattr(runtime, "config") and runtime.config:
            tid = runtime.config.get("configurable", {}).get("thread_id")
            if tid:
                return str(tid)

        # 4. runtime context backup
        if runtime and getattr(runtime, "context", None) is not None:
            tid = getattr(runtime.context, "session_id", None)
            if tid:
                return str(tid)

        return "unknown"


    def before_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        """Log agent run start and initialize logging state."""
        session_id = self._get_session_id(runtime)
        self._last_session_id = session_id
        
        messages = state.get("messages", [])
        user_query = normalize_content(messages[-1].content) if messages else "unknown"
        
        # Inject dynamic variables to runtime context
        if runtime and getattr(runtime, "context", None) is not None:
            runtime.context.start_time = time.time()
            runtime.context.user_query = user_query
            runtime.context.session_id = session_id

        # Setup runtime session data
        self._active_runs[session_id] = {
            "session_id": session_id,
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

        if self.verbose:
            print(f"\n🪵 [AgentTracer] === Agent Execution Started ===")
            print(f"📥 Query: {user_query}")
            print(f"🆔 Session: {session_id}")

        return None

    def after_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        """Finalize agent execution, serialize session info to JSONL."""
        session_id = self._get_session_id(runtime)
        if session_id == "unknown" and session_id not in self._active_runs and self._last_session_id:
            session_id = self._last_session_id
            
        run_info = self._active_runs.get(session_id)
        if not run_info:
            return None

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
                    "role": msg.type,
                    "content": normalize_content(msg.content)
                })
        else:
            run_info["status"] = "FAILED"

        # Log structure for JSONL output
        audit_log = {
            "event": "agent_execution",
            "session_id": session_id,
            "query": run_info["user_query"],
            "response": final_resp,
            "dialogue_history": dialogue_history,
            "latency_ms": duration_ms,
            "status": run_info["status"],
            "total_input_tokens": run_info["total_input_tokens"],
            "total_output_tokens": run_info["total_output_tokens"],
            "total_tool_calls": run_info["total_tool_calls"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "steps": run_info["events"]
        }
        
        self._append_log(session_id, audit_log)
        
        if self.verbose:
            print(f"📤 Agent Completed in {duration_ms}ms (Status: {run_info['status']})")
            
        return None

    def wrap_model_call(self, request, handler) -> ModelResponse:
        """Trace LLM call, record input/output tokens and latency."""
        session_id = self._get_session_id(request.runtime)
        
        # Dynamically map and restore session id if initialized as unknown
        if session_id != "unknown" and "unknown" in self._active_runs:
            self._active_runs[session_id] = self._active_runs.pop("unknown")
            self._active_runs[session_id]["session_id"] = session_id
            self._last_session_id = session_id
            
        start_time = time.time()
        
        # Calculate messages count
        msg_count = len(request.messages) if hasattr(request, "messages") else 0
        
        # Execute actual model call
        response = handler(request)
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        ai_msg = response.result[0] if response.result else None
        response_text = ai_msg.content if ai_msg else ""
        
        # Gather token count metadata
        usage = getattr(ai_msg, "usage_metadata", {}) or {}
        if not usage and ai_msg and hasattr(ai_msg, "response_metadata"):
            usage = ai_msg.response_metadata.get("token_usage", {})
            
        input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0
        
        event = {
            "type": "model_call",
            "timestamp": time.time(),
            "input_messages_count": msg_count,
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": duration_ms
        }
        
        if session_id in self._active_runs:
            self._active_runs[session_id]["events"].append(event)
            self._active_runs[session_id]["total_input_tokens"] += input_tokens
            self._active_runs[session_id]["total_output_tokens"] += output_tokens
            
        if self.verbose:
            print(f"🧠 [AgentTracer] LLM response ({input_tokens} prompt / {output_tokens} completion tokens, took {duration_ms}ms)")
            
        return response

    def wrap_tool_call(self, request, handler):
        """Intercept and log individual tool execution (synchronous)."""
        session_id = self._get_session_id(request.runtime, request=request)
        
        # Dynamically map and restore session id if initialized as unknown
        if session_id != "unknown" and "unknown" in self._active_runs:
            self._active_runs[session_id] = self._active_runs.pop("unknown")
            self._active_runs[session_id]["session_id"] = session_id
            self._last_session_id = session_id
            
        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})
        start_time = time.time()
        
        if self.verbose:
            print(f"🔧 [AgentTracer] Tool Executing: {tool_name} with args: {tool_args}")
            
        # Execute tool
        response = handler(request)
        
        duration_ms = int((time.time() - start_time) * 1000)
        result_str = str(response)
        
        event = {
            "type": "tool_call",
            "timestamp": time.time(),
            "tool_name": tool_name,
            "arguments": tool_args,
            "result": result_str,
            "latency_ms": duration_ms,
            "status": "SUCCESS"  # If handler doesn't throw, assume success
        }
        
        if session_id in self._active_runs:
            self._active_runs[session_id]["events"].append(event)
            self._active_runs[session_id]["total_tool_calls"] += 1
            
        if self.verbose:
            print(f"🔧 [AgentTracer] Tool Executed in {duration_ms}ms")
            
        return response

    # --- Async Middleware Wrappers ---
    async def abefore_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        return self.before_agent(state, runtime)

    async def aafter_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        return self.after_agent(state, runtime)

    async def awrap_model_call(self, request, handler) -> ModelResponse:
        session_id = self._get_session_id(request.runtime)
        
        # Dynamically map and restore session id if initialized as unknown
        if session_id != "unknown" and "unknown" in self._active_runs:
            self._active_runs[session_id] = self._active_runs.pop("unknown")
            self._active_runs[session_id]["session_id"] = session_id
            self._last_session_id = session_id
            
        start_time = time.time()
        msg_count = len(request.messages) if hasattr(request, "messages") else 0
        
        response = await handler(request)
        
        duration_ms = int((time.time() - start_time) * 1000)
        ai_msg = response.result[0] if response.result else None
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
            "response_text": response_text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": duration_ms
        }
        
        if session_id in self._active_runs:
            self._active_runs[session_id]["events"].append(event)
            self._active_runs[session_id]["total_input_tokens"] += input_tokens
            self._active_runs[session_id]["total_output_tokens"] += output_tokens
            
        if self.verbose:
            print(f"🧠 [AgentTracer (Async)] LLM response ({input_tokens} prompt / {output_tokens} completion tokens, took {duration_ms}ms)")
            
        return response

    async def awrap_tool_call(self, request, handler):
        session_id = self._get_session_id(request.runtime, request=request)
        
        # Dynamically map and restore session id if initialized as unknown
        if session_id != "unknown" and "unknown" in self._active_runs:
            self._active_runs[session_id] = self._active_runs.pop("unknown")
            self._active_runs[session_id]["session_id"] = session_id
            self._last_session_id = session_id
            
        tool_name = request.tool_call.get("name", "unknown")
        tool_args = request.tool_call.get("args", {})
        start_time = time.time()
        
        if self.verbose:
            print(f"🔧 [AgentTracer (Async)] Tool Executing: {tool_name} with args: {tool_args}")
            
        response = await handler(request)
        
        duration_ms = int((time.time() - start_time) * 1000)
        result_str = str(response)
        
        event = {
            "type": "tool_call",
            "timestamp": time.time(),
            "tool_name": tool_name,
            "arguments": tool_args,
            "result": result_str,
            "latency_ms": duration_ms,
            "status": "SUCCESS"
        }
        
        if session_id in self._active_runs:
            self._active_runs[session_id]["events"].append(event)
            self._active_runs[session_id]["total_tool_calls"] += 1
            
        if self.verbose:
            print(f"🔧 [AgentTracer (Async)] Tool Executed in {duration_ms}ms")
            
        return response

    # --- Inspection & Visualization ---
    def show_timeline(self, session_id=None):
        """Render session timeline inside Jupyter."""
        sid = session_id or self._last_session_id
        if not sid or sid not in self._active_runs:
            print("No active run found for timeline rendering.")
            return
        run_info = self._active_runs[sid]
        html_code = render_timeline_html(run_info)
        display(HTML(html_code))

    def show_tool_summary(self, session_id=None):
        """Render tool execution stats inside Jupyter."""
        sid = session_id or self._last_session_id
        if not sid or sid not in self._active_runs:
            print("No active run found for tool summary.")
            return
        run_info = self._active_runs[sid]
        
        # Aggregate tool statistics
        tool_stats = {}
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

    def get_logs(self, session_id=None) -> List[Dict[str, Any]]:
        """Return the collected events for a session."""
        sid = session_id or self._last_session_id
        if not sid or sid not in self._active_runs:
            return []
        return self._active_runs[sid].get("events", [])

    def _append_log(self, session_id: str, log_data: Dict[str, Any]):
        """Persist session logs in JSONL format."""
        log_file = os.path.join(self.log_dir, f"{session_id}.jsonl")
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Error appending log: {e}")
