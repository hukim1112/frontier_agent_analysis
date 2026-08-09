import os
import time
import json
from typing import Any, Dict
from langchain.agents.middleware import AgentMiddleware

class ReferenceLoggingMiddleware(AgentMiddleware):
    def __init__(self, log_path="./artifacts/agent_audit_trail.json"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        # Store run-specific details indexed by the unique id of the runtime object
        self._active_runs = {}

    def before_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        """에이전트 전체 실행의 시작 로깅"""
        logging_enabled = getattr(runtime.context, "logging_enabled", False) if runtime and runtime.context else False
        if not logging_enabled:
            return None

        start_time = time.time()
        user_query = state.get("messages", [])[-1].content if state.get("messages") else "unknown"
        
        self._active_runs[id(runtime)] = {
            "start_time": start_time,
            "query": user_query
        }

        print(f"\n🪵 [LoggingMiddleware] === 에이전트 실행 시작 ===")
        print(f"📥 사용자 질문: {user_query}")
        return None

    def after_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        """에이전트 전체 실행의 완료 및 감사 로그 생성"""
        logging_enabled = getattr(runtime.context, "logging_enabled", False) if runtime and runtime.context else False
        if not logging_enabled:
            return None

        run_data = self._active_runs.pop(id(runtime), {})
        start_time = run_data.get("start_time")
        duration_ms = int((time.time() - start_time) * 1000) if start_time else 0
        print(f"📤 에이전트 실행 완료 (소요: {duration_ms}ms)")

        user_query = run_data.get("query", "unknown")
        
        agent_response = ""
        dialogue_history = []
        messages = state.get("messages", [])
        if messages:
            agent_response = messages[-1].content
            for msg in messages:
                dialogue_history.append({
                    "role": msg.type,
                    "content": str(msg.content)
                })

        session_id = "unknown"
        if runtime and hasattr(runtime, "config") and runtime.config:
            session_id = runtime.config.get("configurable", {}).get("thread_id", "unknown")

        audit_log = {
            "event": "agent_execution",
            "session_id": session_id,
            "query": user_query,
            "response": agent_response,
            "dialogue_history": dialogue_history,
            "latency_ms": duration_ms,
            "status": "SUCCESS" if messages else "FAILED",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        self._append_log(audit_log)
        return None

    def wrap_tool_call(self, request, handler):
        """개별 도구(Tool Call) 격발의 시작과 완료를 감싸서 로깅"""
        logging_enabled = getattr(request.runtime.context, "logging_enabled", False) if request.runtime and request.runtime.context else False
        if not logging_enabled:
            return handler(request)

        tool_name = request.tool_call.get("name", "unknown_tool")
        tool_args = request.tool_call.get("args", {})
        start_time = time.time()
        
        print(f"🔧 [LoggingMiddleware] ➡️ 도구 격발 시작: {tool_name}({tool_args})")
        
        response = handler(request)
        
        duration_ms = int((time.time() - start_time) * 1000)
        print(f"🔧 [LoggingMiddleware] ⬅️ 도구 격발 완료: {tool_name} (소요: {duration_ms}ms)")
        
        session_id = "unknown"
        if hasattr(request, "runtime") and hasattr(request.runtime, "config"):
            session_id = request.runtime.config.get("configurable", {}).get("thread_id", "unknown")
            
        tool_log = {
            "event": "tool_execution",
            "session_id": session_id,
            "tool_name": tool_name,
            "arguments": tool_args,
            "latency_ms": duration_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        self._append_log(tool_log)
        return response

    def _append_log(self, log_data):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
