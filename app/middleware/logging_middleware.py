import os
import time
import json
from typing import Any, Dict
from langchain.agents.middleware import AgentMiddleware
from app.utils.message_utils import normalize_content, sanitize_text

class LoggingMiddleware(AgentMiddleware):
    def __init__(self, log_dir="./artifacts/logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        # Store run-specific details indexed by the unique id of the runtime object
        self._active_runs = {}

    def _get_session_id(self, runtime, request=None) -> str:
        """스레드 및 컨텍스트에 따라 최우선의 session_id(thread_id)를 식별합니다."""
        # 1. contextvar 기반의 active config` 탐색
        try:
            from langchain_core.runnables.config import get_config_from_context
            config = get_config_from_context()
            if config:
                tid = config.get("configurable", {}).get("thread_id")
                if tid:
                    return tid
        except Exception:
            pass

        # 2. ToolRequest의 request.runtime.config 탐색
        if request and hasattr(request, "runtime") and hasattr(request.runtime, "config") and request.runtime.config:
            tid = request.runtime.config.get("configurable", {}).get("thread_id")
            if tid:
                return tid

        # 3. AgentRuntime의 runtime.config 탐색
        if runtime and hasattr(runtime, "config") and runtime.config:
            tid = runtime.config.get("configurable", {}).get("thread_id")
            if tid:
                return tid

        # 4. runtime.context 내에 이미 백업된 session_id 탐색
        if runtime and hasattr(runtime, "context"):
            tid = getattr(runtime.context, "session_id", None)
            if tid:
                return tid

        return "unknown"

    def before_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        """에이전트 전체 실행의 시작 로깅"""
        logging_enabled = getattr(runtime.context, "logging_enabled", False) if runtime and runtime.context else False
        if not logging_enabled:
            return None

        start_time = time.time()
        user_query = normalize_content(state.get("messages", [])[-1].content) if state.get("messages") else "unknown"
        session_id = self._get_session_id(runtime)
        
        # 런타임의 context 객체에 실행 컨텍스트 정보를 동적으로 바인딩하여 
        # id(runtime) 격리 실패 및 멀티스레드 충돌 문제를 원천 예방합니다.
        if runtime and runtime.context:
            runtime.context.start_time = start_time
            runtime.context.user_query = user_query
            runtime.context.session_id = session_id

        print(f"\n🪵 [LoggingMiddleware] === 에이전트 실행 시작 ===")
        print(f"📥 사용자 질문: {user_query}")
        return None

    def after_agent(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any] | None:
        """에이전트 전체 실행의 완료 및 감사 로그 생성"""
        logging_enabled = getattr(runtime.context, "logging_enabled", False) if runtime and runtime.context else False
        if not logging_enabled:
            return None

        start_time = getattr(runtime.context, "start_time", None) if runtime and runtime.context else None
        user_query = getattr(runtime.context, "user_query", "unknown") if runtime and runtime.context else "unknown"
        session_id = self._get_session_id(runtime)
        
        duration_ms = int((time.time() - start_time) * 1000) if start_time else 0
        print(f"📤 에이전트 실행 완료 (소요: {duration_ms}ms)")

        # 에이전트 최종 답변 및 전체 대화 궤적 추출
        agent_response = ""
        dialogue_history = []
        messages = state.get("messages", [])
        if messages:
            agent_response = normalize_content(messages[-1].content)
            for msg in messages:
                dialogue_history.append({
                    "role": msg.type,  # 'human', 'ai', 'tool' 등
                    "content": sanitize_text(normalize_content(msg.content))
                })

        # 에이전트 감사 로그 생성 및 파일 적재
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
        self._append_log(session_id, audit_log)
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
        
        # 도구 실행 수행
        response = handler(request)
        
        duration_ms = int((time.time() - start_time) * 1000)
        print(f"🔧 [LoggingMiddleware] ⬅️ 도구 격발 완료: {tool_name} (소요: {duration_ms}ms)")
        
        session_id = self._get_session_id(request.runtime, request=request)
        if request.runtime and request.runtime.context:
            request.runtime.context.session_id = session_id
            
        tool_log = {
            "event": "tool_execution",
            "session_id": session_id,
            "tool_name": tool_name,
            "arguments": tool_args,
            "result": str(response),
            "latency_ms": duration_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        self._append_log(session_id, tool_log)
        return response

    async def awrap_tool_call(self, request, handler):
        """개별 도구(Tool Call) 격발의 시작과 완료를 감싸서 비동기 로깅"""
        logging_enabled = getattr(request.runtime.context, "logging_enabled", False) if request.runtime and request.runtime.context else False
        if not logging_enabled:
            return await handler(request)

        tool_name = request.tool_call.get("name", "unknown_tool")
        tool_args = request.tool_call.get("args", {})
        start_time = time.time()
        
        print(f"🔧 [LoggingMiddleware] ➡️ 도구 격발 시작: {tool_name}({tool_args})")
        
        # 도구 실행 수행
        response = await handler(request)
        
        duration_ms = int((time.time() - start_time) * 1000)
        print(f"🔧 [LoggingMiddleware] ⬅️ 도구 격발 완료: {tool_name} (소요: {duration_ms}ms)")
        
        session_id = self._get_session_id(request.runtime, request=request)
        if request.runtime and request.runtime.context:
            request.runtime.context.session_id = session_id
            
        tool_log = {
            "event": "tool_execution",
            "session_id": session_id,
            "tool_name": tool_name,
            "arguments": tool_args,
            "result": str(response),
            "latency_ms": duration_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        self._append_log(session_id, tool_log)
        return response

    def _append_log(self, session_id, log_data):
        """감사 로그 지정 경로에 세션별 JSON라인 파일로 적재"""
        log_file = os.path.join(self.log_dir, f"{session_id}.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
