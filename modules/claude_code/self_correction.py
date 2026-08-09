"""
Claude Code Error Correction & State Transition Harness Module

Implements 5 core transition & error correction handlers:
1. stop_hook_blocking: Code compilation/syntax failure -> inject blockingError for self-repair
2. model_error: Transient API/network error handling & exponential retry fallback
3. model_fallback: Primary model failure -> dynamic fallback model routing
4. aborted_streaming: User streaming abort (Ctrl+C / Abort signal) handling
5. aborted_tools: Async tool execution interrupt signal handling
"""

import re
import time
from typing import Callable, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.agents.middleware import AgentMiddleware, ModelResponse


def extract_code_blocks(text: str) -> list[str]:
    """Helper to extract all python code blocks from text."""
    pattern = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL)
    return pattern.findall(text)


# =============================================================================
# 1. stop_hook_blocking (Self-Correction on Code Compilation / Syntax Error)
# =============================================================================
class StopHooksMiddleware(AgentMiddleware):
    """Intercepts agent response, validates python code blocks, and injects blockingError for self-repair."""
    def __init__(self, validators: Optional[list[dict[str, Any]]] = None):
        if validators is None:
            # Default built-in python syntax & indentation validator
            def check_python_syntax(code_str: str) -> Optional[str]:
                try:
                    compile(code_str, "<string>", "exec")
                    return None
                except SyntaxError as e:
                    return f"SyntaxError at line {e.lineno}: {e.msg} -> '{e.text.strip() if e.text else ''}'"
                except Exception as e:
                    return f"Code validation error: {e}"

            self.validators = [{"name": "python_syntax", "fn": check_python_syntax}]
        else:
            self.validators = validators

    def after_agent(self, state: dict, runtime=None) -> dict | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.content:
            return None

        from app.utils.message_utils import normalize_content
        content_str = normalize_content(last_msg.content)
        code_blocks = extract_code_blocks(content_str)
        if not code_blocks:
            return None

        validation_errors = []
        for code in code_blocks:
            for val in self.validators:
                val_name = val.get("name", "unknown")
                val_fn = val.get("fn")
                if val_fn:
                    try:
                        error = val_fn(code)
                        if error:
                            validation_errors.append(f"[{val_name}]: {error}")
                    except Exception as e:
                        validation_errors.append(f"[{val_name}] execution error: {e}")

        if validation_errors:
            error_details = "\n".join(validation_errors)
            correction_prompt = (
                f"🛑 [Stop Hook Blocking - Code Validation Failed]\n"
                f"The code you generated failed quality validation tests:\n"
                f"{error_details}\n\n"
                f"Please analyze the error details above, fix the bug in your code, and output the corrected version."
            )
            print(f"\n🔴 [stop_hook_blocking] Validation Failed -> Injecting blockingError for Self-Repair!")
            updated_messages = list(messages) + [HumanMessage(content=correction_prompt)]
            return {"messages": updated_messages, "transition": "stop_hook_blocking"}

        return {"transition": "completed"}


# =============================================================================
# 2. model_error (Transient Network Exception & Retry Fallback)
# =============================================================================
class ModelErrorHandlerMiddleware(AgentMiddleware):
    """Catches API connection / 500 transient errors and performs retry with exponential backoff."""
    def __init__(self, max_retries: int = 3, initial_delay: float = 0.5):
        self.max_retries = max_retries
        self.initial_delay = initial_delay

    def wrap_model_call(self, request, handler):
        for attempt in range(1, self.max_retries + 1):
            try:
                return handler(request)
            except Exception as error:
                print(f"⚠️ [model_error] Model call attempt {attempt}/{self.max_retries} failed: {error}")
                if attempt == self.max_retries:
                    print(f"🛑 [model_error] All retries exhausted. Gracefully handling model_error transition.")
                    fallback_msg = AIMessage(
                        content=f"🔌 [model_error Transition] LLM API 통신 장애가 발생했습니다 ({error}). 네트워크 및 API 키를 점검해 주세요."
                    )
                    return ModelResponse(result=[fallback_msg])
                sleep_time = self.initial_delay * (2 ** (attempt - 1))
                time.sleep(sleep_time)

    async def awrap_model_call(self, request, handler):
        import asyncio
        for attempt in range(1, self.max_retries + 1):
            try:
                return await handler(request)
            except Exception as error:
                print(f"⚠️ [model_error] Model call attempt {attempt}/{self.max_retries} failed: {error}")
                if attempt == self.max_retries:
                    print(f"🛑 [model_error] All retries exhausted. Gracefully handling model_error transition.")
                    fallback_msg = AIMessage(
                        content=f"🔌 [model_error Transition] LLM API 통신 장애가 발생했습니다 ({error}). 네트워크 및 API 키를 점검해 주세요."
                    )
                    return ModelResponse(result=[fallback_msg])
                sleep_time = self.initial_delay * (2 ** (attempt - 1))
                await asyncio.sleep(sleep_time)


# =============================================================================
# 3. model_fallback (Dynamic Primary -> Secondary Model Routing on Failure)
# =============================================================================
class ModelFallbackMiddleware(AgentMiddleware):
    """Catches exceptions from primary model call and routes request to a backup fallback model."""
    def __init__(self, fallback_model_name: str = "gemini-2.5-pro", fallback_llm: Optional[Any] = None):
        self.fallback_model_name = fallback_model_name
        self.fallback_llm = fallback_llm

    def _get_fallback_llm(self):
        if self.fallback_llm is not None:
            return self.fallback_llm
        from app.utils.llm import get_llm
        return get_llm(model_name=self.fallback_model_name, temperature=0.0)

    def wrap_model_call(self, request, handler):
        try:
            return handler(request)
        except Exception as error:
            print(f"🔄 [ModelFallback] Primary model call failed ({error}). Activating Fallback backup model '{self.fallback_model_name}'...")
            fallback_llm = self._get_fallback_llm()
            request.model = fallback_llm
            return handler(request)

    async def awrap_model_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as error:
            print(f"🔄 [ModelFallback] Primary model call failed ({error}). Activating Fallback backup model '{self.fallback_model_name}'...")
            fallback_llm = self._get_fallback_llm()
            request.model = fallback_llm
            return await handler(request)


# =============================================================================
# 4. aborted_streaming (User Interruption during LLM Token Streaming)
# =============================================================================
class AbortStreamingMiddleware(AgentMiddleware):
    """Intercepts Ctrl+C or Abort signal during response streaming."""
    def wrap_model_call(self, request, handler):
        try:
            return handler(request)
        except (KeyboardInterrupt, Exception) as error:
            error_str = str(error).lower()
            if isinstance(error, KeyboardInterrupt) or "abort" in error_str or "cancelled" in error_str or "interrupted" in error_str:
                print("\n🛑 [aborted_streaming] User pressed Ctrl+C / Abort signal during streaming.")
                abort_notice = AIMessage(
                    content="🛑 [aborted_streaming Transition] 사용자가 스트리밍 응답 생성을 취소(Ctrl+C / Abort)했습니다."
                )
                return ModelResponse(result=[abort_notice])
            raise error

    async def awrap_model_call(self, request, handler):
        try:
            return await handler(request)
        except (KeyboardInterrupt, Exception) as error:
            error_str = str(error).lower()
            if isinstance(error, KeyboardInterrupt) or "abort" in error_str or "cancelled" in error_str or "interrupted" in error_str:
                print("\n🛑 [aborted_streaming] User pressed Ctrl+C / Abort signal during streaming.")
                abort_notice = AIMessage(
                    content="🛑 [aborted_streaming Transition] 사용자가 스트리밍 응답 생성을 취소(Ctrl+C / Abort)했습니다."
                )
                return ModelResponse(result=[abort_notice])
            raise error

    def handle_abort(self, messages: list) -> list:
        """Standalone helper for direct invocation testing."""
        print("\n🛑 [aborted_streaming] User pressed Ctrl+C / Abort signal during streaming.")
        abort_notice = SystemMessage(
            content="[aborted_streaming Transition] 사용자가 스트리밍 응답 생성을 취소(Ctrl+C / Abort)했습니다."
        )
        return list(messages) + [abort_notice]


# Alias for backward compatibility
AbortStreamingHandler = AbortStreamingMiddleware


# =============================================================================
# 5. aborted_tools (User Interruption during Tool Execution)
# =============================================================================
class AbortToolsMiddleware(AgentMiddleware):
    """Handles User Interruption signal during long-running tool execution (e.g. bash_command)."""
    def wrap_tool_call(self, request, handler):
        tool_name = "unknown_tool"
        tool_call_id = "abort_call_001"
        if hasattr(request, "tool_call") and isinstance(request.tool_call, dict):
            tool_name = request.tool_call.get("name", "unknown_tool")
            tool_call_id = request.tool_call.get("id", "abort_call_001")
        elif hasattr(request, "name"):
            tool_name = getattr(request, "name", "unknown_tool")

        try:
            return handler(request)
        except (KeyboardInterrupt, Exception) as error:
            error_str = str(error).lower()
            if isinstance(error, KeyboardInterrupt) or "abort" in error_str or "cancelled" in error_str or "interrupted" in error_str:
                print(f"\n🛑 [aborted_tools] User interrupted long-running tool '{tool_name}'.")
                abort_tool_result = ToolMessage(
                    content=f"[aborted_tools Transition] '{tool_name}' 도구 실행 도중 사용자 중단 Signal을 수신하여 실행을 취소했습니다.",
                    tool_call_id=tool_call_id,
                    name=tool_name
                )
                return abort_tool_result
            raise error

    async def awrap_tool_call(self, request, handler):
        tool_name = "unknown_tool"
        tool_call_id = "abort_call_001"
        if hasattr(request, "tool_call") and isinstance(request.tool_call, dict):
            tool_name = request.tool_call.get("name", "unknown_tool")
            tool_call_id = request.tool_call.get("id", "abort_call_001")
        elif hasattr(request, "name"):
            tool_name = getattr(request, "name", "unknown_tool")

        try:
            return await handler(request)
        except (KeyboardInterrupt, Exception) as error:
            error_str = str(error).lower()
            if isinstance(error, KeyboardInterrupt) or "abort" in error_str or "cancelled" in error_str or "interrupted" in error_str:
                print(f"\n🛑 [aborted_tools] User interrupted long-running tool '{tool_name}'.")
                abort_tool_result = ToolMessage(
                    content=f"[aborted_tools Transition] '{tool_name}' 도구 실행 도중 사용자 중단 Signal을 수신하여 실행을 취소했습니다.",
                    tool_call_id=tool_call_id,
                    name=tool_name
                )
                return abort_tool_result
            raise error

    def handle_tool_abort(self, messages: list, active_tool_name: str = "bash_command") -> list:
        """Standalone helper for direct invocation testing."""
        print(f"\n🛑 [aborted_tools] User interrupted long-running tool '{active_tool_name}'.")
        abort_tool_result = ToolMessage(
            content=f"[aborted_tools Transition] '{active_tool_name}' 도구 실행 도중 사용자 중단 Signal을 수신하여 실행을 취소했습니다.",
            tool_call_id="abort_call_001",
            name=active_tool_name
        )
        return list(messages) + [abort_tool_result]


# Alias for backward compatibility
AbortToolsHandler = AbortToolsMiddleware
