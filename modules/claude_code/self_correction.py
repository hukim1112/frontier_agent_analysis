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
from langchain.agents.middleware import AgentMiddleware, ModelResponse, hook_config


# 주입되는 blockingError 메시지 식별 태그 (연속 재시도 횟수 계산 및 로그 표기에 사용)
from .message_markers import (
    ORIGIN_STOP_HOOK,
    STOP_HOOK_TAG,  # 주입되는 blockingError 메시지 식별 태그 (로그 표기 및 구버전 메시지 판별)
    is_stop_hook_feedback,
    is_user_turn_start,
    origin_kwargs,
)

# 언어 태그가 python / python3 / py 인 펜스 코드 블록만 검증 대상으로 삼습니다.
# (mermaid / sql / html / bash 및 태그 없는 블록(ASCII 다이어그램 등)은 오탐 방지를 위해 제외)
_PYTHON_FENCE_RE = re.compile(r"```[ \t]*(?:python3?|py)[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code_blocks(text: str) -> list[str]:
    """Helper to extract python-tagged (python/python3/py) code blocks from text."""
    return [block.strip("\n") for block in _PYTHON_FENCE_RE.findall(text or "")]


def _strip_interactive_syntax(code_str: str) -> str:
    """REPL 프롬프트(>>> / ...)와 IPython 매직(!pip, %timeit)을 제거하여 순수 Python 코드만 남깁니다."""
    lines = code_str.splitlines()
    if any(line.lstrip().startswith(">>>") for line in lines):
        repl_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith((">>> ", "... ")):
                repl_lines.append(stripped[4:])
            elif stripped in (">>>", "..."):
                repl_lines.append("")
        lines = repl_lines
    return "\n".join("" if line.lstrip().startswith(("!", "%")) else line for line in lines)


def check_python_syntax(code_str: str) -> Optional[str]:
    """Default built-in python syntax & indentation validator."""
    try:
        compile(_strip_interactive_syntax(code_str), "<string>", "exec")
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg} -> '{e.text.strip() if e.text else ''}'"
    except Exception as e:
        return f"Code validation error: {e}"


def is_stop_hook_message(message: Any) -> bool:
    """StopHooksMiddleware가 주입한 blockingError HumanMessage인지 판별합니다."""
    return is_stop_hook_feedback(message)


# =============================================================================
# 1. stop_hook_blocking (Self-Correction on Code Compilation / Syntax Error)
# =============================================================================
class StopHooksMiddleware(AgentMiddleware):
    """Intercepts agent response, validates python code blocks, and injects blockingError for self-repair.

    Claude Code의 `transition: stop_hook_blocking`은 LangChain에서
    @hook_config(can_jump_to=["model"]) + {"jump_to": "model"} 반환으로 구현됩니다.
    (데코레이터가 없으면 after_agent → END 엣지가 무조건 연결되어 루프가 돌지 않습니다.)

    Returns (after_agent):
        - 검증 실패: {"messages": [blockingError HumanMessage], "jump_to": "model"}
        - 검증 통과 / 코드 없음 / 재시도 한도(max_retries) 도달: None
    """
    def __init__(self, validators: Optional[list[dict[str, Any]]] = None, max_retries: int = 2):
        if validators is None:
            self.validators = [{"name": "python_syntax", "fn": check_python_syntax}]
        else:
            self.validators = validators
        self.max_retries = max_retries

    @staticmethod
    def _count_consecutive_blocks(messages: list) -> int:
        """현재 사용자 턴에서 이미 주입된 blockingError 개수 (실제 사용자 메시지를 만나면 중단)."""
        count = 0
        for msg in reversed(messages):
            if is_user_turn_start(msg):
                break
            if is_stop_hook_message(msg):
                count += 1
        return count

    def _validate(self, content_str: str) -> list[str]:
        validation_errors = []
        for code in extract_code_blocks(content_str):
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
        return validation_errors

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: dict, runtime=None) -> dict | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.content:
            return None

        from app.utils.message_utils import normalize_content
        validation_errors = self._validate(normalize_content(last_msg.content))
        if not validation_errors:
            return None

        retries = self._count_consecutive_blocks(messages)
        if retries >= self.max_retries:
            print(f"\n⚠️ {STOP_HOOK_TAG} Retry limit reached ({retries}/{self.max_retries}) -> Ending turn without further self-repair.")
            return None

        error_details = "\n".join(validation_errors)
        correction_prompt = (
            f"🛑 {STOP_HOOK_TAG} Code Validation Failed (self-repair {retries + 1}/{self.max_retries})\n"
            f"The code you generated failed quality validation tests:\n"
            f"{error_details}\n\n"
            f"Please analyze the error details above, fix the bug in your code, and output the corrected version."
        )
        print(f"\n🔴 {STOP_HOOK_TAG} Validation Failed -> Injecting blockingError & jumping back to model ({retries + 1}/{self.max_retries})")
        # messages 채널은 add_messages 리듀서이므로 새 메시지만 반환하면 기존 이력 뒤에 추가됩니다.
        # harness_origin 표식: 사용자 턴 경계로 세지 않도록 (message_markers 규칙)
        feedback = HumanMessage(content=correction_prompt, additional_kwargs=origin_kwargs(ORIGIN_STOP_HOOK))
        return {"messages": [feedback], "jump_to": "model"}


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
