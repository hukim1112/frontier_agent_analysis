"""
===============================================================================
[Error Control / Self-Recovery] Tool Error Handler Middleware
===============================================================================
도구(Tool) 실행 도중 발생하는 예외를 가로채어 에이전트 크래시 대신
에러 정보를 ToolMessage로 반환하는 미들웨어.

LLM은 에러 ToolMessage를 관찰값(Observation)으로 수신하여
자율적으로 재시도, 우회, 또는 사용자에게 보고를 결정합니다.

참조:
  - Claude Code: Error-as-Observation 패턴
  - LangGraph: ToolNode(handle_tool_errors=True) 대안
===============================================================================
"""

from langchain_core.messages import ToolMessage
from langchain.agents.middleware import AgentMiddleware


class ToolErrorHandlerMiddleware(AgentMiddleware):
    """도구 실행 에러를 안전한 ToolMessage로 변환하는 미들웨어.

    도구가 raise한 예외를 가로채고, 에러 정보를 ToolMessage의 content로 담아
    LLM에 관찰값(Observation)으로 전달합니다. 에이전트는 에러를 읽고
    자율적으로 재시도 또는 대안 행동을 결정합니다.

    Args:
        max_retries: 같은 도구 호출의 자동 재시도 횟수 (기본 0, 즉시 에러 반환).

    Example:
        ```python
        from app.middleware.error_control.self_recovery import ToolErrorHandlerMiddleware

        agent = create_agent(
            model=llm,
            tools=tools,
            middleware=[ToolErrorHandlerMiddleware()]
        )
        ```
    """

    def __init__(self, max_retries: int = 0):
        self.max_retries = max_retries

    def _extract_tool_info(self, request):
        """ToolRequest에서 도구 이름과 ID를 안전하게 추출합니다."""
        tool_name = "unknown_tool"
        tool_call_id = "error_call_001"
        if hasattr(request, "tool_call") and isinstance(request.tool_call, dict):
            tool_name = request.tool_call.get("name", tool_name)
            tool_call_id = request.tool_call.get("id", tool_call_id)
        elif hasattr(request, "name"):
            tool_name = getattr(request, "name", tool_name)
        return tool_name, tool_call_id

    def wrap_tool_call(self, request, handler):
        tool_name, tool_call_id = self._extract_tool_info(request)
        last_error = None

        for attempt in range(1 + self.max_retries):
            try:
                return handler(request)
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    print(f"⚠️ [ToolErrorHandler] Tool '{tool_name}' failed (attempt {attempt+1}/{1+self.max_retries}): {error}")
                    continue

        # 모든 재시도 소진 → 에러를 ToolMessage로 변환 (Error-as-Observation)
        print(f"🛡️ [ToolErrorHandler] Tool '{tool_name}' error caught → converting to ToolMessage observation.")
        return ToolMessage(
            content=(
                f"[Tool Error - {type(last_error).__name__}] "
                f"도구 '{tool_name}' 실행 중 오류가 발생했습니다: {last_error}\n"
                f"다른 방법을 시도하거나 사용자에게 상황을 보고해 주세요."
            ),
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    async def awrap_tool_call(self, request, handler):
        tool_name, tool_call_id = self._extract_tool_info(request)
        last_error = None

        for attempt in range(1 + self.max_retries):
            try:
                return await handler(request)
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    print(f"⚠️ [ToolErrorHandler] Tool '{tool_name}' failed (attempt {attempt+1}/{1+self.max_retries}): {error}")
                    continue

        print(f"🛡️ [ToolErrorHandler] Tool '{tool_name}' error caught → converting to ToolMessage observation.")
        return ToolMessage(
            content=(
                f"[Tool Error - {type(last_error).__name__}] "
                f"도구 '{tool_name}' 실행 중 오류가 발생했습니다: {last_error}\n"
                f"다른 방법을 시도하거나 사용자에게 상황을 보고해 주세요."
            ),
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )
