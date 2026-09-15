"""
Self-Recovery Middleware Module (H-03)
시스템 및 인프라 장애(API 에러, 도구 실행 예외, 무한 루프 등)를 자동으로 격리/복구합니다.
"""

from typing import Optional, Any
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelFallbackMiddleware as LangChainModelFallbackMiddleware,
    ModelCallLimitMiddleware as LangChainModelCallLimitMiddleware,
)
from langchain_core.messages import ToolMessage, AIMessage
from langchain.agents.middleware import ModelResponse
from modules.claude_code.self_correction import (
    ModelFallbackMiddleware,
    ModelErrorHandlerMiddleware,
    AbortStreamingMiddleware,
    AbortToolsMiddleware,
)

# LangChain 기본 제공 ModelCallLimitMiddleware 활용
ModelCallLimitMiddleware = LangChainModelCallLimitMiddleware


class ToolErrorHandlerMiddleware(AgentMiddleware):
    """도구 실행 중 예외가 발생하면 크래시 대신 에러 ToolMessage로 안전하게 캡슐화합니다."""

    def __init__(self, max_retries: int = 0):
        self.max_retries = max_retries

    def wrap_tool_call(self, request, handler):
        try:
            return handler(request)
        except Exception as e:
            tool_name = getattr(request, "name", "tool")
            tool_call_id = getattr(request, "tool_call_id", getattr(request, "id", "call_error"))
            if hasattr(request, "tool_call") and isinstance(request.tool_call, dict):
                tool_name = request.tool_call.get("name", tool_name)
                tool_call_id = request.tool_call.get("id", tool_call_id)
            return ToolMessage(
                content=f"Error executing tool '{tool_name}': {str(e)}",
                tool_call_id=tool_call_id,
                name=tool_name,
            )

    async def awrap_tool_call(self, request, handler):
        try:
            return await handler(request)
        except Exception as e:
            tool_name = getattr(request, "name", "tool")
            tool_call_id = getattr(request, "tool_call_id", getattr(request, "id", "call_error"))
            if hasattr(request, "tool_call") and isinstance(request.tool_call, dict):
                tool_name = request.tool_call.get("name", tool_name)
                tool_call_id = request.tool_call.get("id", tool_call_id)
            return ToolMessage(
                content=f"Error executing tool '{tool_name}': {str(e)}",
                tool_call_id=tool_call_id,
                name=tool_name,
            )


__all__ = [
    "ModelFallbackMiddleware",
    "ModelErrorHandlerMiddleware",
    "ToolErrorHandlerMiddleware",
    "ModelCallLimitMiddleware",
    "AbortStreamingMiddleware",
    "AbortToolsMiddleware",
]
