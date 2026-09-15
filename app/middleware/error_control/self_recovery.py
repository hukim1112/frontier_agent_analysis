"""
Self-Recovery Middleware Module (H-03)
시스템 및 인프라 장애(API 에러, 도구 실행 예외, 무한 루프 등)를 자동으로 격리/복구합니다.
"""

import time
import asyncio
from typing import Optional, Any
from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware as LangChainModelCallLimitMiddleware,
    ModelResponse,
)
from langchain_core.messages import ToolMessage, AIMessage
from modules.claude_code.self_correction import (
    ModelErrorHandlerMiddleware,
    AbortStreamingMiddleware,
    AbortToolsMiddleware,
)

# LangChain 기본 제공 ModelCallLimitMiddleware 활용
ModelCallLimitMiddleware = LangChainModelCallLimitMiddleware


class ModelFallbackMiddleware(AgentMiddleware):
    """메인 LLM 장애 시 지정된 횟수만큼 재시도 후 백업 Fallback 모델로 자동 failover합니다."""

    def __init__(
        self,
        fallback_model_name: str = "gemini-2.5-flash",
        fallback_llm: Optional[Any] = None,
        max_retries: int = 2,
        initial_delay: float = 0.5,
        **kwargs,
    ):
        self.fallback_model_name = fallback_model_name
        self.fallback_llm = fallback_llm
        self.max_retries = max_retries
        self.initial_delay = initial_delay

    def _get_fallback_llm(self):
        if self.fallback_llm is not None:
            return self.fallback_llm
        try:
            from app.utils import init_chat_model
            return init_chat_model(model=self.fallback_model_name, temperature=0.0)
        except Exception:
            from app.utils.llm import get_llm
            return get_llm(model_name=self.fallback_model_name, temperature=0.0)

    def wrap_model_call(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return handler(request)
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    sleep_time = self.initial_delay * (2 ** attempt)
                    time.sleep(sleep_time)

        print(f"🔄 [ModelFallback] Primary model call failed ({last_error}). Activating Fallback backup model '{self.fallback_model_name}'...")
        request.model = self._get_fallback_llm()
        return handler(request)

    async def awrap_model_call(self, request, handler):
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return await handler(request)
            except Exception as error:
                last_error = error
                if attempt < self.max_retries:
                    sleep_time = self.initial_delay * (2 ** attempt)
                    await asyncio.sleep(sleep_time)

        print(f"🔄 [ModelFallback] Primary model call failed ({last_error}). Activating Fallback backup model '{self.fallback_model_name}'...")
        request.model = self._get_fallback_llm()
        return await handler(request)


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
