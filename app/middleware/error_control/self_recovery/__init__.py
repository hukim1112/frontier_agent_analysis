# ===============================================================================
# Self-Recovery Middleware Package
# ===============================================================================
# 시스템/인프라 장애 복구 및 그레이스풀 셧다운 미들웨어.
#
# - ModelFallbackMiddleware: Retry + Fallback + Safe Failure 통합 복구
# - ToolErrorHandlerMiddleware: 도구 실행 에러 → 안전한 ToolMessage 변환
# - ModelCallLimitMiddleware: LangChain 내장 무한 루프 차단 (re-export)
# - AbortStreamingMiddleware: 사용자 스트리밍 중단 처리
# - AbortToolsMiddleware: 도구 실행 인터럽트 처리
# ===============================================================================

from .model_fallback import ModelFallbackMiddleware
from .tool_error_handler import ToolErrorHandlerMiddleware
from .abort_handlers import AbortStreamingMiddleware, AbortToolsMiddleware

# Built-in re-export: 무한 루프 차단
from langchain.agents.middleware import ModelCallLimitMiddleware

__all__ = [
    "ModelFallbackMiddleware",
    "ToolErrorHandlerMiddleware",
    "ModelCallLimitMiddleware",
    "AbortStreamingMiddleware",
    "AbortToolsMiddleware",
]


ModelErrorHandlerMiddleware = ModelFallbackMiddleware
__all__.append('ModelErrorHandlerMiddleware')
