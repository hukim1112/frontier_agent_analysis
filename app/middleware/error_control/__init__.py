# ===============================================================================
# Error Control Middleware Package
# ===============================================================================
# 에이전트 장애 대응 미들웨어 통합 패키지.
#
# 하위 패키지:
# - self_recovery/   : 시스템/인프라 장애 복구 (Retry, Fallback, Loop Breaker, Abort)
# - self_correction/ : 인지/품질 피드백 교정 (Stop Hooks, Output Validation)
# ===============================================================================

# --- Self-Recovery (시스템/인프라 장애 복구) ---
from .self_recovery import (
    ModelFallbackMiddleware,
    ToolErrorHandlerMiddleware,
    ModelCallLimitMiddleware,
    AbortStreamingMiddleware,
    AbortToolsMiddleware,
)

# --- Self-Correction (인지/품질 피드백 교정) ---
from .self_correction import (
    StopHooksMiddleware,
    extract_code_blocks,
)
ModelErrorHandlerMiddleware = ModelFallbackMiddleware

__all__ = [
    # Self-Recovery
    "ModelFallbackMiddleware",
    "ToolErrorHandlerMiddleware",
    "ModelCallLimitMiddleware",
    "AbortStreamingMiddleware",
    "AbortToolsMiddleware",
    # Self-Correction
    "StopHooksMiddleware",
    "extract_code_blocks",
    "ModelErrorHandlerMiddleware",
]
