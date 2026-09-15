# ===============================================================================
# Middleware Root Package
# ===============================================================================
# 하위 패키지별 미들웨어를 조직화하여 제공합니다.
#
# 패키지 구조:
# - observability/   : 관측성 (로깅, 추적, 시각화)
# - compaction/       : 컨텍스트 윈도우 압축
# - error_control/    : 장애 대응 (자가 복구 + 자가 교정)
#   - self_recovery/  : 시스템/인프라 장애 복구 (Retry, Fallback, Loop Breaker)
#   - self_correction/: 인지/품질 피드백 교정 (Stop Hooks)
# - prompt/           : 프롬프트 엔지니어링 (5계층 조립)
# - guardrails/       : 안전 필터 (Tier 2)
# - memory/           : Hermes 메모리 (Tier 2)
# - evaluator/        : 평가 하네스 & LLM-as-a-Judge (Tier 2)
# ===============================================================================

# --- Observability (H-04) ---
from .observability.agent_log_tracer import AgentLogTracer, AgentTracer, LoggingMiddleware
from .observability.visualizer import HierarchicalVisualizerMiddleware

# --- Compaction (H-01, H-02) ---
from .compaction.compactor import create_compactor_middleware
from .compaction.amnesia_guard import AmnesiaGuardMiddleware, create_amnesia_guard_middleware

# --- Error Control (H-03) ---
from .error_control import (
    # Self-Recovery
    ModelFallbackMiddleware,
    ToolErrorHandlerMiddleware,
    ModelCallLimitMiddleware,
    AbortStreamingMiddleware,
    AbortToolsMiddleware,
    # Self-Correction
    StopHooksMiddleware,
)

# --- Prompt Engineering (H-05) ---
from .prompt.prompt_assembler import PromptAssembler, create_prompt_assembler_middleware

# --- Memory (Tier 2) ---
from .memory.semantic_store import SemanticMemoryStore
from .memory.episodic_store import EpisodicStore
from .memory.memory_middleware import MemoryMiddleware

# --- Guardrails (Tier 2) ---
try:
    from .guardrails.guardrails import (
        InputSafetyGuardrail,
        TopicAlignmentGuardrail,
        OutputSchemaRepairGuardrail,
    )
except ImportError:
    InputSafetyGuardrail = None
    TopicAlignmentGuardrail = None
    OutputSchemaRepairGuardrail = None

# --- Evaluator & LLM-as-a-Judge (H-06) ---
try:
    from .evaluator.evaluator import EvaluatorHarness, JudgeVerdict
except ImportError:
    EvaluatorHarness = None
    JudgeVerdict = None

__all__ = [
    # Observability
    "AgentLogTracer",
    "AgentTracer",
    "LoggingMiddleware",
    "HierarchicalVisualizerMiddleware",
    # Compaction
    "create_compactor_middleware",
    "AmnesiaGuardMiddleware",
    "create_amnesia_guard_middleware",
    # Error Control
    "StopHooksMiddleware",
    "ModelFallbackMiddleware",
    "ToolErrorHandlerMiddleware",
    "ModelCallLimitMiddleware",
    "AbortStreamingMiddleware",
    "AbortToolsMiddleware",
    # Prompt Engineering
    "PromptAssembler",
    "create_prompt_assembler_middleware",
    # Memory
    "SemanticMemoryStore",
    "EpisodicStore",
    "MemoryMiddleware",
    # Guardrails
    "InputSafetyGuardrail",
    "TopicAlignmentGuardrail",
    "OutputSchemaRepairGuardrail",
    # Evaluator & LLM-as-a-Judge
    "EvaluatorHarness",
    "JudgeVerdict",
]
