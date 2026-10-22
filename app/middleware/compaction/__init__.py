# ===============================================================================
# Compaction Middleware Package
# ===============================================================================
# 컨텍스트 윈도우 압축 미들웨어: Claude Code 압축 파이프라인 + Amnesia Guard
# - 순서 (Claude Code query.ts): Tool Result Budget → Snip → Microcompact → Context Collapse
#   → Auto-Compact (+ 413이면 Reactive Compact)
# - CompactorMiddleware: AgentMiddleware 클래스 (sync + async 지원)
# - AmnesiaGuard: 컴팩션 후 최근 작업 컨텍스트 복원
# ===============================================================================

from .compactor import (
    ToolResultBudget,
    SnipCompactor,
    MicroCompactor,
    ContextCollapse,
    AutoCompactor,
    ReactiveCompactor,
    CompactorMiddleware,
    CompactionState,
    create_compactor_middleware,
    claude_code_autocompact_threshold,
)
from .amnesia_guard import (
    AmnesiaGuardMiddleware,
    create_amnesia_guard_middleware,
)

__all__ = [
    "ToolResultBudget",
    "SnipCompactor",
    "MicroCompactor",
    "ContextCollapse",
    "AutoCompactor",
    "ReactiveCompactor",
    "CompactorMiddleware",
    "CompactionState",
    "create_compactor_middleware",
    "claude_code_autocompact_threshold",
    "AmnesiaGuardMiddleware",
    "create_amnesia_guard_middleware",
]
