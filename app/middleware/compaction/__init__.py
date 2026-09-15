# ===============================================================================
# Compaction Middleware Package
# ===============================================================================
# 컨텍스트 윈도우 압축 미들웨어: Compactor Pipeline + Amnesia Guard
# - 5-Stage Pipeline: Snip → Micro → Collapse → Auto → Reactive
# - CompactorMiddleware: AgentMiddleware 클래스 (sync + async 지원)
# - AmnesiaGuard: 컴팩션 후 최근 작업 컨텍스트 복원
# ===============================================================================

from .compactor import (
    SnipCompactor,
    MicroCompactor,
    ContextCollapse,
    AutoCompactor,
    ReactiveCompactor,
    CompactorMiddleware,
    create_compactor_middleware,
    UNSAFE_TO_COLLAPSE_TOOLS,
)
from .amnesia_guard import (
    AmnesiaGuardMiddleware,
    create_amnesia_guard_middleware,
)

__all__ = [
    "SnipCompactor",
    "MicroCompactor",
    "ContextCollapse",
    "AutoCompactor",
    "ReactiveCompactor",
    "CompactorMiddleware",
    "create_compactor_middleware",
    "UNSAFE_TO_COLLAPSE_TOOLS",
    "AmnesiaGuardMiddleware",
    "create_amnesia_guard_middleware",
]
