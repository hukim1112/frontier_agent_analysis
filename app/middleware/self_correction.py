"""
Self Correction Middleware — app/middleware/ 이식 버전.

modules/claude_code/self_correction.py에서 이식.
"""

from modules.claude_code.self_correction import (
    SelfCorrectionMiddleware,
    PostCompactionGuard,
)

__all__ = ["SelfCorrectionMiddleware", "PostCompactionGuard"]
