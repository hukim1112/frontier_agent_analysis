"""
AutoCompactor Middleware — app/middleware/ 이식 버전.

modules/claude_code/compactor.py에서 이식.
"""

from modules.claude_code.compactor import (
    AutoCompactor,
    create_compactor_middleware,
)

__all__ = ["AutoCompactor", "create_compactor_middleware"]
