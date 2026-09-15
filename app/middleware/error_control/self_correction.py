"""
Self-Correction Middleware Module (H-03)
코드 생성 오류(SyntaxError, 들여쓰기 에러 등)를 포착하여 blockingError를 주입함으로써
에이전트가 사용자 개입 없이 자가 수정 루프를 돌 수 있도록 합니다.
"""

from modules.claude_code.self_correction import (
    StopHooksMiddleware,
    extract_code_blocks,
)

__all__ = [
    "StopHooksMiddleware",
    "extract_code_blocks",
]

