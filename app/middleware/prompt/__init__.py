# ===============================================================================
# Prompt Engineering Middleware Package
# ===============================================================================
# 프롬프트 엔지니어링 미들웨어: 5계층 프롬프트 조립 + KV Cache 최적화
# - PromptAssembler: L1~L5 5계층 프롬프트 + cache_control 메타데이터
# - SkillPromptBuilder: 스킬 디렉토리 스캔 (실행 지침 + 스킬 목록)
# - MCPServerRegistry: configs/mcp.config의 MCP 서버 → 도구(mcp__서버__도구) + 서버 지침
# - prompt_inspector: 노트북/CLI용 조립 결과 요약 · 턴 시뮬레이션 도우미
# ===============================================================================

from .prompt_assembler import (
    PromptAssembler,
    create_prompt_assembler_middleware,
)
from .skill_builder import SkillPromptBuilder
from .mcp_registry import MCPServerRegistry
from .prompt_inspector import (
    build_live_request,
    show_prompt_map,
    show_system_skeleton,
    simulate_turn,
)

__all__ = [
    "PromptAssembler",
    "create_prompt_assembler_middleware",
    "SkillPromptBuilder",
    "MCPServerRegistry",
    "show_prompt_map",
    "show_system_skeleton",
    "build_live_request",
    "simulate_turn",
]
