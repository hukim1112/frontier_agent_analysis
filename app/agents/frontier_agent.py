"""
Frontier Agent — Hermes 메모리 시스템이 탑재된 프론티어 에이전트.

harness_agent.py 패턴을 따르며 다음이 통합:
- L1: AsyncSqliteSaver (checkpoints.db)
- L2: EpisodicStore (episodic.db) — 세션 검색/인출 도구 + 자동 인출
- L3: SemanticMemoryStore (MEMORY.md/USER.md) — 프리페치 + CRUD 도구
- Background Review: 비동기 데몬 스레드 (Closed Learning Loop)
- PromptAssembler: 4-Layer 동적 프롬프트 조립
"""

import os
import aiosqlite

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.agents import create_agent
from app.utils import get_llm
from app.tools import common_tools
from app.utils.context import AgentContext
from app.middleware.memory_middleware import MemoryMiddleware
from modules.common.agent_tracer import AgentTracer
from modules.hermes.memory_store import SemanticMemoryStore
from modules.hermes.session_store import EpisodicStore
from modules.hermes.memory_tool import create_memory_tool
from modules.hermes.session_search_tool import (
    create_session_search_tool,
    create_session_recall_tool,
)
from modules.hermes.prompt_assembler import PromptAssembler, create_prompt_middleware

AGENT_METADATA = {
    "name": "frontier_agent",
    "description": "Hermes 4계층 메모리 시스템이 탑재된 프론티어 에이전트",
}

# ── System Prompt (PROMPT.md 대체) ──

FRONTIER_SYSTEM_RULES = """You are the Frontier Agent, an advanced AI coding assistant with long-term memory capabilities.

## Core Identity
- You are systematic, concise, and highly effective.
- You have access to 4-layer memory: Working Memory (L1), Episodic Memory (L2), Semantic Memory (L3), and Procedural Memory (L4).
- You search past sessions when user asks about previous work or context.
- You record new facts, user preferences, and tech conventions into semantic memory.
"""


async def create_agent_executor():
    """Frontier Agent 생성 — 비동기 팩토리."""
    
    # 1. LLM 팩토리 (메인 모델: gemini-3.5-flash, 리뷰 모델: openai:gpt-4o-mini)
    llm = get_llm(model_name="gemini-3.5-flash", temperature=0.0)
    review_llm = get_llm(model_name="openai:gpt-4o-mini", temperature=0.0)

    # 2. L1: AsyncSqliteSaver 기반 체크포인터 (Working Memory)
    db_dir = "app/database"
    os.makedirs(db_dir, exist_ok=True)
    checkpoints_path = os.path.join(db_dir, "checkpoints.db")

    conn = await aiosqlite.connect(checkpoints_path, check_same_thread=False)
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()

    # 3. L2: EpisodicStore (Episodic Memory — episodic.db)
    episodic_path = os.path.join(db_dir, "episodic.db")
    episodic_store = EpisodicStore(episodic_path)
    await episodic_store.setup()

    # 4. L3: SemanticMemoryStore (Semantic Memory — MEMORY.md / USER.md)
    memory_dir = "./artifacts/memory"
    os.makedirs(memory_dir, exist_ok=True)
    semantic_store = SemanticMemoryStore(memory_dir)
    semantic_store.load_from_disk()

    # 5. 도구 바인딩: 기본 도구 + 메모리 도구
    memory_tool = create_memory_tool(semantic_store)
    session_search_tool = create_session_search_tool(episodic_store)
    session_recall_tool = create_session_recall_tool(episodic_store)

    tools = common_tools + [memory_tool, session_search_tool, session_recall_tool]

    # 6. PromptAssembler 구성 (Layer 1~5 템플릿 파일 바인딩)
    prompts_dir = "app/prompts"
    prompt_file = os.path.join(prompts_dir, "PROMPT.md")
    system_rules = FRONTIER_SYSTEM_RULES
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            system_rules = f.read().strip()

    assembler = PromptAssembler(
        system_rules=system_rules,
        tool_schemas=tools,
    )

    # Layer 4: Dynamic Session Documents (MCP.md)
    mcp_file = os.path.join(prompts_dir, "MCP.md")
    if os.path.exists(mcp_file):
        assembler.add_l4_doc("MCP.md", mcp_file)

    # Layer 5: User & Project Rules (AGENT.md, SKILL.md)
    agent_file = os.path.join(prompts_dir, "AGENT.md")
    if os.path.exists(agent_file):
        assembler.add_l5_doc("AGENT.md", agent_file)

    skill_file = os.path.join(prompts_dir, "SKILL.md")
    if os.path.exists(skill_file):
        assembler.add_l5_doc("SKILL.md", skill_file)

    prompt_middleware = create_prompt_middleware(assembler)

    # 7. MemoryMiddleware 구성 (초고속 review_llm 적용)
    memory_middleware = MemoryMiddleware(
        semantic_store=semantic_store,
        episodic_store=episodic_store,
        review_llm=review_llm,
    )

    # 8. 미들웨어 스택
    middleware = [
        AgentTracer(log_dir="./artifacts/logs", verbose=True),
        memory_middleware,
        prompt_middleware,
    ]

    # 9. 에이전트 구축
    frontier_agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=None,  # PromptAssembler 미들웨어에서 동적 주입
        checkpointer=checkpointer,
        middleware=middleware,
        context_schema=AgentContext,
    )
    return frontier_agent
