"""
===============================================================================
[Analyst Agent] 데이터 분석·시각화·보고서 전문 에이전트
===============================================================================
데이터 파일(JSON/CSV/Excel 등)을 분석하고, 차트/그래프를 생성하며,
전문적인 Excel 보고서와 인터랙티브 HTML 대시보드를 생성합니다.

패키지 구조:
  app/agents/analyst/
   ├── agent.py       (에이전트 조립 및 서브에이전트 맞춤 미들웨어 파이프라인)
   ├── prompt.py      (ANALYST_SYSTEM_PROMPT — PromptAssembler의 Layer 1)
   ├── tools.py       (분석 3종 + 출력 3종 + 공통 6종 = 12종 도구 바인딩)
   └── skills/        (애널리스트 전용 스킬 문서 — Layer 2.2 카탈로그로 노출)
===============================================================================
"""

import os
import json
from pathlib import Path
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.agents import create_agent

from app.utils import init_chat_model
from app.context import AgentContext
from app.middleware.prompt import PromptAssembler, SkillPromptBuilder, create_prompt_assembler_middleware
from app.middleware.observability import AgentLogTracer
from app.middleware.error_control.self_recovery import (
    ModelFallbackMiddleware,
    ToolErrorHandlerMiddleware,
    ModelCallLimitMiddleware,
)

from .prompt import ANALYST_SYSTEM_PROMPT
from .tools import tools_analyst

SKILLS_DIR = Path(__file__).resolve().parent / "skills"

AGENT_METADATA = {
    "name": "analyst",
    "description": "데이터 분석·시각화·보고서 전문 에이전트 — 데이터 프로파일링, 차트 생성, Excel/HTML 보고서를 생성합니다.",
    "roles": ["main", "sub"],  # main: UI 대화 대상 / sub: Supervisor 위임 대상
}


def _load_config(path: str, default: dict) -> dict:
    """설정 파일을 로드합니다. 실패 시 기본값을 반환합니다."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


async def create_agent_executor(checkpointer=None):
    # 1. LLM 설정
    model_cfg = _load_config("./configs/model.config", {
        "model_name": "gemini-3.8-flash",
        "temperature": 0.0,
    })
    llm = init_chat_model(
        model=model_cfg.get("model_name", "gemini-3.8-flash"),
        temperature=model_cfg.get("temperature", 0.0),
    )

    # 2. AsyncSqliteSaver 기반 체크포인터 (SQLite 영구 메모리)
    db_dir = "app/database"
    os.makedirs(db_dir, exist_ok=True)
    checkpoints_path = os.path.join(db_dir, "checkpoints.db")

    conn = None
    if checkpointer is None:
        conn = await aiosqlite.connect(checkpoints_path, check_same_thread=False)
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA busy_timeout=5000;")
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()

    # 3. 5-Layer Prompt Assembler (Analyst 내재화 skills/ 카탈로그 + L3 Session Workspace Root 동적 주입)
    #    skills/SKILL.md(가이드 허브)는 Layer 2.2 카탈로그 상단에만 주입 (L5 중복 주입 안 함)
    skill_guidelines = str(SKILLS_DIR / "SKILL.md") if (SKILLS_DIR / "SKILL.md").exists() else None
    skill_builder = SkillPromptBuilder(
        skills_dirs=[str(SKILLS_DIR)],
        guidelines_path=skill_guidelines,
    )
    assembler = PromptAssembler(
        system_rules=ANALYST_SYSTEM_PROMPT,
        tool_schemas=list(tools_analyst),
        skill_catalog=skill_builder.assemble,
    )
    prompt_mw = create_prompt_assembler_middleware(assembler, merge_system=True)

    # 4. Analyst 맞춤 미들웨어 파이프라인
    #    [Log] → [PromptAssembler (L1~L5)] → [Self-Recovery 3종]
    #    ※ HITL 미적용: 서브에이전트는 백그라운드 Job으로 실행되어 승인 UI가 없으므로
    #      사용자 승인 게이트는 메인 에이전트(main_agent)에서만 운용합니다.
    middleware = []

    # log_dir 미지정(기본): 서브에이전트 워크스페이스 logs/audit.jsonl / 지정 시: {log_dir}/{session_id}/audit.jsonl
    logging_cfg = _load_config("./configs/logging.config", {"logging_enabled": False})
    if logging_cfg.get("logging_enabled"):
        log_dir = logging_cfg.get("log_dir")
        middleware.append(AgentLogTracer(log_dir=log_dir) if log_dir else AgentLogTracer())

    middleware.append(prompt_mw)

    backup_model = os.getenv("FALLBACK_MODEL_NAME", "gpt-6.1-sol")
    middleware.extend([
        ModelFallbackMiddleware(max_retries=2, initial_delay=0.5, fallback_model_name=backup_model),
        ToolErrorHandlerMiddleware(max_retries=0),
        ModelCallLimitMiddleware(run_limit=50, exit_behavior="end"),
    ])

    # 5. Analyst 에이전트 구축 (Analysis 3종 + Output 3종 + Common 6종 = 12종)
    analyst_agent = create_agent(
        model=llm,
        tools=tools_analyst,
        middleware=middleware,
        checkpointer=checkpointer,
        context_schema=AgentContext,
    )
    analyst_agent.registered_tools = list(tools_analyst)
    analyst_agent.checkpointer_conn = conn
    analyst_agent.assembler = assembler
    return analyst_agent

