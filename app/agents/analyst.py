"""
===============================================================================
[AAWS Agent] Analyst — 데이터 분석·시각화·보고서 전문 에이전트
===============================================================================
데이터 파일(JSON/CSV/Excel 등)을 분석하고, 차트/그래프를 생성하며,
전문적인 Excel 보고서와 인터랙티브 HTML 대시보드를 생성합니다.

아키텍처:
  🔬 Analyst (이 파일)
   ├── Analysis Tools: data_profiler, data_query, chart_generator
   ├── Output Tools: file_converter, excel_writer, html_report
   └── Common Tools: file_read, file_writer, file_edit, grep_search, glob_search, bash_command
===============================================================================
"""

import os
import json
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from app.utils import init_chat_model
from app.prompts import ANALYST_SYSTEM_PROMPT
from app.tools import tools_analyst
from app.utils.context import AgentContext

AGENT_METADATA = {
    "name": "analyst",
    "description": "데이터 분석·시각화·보고서 전문 에이전트 — 데이터 프로파일링, 차트 생성, Excel/HTML 보고서를 생성합니다.",
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


async def create_agent_executor():
    # 1. LLM 설정
    llm = init_chat_model(model="gemini-3.7-flash", temperature=0.0)

    # 2. AsyncSqliteSaver 기반 체크포인터 (SQLite 영구 메모리)
    db_dir = "app/database"
    os.makedirs(db_dir, exist_ok=True)
    checkpoints_path = os.path.join(db_dir, "checkpoints.db")

    conn = await aiosqlite.connect(checkpoints_path, check_same_thread=False)
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()

    # 3. HITL 미들웨어 동적 구성 (configs/hitl.config 기반)
    hitl_cfg = _load_config("./configs/hitl.config", {"hitl_enabled": False})
    middleware = []
    if hitl_cfg.get("hitl_enabled"):
        interrupt_on = hitl_cfg.get("interrupt_on", {})
        if interrupt_on:
            middleware.append(
                HumanInTheLoopMiddleware(
                    interrupt_on=interrupt_on,
                    description_prefix="Analyst 도구 실행 승인 요청",
                )
            )

    # 4. Analyst 에이전트 구축
    #    tools_analyst = Analysis(3) + Output(3) + Common(6) = 12종
    analyst_agent = create_agent(
        model=llm,
        tools=tools_analyst,
        system_prompt=ANALYST_SYSTEM_PROMPT,
        middleware=middleware,
        checkpointer=checkpointer,
        context_schema=AgentContext,
    )
    return analyst_agent
