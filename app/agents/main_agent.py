"""
app/agents/main_agent.py — 메인 오케스트레이터 에이전트 (Instructor 완성본)

교육생이 Mission 00~07을 순서대로 완수하면 이 파일과 동일한 결과가 됩니다.

[하네스 구성 요소]
1. 에이전트 프로필 (AGENT_METADATA)
2. 5-Layer 프롬프트 스택 (Claude Code 표준):
   - L1 System Rules & Identity ← SUPERVISOR_SYSTEM_PROMPT (5-Phase 오케스트레이션 루프)
   - L2 Capabilities           ← 도구 스키마 + skills/ 카탈로그 (SkillPromptBuilder 자동 스캔)
   - L3 Dynamic Session Context ← CWD, Session ID, Date, Host OS
   - L4 Recalled Memory         ← Semantic 스냅샷 + Episodic 과거 세션 요약 힌트
   - L5 Project Rules            ← AGENT.md / SKILL.md
3. 계층형 메모리 (Layered Memory):
   - L1 단기 기억  : AsyncSqliteSaver (세션별 체크포인터)
   - L2 에피소드 기억: EpisodicStore (SQLite FTS5 + finalize_session + session_recall JIT 인출)
   - L3 시맨틱 기억 : SemanticMemoryStore (USER.md / MEMORY.md + memory 도구)
4. Self-Recovery 미들웨어 3종:
   - ModelFallbackMiddleware    : LLM 장애 시 백업 모델로 failover
   - ToolErrorHandlerMiddleware : 도구 예외 시 ToolMessage(Observation)로 변환
   - ModelCallLimitMiddleware   : 무한 루프 차단 (run_limit=50)
5. 도구: tools_supervisor(15종) + memory_tools(2종) + custom_tools(2종) = 총 19종

[미들웨어 파이프라인 실행 순서]
   사용자 요청 →
     [1] AgentLogTracer            ← 가장 바깥: 차단 포함 전체 라이프사이클 감사 로깅
     [2] MemoryMiddleware          ← 안전한 입력만 메모리 조회 (리소스 낭비 방지)
     [3] PromptAssemblerMiddleware ← 5계층 시스템 프롬프트 조립
     [4] ModelFallbackMiddleware   ← 모델 호출 실패 시 백업 모델로 재시도
     [5] ToolErrorHandlerMiddleware← 도구 실행 오류 시 에러 메시지로 변환
     [6] ModelCallLimitMiddleware  ← 무한 루프 방지 (최대 호출 횟수 제한)
                                  → LLM 모델 호출
"""

import os
import json
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from app.utils import init_chat_model
from app.utils.context import AgentContext
from app.prompts import SUPERVISOR_SYSTEM_PROMPT

# Mission 01: 커스텀 도구 (roll_dice, convert_currency)
from app.tools import tools_supervisor
from app.tools.custom_tools import roll_dice, convert_currency

# Mission 03: 계층형 메모리 + 5계층 프롬프트 조립기
from app.middleware.memory import SemanticMemoryStore, EpisodicStore, MemoryMiddleware
from app.middleware.prompt import PromptAssembler, SkillPromptBuilder, create_prompt_assembler_middleware

# Mission 02: Self-Recovery 미들웨어 3종
from app.middleware.error_control.self_recovery import (
    ModelFallbackMiddleware,
    ToolErrorHandlerMiddleware,
    ModelCallLimitMiddleware,
)

# Mission 06: 입력 보안 & 주제 정렬 가드레일 (선택 모듈)
try:
    from app.middleware.guardrails import InputSafetyGuardrail, TopicAlignmentGuardrail
except ImportError:
    InputSafetyGuardrail = None
    TopicAlignmentGuardrail = None

# Mission 07: 비동기 감사 로깅 (AgentLogTracer)
from app.middleware.observability import AgentLogTracer

# 1. 에이전트 프로필
AGENT_METADATA = {
    "name": "main_agent",
    "description": "자가 복구, 가드레일, 비동기 감사 로깅, 5계층 프롬프트 및 계층형 메모리가 탑재된 메인 오케스트레이터 에이전트"
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
    # 2. LLM 초기화 (메인 모델: configs/model.config 기반, 리뷰 모델: OpenAI gpt-4o-mini 분리)
    model_cfg = _load_config("./configs/model.config", {
        "model_name": "gemini-3.7-flash",
        "temperature": 0.0,
    })
    llm = init_chat_model(
        model=model_cfg.get("model_name", "gemini-3.7-flash"),
        temperature=model_cfg.get("temperature", 0.0)
    )
    # 💡 백그라운드 Closed Learning Loop 전용 초고속 OpenAI 리뷰 LLM
    review_llm = init_chat_model(
        model=model_cfg.get("review_model_name", "openai:gpt-4o-mini"),
        temperature=0.0
    )

    # 3. L1 단기 기억 (SQLite 기반 세션 체크포인터)
    db_dir = "app/database"
    os.makedirs(db_dir, exist_ok=True)
    checkpoints_path = os.path.join(db_dir, "checkpoints.db")

    conn = await aiosqlite.connect(checkpoints_path, check_same_thread=False)
    checkpointer = AsyncSqliteSaver(conn)
    await checkpointer.setup()

    # 4. L3 Semantic Memory Store 초기화 (app/database/ 의 USER.md / MEMORY.md 사용)
    semantic_store = SemanticMemoryStore(
        memory_dir=db_dir,
        memory_char_limit=4000,
        user_char_limit=2000,
    )
    semantic_store.load_from_disk()

    # 5. L2 Episodic Memory Store 초기화 (과거 세션 대화 + SQLite FTS5 인덱싱)
    episodic_db_path = os.path.join(db_dir, "episodic.db")
    episodic_store = EpisodicStore(db_path=episodic_db_path)
    await episodic_store.setup()

    # 6. MemoryMiddleware 구성 (스토어 + 3종 도구 + review_llm)
    memory_mw = MemoryMiddleware(
        semantic_store=semantic_store,
        episodic_store=episodic_store,
        review_llm=review_llm,
    )
    # 💡 3종 도구: memory, session_search, session_recall 자동 반환
    memory_tools = memory_mw.get_tools()

    # 7. 전체 도구 세트 통합 (Supervisor 15종 + Memory 3종 = 18종)
    active_tools = list(tools_supervisor) + list(memory_tools)

    # 8. 5-Layer Prompt Assembler 구성
    skill_builder = SkillPromptBuilder(
        skills_dirs=["./skills", "./.agents/skills", "skills", os.path.join(os.getcwd(), "skills")],
        guidelines_path="app/prompts/SKILL.md" if os.path.exists("app/prompts/SKILL.md") else None,
    )

    l5_docs = {}
    if os.path.exists("app/prompts/MCP.md"):
        l5_docs["MCP.md"] = "app/prompts/MCP.md"

    assembler = PromptAssembler(
        system_rules=SUPERVISOR_SYSTEM_PROMPT,
        tool_schemas=active_tools,
        skill_catalog=skill_builder.assemble,
        l5_docs=l5_docs,
        agent_rules_path="app/prompts/AGENT.md" if os.path.exists("app/prompts/AGENT.md") else ("app/prompts/SKILL.md" if os.path.exists("app/prompts/SKILL.md") else None),
    )
    prompt_mw = create_prompt_assembler_middleware(assembler, merge_system=True)


    middleware = []

    # (1) Logging & Observability — 세션별 폴더 분할 감사 로깅
    logging_cfg = _load_config("./configs/logging.config", {"logging_enabled": False})
    if logging_cfg.get("logging_enabled"):
        log_dir = logging_cfg.get("log_dir", "./artifacts/logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = logging_cfg.get("log_path")  # None이면 세션별 {session_id}/audit.jsonl로 자동 분할
        middleware.append(AgentLogTracer(log_dir=log_dir, log_path=log_path))

    # (2) Memory & Prompt — 안전한 입력만 메모리 조회 → 5계층 프롬프트 조립
    middleware.extend([memory_mw, prompt_mw])


    # (3) Self-Recovery Circuit Breakers — 모델/도구 장애 자동 복구
    backup_model = os.getenv("FALLBACK_MODEL_NAME", "gemini-2.5-flash")
    middleware.extend([
        ModelFallbackMiddleware(       # 모델 호출 실패 시 백업 모델로 재시도
            max_retries=2,
            initial_delay=0.5,
            fallback_model_name=backup_model
        ),
        ToolErrorHandlerMiddleware(max_retries=0),   # 도구 예외 → ToolMessage로 변환
        ModelCallLimitMiddleware(run_limit=50, exit_behavior="end"),  # 무한 루프 방지
    ])

    # 9. 하네스로 결합된 최종 메인 에이전트 인스턴스 구축
    #     system_prompt는 PromptAssemblerMiddleware가 동적으로 조립하므로 여기서는 전달하지 않음
    main_agent = create_agent(
        model=llm,
        tools=active_tools,
        middleware=middleware,
        checkpointer=checkpointer,
        context_schema=AgentContext,
    )

    # 리소스 정리 및 검증용 참조 보존
    main_agent.registered_tools = active_tools
    main_agent.checkpointer_conn = conn
    main_agent.episodic_store = episodic_store
    main_agent.semantic_store = semantic_store
    main_agent.assembler = assembler
    main_agent.memory_middleware = memory_mw

    return main_agent
