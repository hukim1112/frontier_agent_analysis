"""
app/agents/main_agent/agent.py — 메인 오케스트레이터 에이전트 (Instructor 완성본)

교육생이 Mission 00~07을 순서대로 완수하면 이 파일과 동일한 결과가 됩니다.

패키지 구조:
  app/agents/main_agent/
   ├── agent.py       (에이전트 조립 및 9단계 미들웨어 파이프라인)
   ├── prompt.py      (SUPERVISOR_SYSTEM_PROMPT — PromptAssembler의 Layer 1)
   ├── tools.py       (계획 5종 + 오케스트레이션 3종 + 공통 7종 = 15종 도구 바인딩)
   └── skills/        (에이전트 내재화 스킬 — Layer 2.2 카탈로그로 노출)

[하네스 구성 요소]
1. 에이전트 프로필 (AGENT_METADATA)
2. 5-Layer 프롬프트 스택 (Claude Code 표준):
   - L1 System Rules & Identity ← SUPERVISOR_SYSTEM_PROMPT (5-Phase 오케스트레이션 루프)
   - L2 Capabilities           ← 도구 스키마 + 스킬 실행 지침(skills/SKILL.md) + skills/ 카탈로그 (SkillPromptBuilder 자동 스캔)
   - L3 Dynamic Session Context ← CWD, Session ID, Session Workspace Root, Date, Host OS
   - L4 Recalled Memory         ← Semantic 스냅샷 + Episodic 과거 세션 요약 힌트
   - L5 Project Rules            ← AGENT.md / CLAUDE.md (현재 미지정)
3. 계층형 메모리 (Layered Memory & Dual-Tier Session Data):
   - L1 단기 기억  : AsyncSqliteSaver (세션별 체크포인터 — AutoCompactor가 압축 교체)
   - L2 에피소드 원문: TranscriptStore (transcript.db — 대화·도구 호출/결과 실시간 Append-only 백업)
   - L2 에피소드 색인: EpisodicStore (episodic.db — 다중 주제 요약·키워드 SQLite FTS5 + session_recall JIT 인출)
   - L3 시맨틱 기억 : SemanticMemoryStore (USER.md / MEMORY.md + memory 도구)
4. Self-Recovery & Self-Correction 미들웨어:
   - ModelFallbackMiddleware    : LLM 장애 시 백업 모델로 failover (413 토큰 초과는 상위로 전파)
   - ToolErrorHandlerMiddleware : 도구 예외 시 ToolMessage(Observation)로 변환
   - ModelCallLimitMiddleware   : 무한 루프 차단 (run_limit=50)
   - StopHooksMiddleware        : 생성된 Python 코드블록 AST 구문 검증 및 자가 교정 루프
5. 도구: tools_supervisor(15종) + memory_tools(2종) + custom_tools(2종) = 총 19종

[미들웨어 파이프라인 실행 순서]
   사용자 요청 →
     [1] AgentLogTracer            ← 가장 바깥: 차단 포함 전체 라이프사이클 감사 로깅
     [2] InputSafetyGuardrail      ← before_agent: 프롬프트 인젝션/탈옥 시도 선제 차단 (jump_to="end")
     [3] TopicAlignmentGuardrail   ← before_agent: 오프토픽 질문 차단 & 대안 안내 (jump_to="end")
     [4] MemoryMiddleware          ← before_agent(원문 기록 & L2/L3 메모리 인출) + wrap_model/tool(원문 기록 & 도구 필터링) + after_agent(유휴 리뷰 예약)
     [5] PromptAssemblerMiddleware ← wrap_model_call: 5계층 시스템 프롬프트 조립
     [6] CompactorMiddleware       ← before_model(AutoCompactor state 교체) + wrap_model_call(Snip/Micro/Collapse + 413 Reactive)
     [7] HumanInTheLoopMiddleware  ← 도구 실행 직전 사용자 승인 요청
     [8] ModelFallbackMiddleware   ← 모델 호출 실패 시 백업 모델로 재시도
     [9] ToolErrorHandlerMiddleware← 도구 실행 오류 시 에러 메시지로 변환
    [10] ModelCallLimitMiddleware  ← 무한 루프 방지 (최대 호출 횟수 제한)
    [11] StopHooksMiddleware       ← after_model: 코드 구문 에러 시 jump_to="model" 자가 교정
                                  → LLM 모델 호출
"""

import os
import json
from pathlib import Path
import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware

from app.utils import init_chat_model
from app.context import AgentContext

from .prompt import SUPERVISOR_SYSTEM_PROMPT
from .tools import tools_supervisor

# Mission 01: 커스텀 도구 (roll_dice, convert_currency)
from app.tools.custom_tools import roll_dice, convert_currency

# Mission 03: 계층형 메모리 + 5계층 프롬프트 조립기
from app.middleware.memory import (
    SemanticMemoryStore,
    EpisodicStore,
    TranscriptStore,
    MemoryMiddleware,
)
from app.middleware.prompt import PromptAssembler, SkillPromptBuilder, create_prompt_assembler_middleware

# Mission 02: Self-Recovery & Self-Correction 미들웨어
from app.middleware.error_control.self_recovery import (
    ModelFallbackMiddleware,
    ToolErrorHandlerMiddleware,
    ModelCallLimitMiddleware,
)
from app.middleware.error_control.self_correction import StopHooksMiddleware

# Claude Code 5-Stage Compaction & Amnesia Guard
from app.middleware.compaction import (
    AmnesiaGuardMiddleware,
    create_compactor_middleware,
)

# Mission 06: 입력 보안 & 주제 정렬 가드레일
from app.middleware.guardrails import InputSafetyGuardrail, TopicAlignmentGuardrail

# Mission 07: 비동기 감사 로깅 (AgentLogTracer)
from app.middleware.observability import AgentLogTracer

SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# 1. 에이전트 프로필
AGENT_METADATA = {
    "name": "main_agent",
    "description": "하네스 기능을 연결할 메인 오케스트레이터 에이전트",
    "roles": ["main"],  # main: UI 대화 전용 (서브에이전트 위임 대상 제외)
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
    # 2. LLM 초기화 (configs/model.config 기반)
    model_cfg = _load_config("./configs/model.config", {
        "model_name": "gemini-3.8-flash",
        "temperature": 0.0,
    })
    llm = init_chat_model(
        model=model_cfg.get("model_name", "gemini-3.8-flash"),
        temperature=model_cfg.get("temperature", 0.0)
    )

    # 3. L1 단기 기억 (SQLite 기반 세션 체크포인터)
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

    # 4. 도구 바인딩 (Supervisor 15종 + Custom 2종 = 17종 / Memory 2종은 memory_mw가 create_agent에 자동 등록 → 총 19종)
    #    ※ PromptAssembler(tool_schemas=active_tools)에서 참조하므로 먼저 선언합니다.
    active_tools = list(tools_supervisor) + [roll_dice, convert_currency]

    # 4-1. 메모리 설정 로드 (configs/memory.config)
    memory_cfg = _load_config("./configs/memory.config", {})
    sem_cfg = memory_cfg.get("semantic_memory", {})
    epi_cfg = memory_cfg.get("episodic_memory", {})
    tr_cfg = memory_cfg.get("transcript", {})

    # 4-2. L3 Semantic Memory Store 초기화 (USER.md, MEMORY.md)
    #    기본값 artifacts/users/default/memory → 사용자별 artifacts/users/{user_id}/memory/ 자동 격리
    memory_dir = sem_cfg.get("memory_dir", "./artifacts/users/default/memory")
    os.makedirs(memory_dir, exist_ok=True)
    semantic_store = SemanticMemoryStore(
        memory_dir=memory_dir,
        memory_char_limit=4000,
        user_char_limit=2000,
    )
    semantic_store.load_from_disk()

    # 4-3. L2 원문 저장소(TranscriptStore) + 요약 색인(EpisodicStore) 초기화
    transcript_db_path = tr_cfg.get("db_path", os.path.join(db_dir, "transcript.db"))
    os.makedirs(os.path.dirname(transcript_db_path) or ".", exist_ok=True)
    transcript_store = TranscriptStore(
        db_path=transcript_db_path,
        recall_max_chars=int(tr_cfg.get("recall_max_chars", 2000)),
    )
    await transcript_store.setup()

    episodic_db_path = epi_cfg.get("db_path", os.path.join(db_dir, "episodic.db"))
    os.makedirs(os.path.dirname(episodic_db_path) or ".", exist_ok=True)
    episodic_store = EpisodicStore(
        db_path=episodic_db_path,
        transcript_store=transcript_store,
    )
    await episodic_store.setup()

    # 4-4. MemoryMiddleware 구성 (스토어 + memory / session_recall 도구 자동 등록 + 원문 기록 + 유휴 요약 훅)
    memory_mw = MemoryMiddleware(
        semantic_store=semantic_store,
        episodic_store=episodic_store,
        review_llm=llm,
        top_k=epi_cfg.get("top_k_prefetch", 2),
        transcript_store=transcript_store,
        transcript_enabled=tr_cfg.get("enabled", True),
        finalize_idle_settle_s=float(epi_cfg.get("finalize_idle_settle_s", 15.0)),
        finalize_max_age_s=float(epi_cfg.get("finalize_max_age_s", 1800.0)),
    )

    # 4-5. Claude Code 표준 16-Module 5-Layer Prompt Assembler 구성
    #    - 스킬: 에이전트 내장 skills/ 만 스캔 (Mission 04 금융 스킬도 이 폴더에 설치)
    #    - skills/SKILL.md(스킬 실행 지침)는 Layer 2.2 카탈로그 상단에만 주입 (L5 중복 주입 안 함)
    #    - claude_code_modules=True: 16대 표준 모듈 + Static/Dynamic 캐시 경계 분리
    #    - bypass_user_context=True: 첫 번째 HumanMessage 앞단에 CLAUDE.md(또는 AGENT.md)와
    #      동결된 Git 스냅샷(git status + recent commits)을 <system-reminder>로 1회 우회 주입
    skill_guidelines = str(SKILLS_DIR / "SKILL.md") if (SKILLS_DIR / "SKILL.md").exists() else None
    skill_builder = SkillPromptBuilder(
        skills_dirs=[str(SKILLS_DIR)],
        guidelines_path=skill_guidelines,
    )

    mcp_md_path = Path("./app/prompts/MCP.md")
    mcp_instructions_text = mcp_md_path.read_text(encoding="utf-8").strip() if mcp_md_path.exists() else None

    claude_md_text = ""
    for candidate_md in (Path("./CLAUDE.md"), Path("./app/prompts/AGENT.md")):
        if candidate_md.exists():
            claude_md_text = candidate_md.read_text(encoding="utf-8").strip()
            break

    assembler = PromptAssembler(
        system_rules=SUPERVISOR_SYSTEM_PROMPT,
        tool_schemas=active_tools,
        skill_catalog=skill_builder.assemble,
        l4_docs={},
        claude_code_modules=True,
        mcp_instructions=mcp_instructions_text,
        mcp_delta_enabled=True,
    )
    prompt_mw = create_prompt_assembler_middleware(
        assembler,
        merge_system=False,
        bypass_user_context=True,
        claude_md_content=claude_md_text,
    )

    # ─────────────────────────────────────────────────────────────────
    # 5. 통합 미들웨어 파이프라인 (보안 & 거버넌스 순서)
    #
    # 미들웨어는 리스트 순서대로 요청을 감싸는 러시안 인형 구조입니다.
    # 리스트 앞쪽 = 바깥 껍데기(먼저 실행), 뒤쪽 = LLM에 가까운 내부.
    #
    #   요청 → [Log] → [Guard] → [Memory] → [Prompt] → [HITL] → [SelfRecovery] → LLM
    # ─────────────────────────────────────────────────────────────────
    middleware = []

    # (1) Logging & Observability — 전체 실행 흐름 JSONL 감사 로깅
    #     log_dir 미지정(기본): artifacts/users/{user_id}/sessions/{session_id}/logs/audit.jsonl
    #     log_dir 지정 시     : {log_dir}/{session_id}/audit.jsonl
    logging_cfg = _load_config("./configs/logging.config", {"logging_enabled": False})
    if logging_cfg.get("logging_enabled"):
        log_dir = logging_cfg.get("log_dir")
        middleware.append(AgentLogTracer(log_dir=log_dir) if log_dir else AgentLogTracer())

    # (2) Guardrails — 프롬프트 인젝션 차단 + 오프토픽 질문 거부
    #     fail_mode: 가드레일 LLM 장애 시 "open"(통과) / "close"(차단) — guardrail.config에서 가드별로 지정
    guardrail_cfg = _load_config("./configs/guardrail.config", {"guardrail_enabled": False})
    if guardrail_cfg.get("guardrail_enabled"):
        guard_model = model_cfg.get("model_name", "gemini-3.8-flash")
        safety_cfg = guardrail_cfg.get("input_safety", {})
        if safety_cfg.get("enabled", True):
            middleware.append(InputSafetyGuardrail(
                model=guard_model,
                fail_mode=safety_cfg.get("fail_mode", "open"),
            ))
        topic_cfg = guardrail_cfg.get("topic_alignment", {})
        if topic_cfg.get("enabled", True):
            middleware.append(TopicAlignmentGuardrail(
                model=guard_model,
                allowed_topics=topic_cfg.get("allowed_topics"),   # 미지정 시 기본 허용 도메인 사용
                blocked_topics=topic_cfg.get("blocked_topics"),
                fail_mode=topic_cfg.get("fail_mode", "open"),
            ))

    # (3) Memory & Prompt — 원문 기록·메모리 인출(before_agent, 가드레일 통과 후 실행) + 도구 필터링 → 5계층 프롬프트 조립
    middleware.extend([memory_mw, prompt_mw])

    # (3-1) Context Compaction & Amnesia Guard (configs/compaction.config)
    compaction_cfg = _load_config("./configs/compaction.config", {"compaction_enabled": False})
    if compaction_cfg.get("compaction_enabled"):
        ag_cfg = compaction_cfg.get("amnesia_guard", {})
        amnesia_guard = None
        if ag_cfg.get("enabled", True):
            amnesia_guard = AmnesiaGuardMiddleware(
                max_restore_files=ag_cfg.get("max_restore_files", 5),
            )
            middleware.append(amnesia_guard)
        auto_cfg = compaction_cfg.get("auto", {})
        micro_cfg = compaction_cfg.get("micro", {})
        middleware.append(
            create_compactor_middleware(
                llm=llm,
                threshold_tokens=auto_cfg.get("threshold_tokens", 8000),
                amnesia_guard=amnesia_guard,
                swap_dir=micro_cfg.get("swap_dir", "./artifacts/swaps"),
                # 원문 기록이 꺼져 있으면 압축 이벤트도 transcript.db에 남기지 않음
                transcript_store=transcript_store if tr_cfg.get("enabled", True) else None,
            )
        )

    # (4) Human-in-the-Loop — 지정 도구(roll_dice) 실행 전 사용자 승인 게이트
    hitl_cfg = _load_config("./configs/hitl.config", {"hitl_enabled": False})
    if hitl_cfg.get("hitl_enabled"):
        interrupt_on = hitl_cfg.get("interrupt_on", {})
        middleware.append(HumanInTheLoopMiddleware(interrupt_on=interrupt_on))

    # (5) Self-Recovery & Self-Correction Circuit Breakers — 모델/도구 장애 복구 및 코드 검증
    backup_model = os.getenv("FALLBACK_MODEL_NAME", "gpt-6.1-sol")
    middleware.extend([
        ModelFallbackMiddleware(
            max_retries=2,
            initial_delay=0.5,
            fallback_model_name=backup_model
        ),
        ToolErrorHandlerMiddleware(max_retries=0),
        ModelCallLimitMiddleware(run_limit=50, exit_behavior="end"),
        StopHooksMiddleware(),
    ])

    # 6. 하네스로 결합된 최종 메인 에이전트 인스턴스 구축
    main_agent = create_agent(
        model=llm,
        tools=active_tools,
        middleware=middleware,
        checkpointer=checkpointer,
        context_schema=AgentContext,
    )

    # 리소스 정리 및 검증용 참조 보존 (테스트 스크립트에서 접근)
    main_agent.registered_tools = active_tools
    main_agent.registered_middlewares = middleware
    main_agent.checkpointer_conn = conn
    main_agent.transcript_store = transcript_store
    main_agent.episodic_store = episodic_store
    main_agent.semantic_store = semantic_store
    main_agent.assembler = assembler
    main_agent.memory_middleware = memory_mw

    return main_agent

