"""
===============================================================================
Memory Middleware — AgentMiddleware (before_agent + wrap hooks + after_agent + tools)
===============================================================================
Source: frontier-agent-lab/modules/hermes/memory_middleware.py
        + memory_tool.py + session_search_tool.py (도구 통합)

고응집 메모리 미들웨어 — 스토어, 도구, 훅을 모두 소유합니다.

세션 데이터 이원화 (Dual-tier Session Data):
- 압축본 (L1 Short-term / Working): checkpointer state — 모델 입력용, Auto-Compact가 교체
- 전체본 (L2 Episodic 원문): transcript.db — 이 미들웨어가 메시지가 생성되는 즉시 추가 기록
- 색인 (L2 Episodic 요약): episodic.db — 대화가 멈춘 뒤 원문을 읽어 주제별 요약 생성
- L3 Semantic: MEMORY.md / USER.md
- Procedural: skills/<name>/SKILL.md — 반복 작업 유형의 절차 (ProceduralMemoryStore, 선택)

Hooks:
- before_agent / abefore_agent:
    새 턴 시작 → 실행 중인 백그라운드 리뷰 취소 + 에피소드 추출 대기 보류, 사용자 질문 원문 기록,
    메모리 리뷰 턴 카운터 +1, L3(시맨틱) + Procedural(스킬 카탈로그) 프리페치(세션별 Frozen Snapshot → memory_snapshot)
    + L2(에피소드 요약) 자동 인출(→ episodic_context, 및 하위 호환 합본 recalled_memory) 주입
- wrap_model_call: 꺼진 메모리 도구 필터링 → 모델 응답(AI 메시지·도구 호출) 원문 기록,
    도구를 호출한 응답이면 스킬 리뷰 카운터 +1
- wrap_tool_call: 사용자 ContextVar 전파 + 꺼진 도구 차단 → 도구 결과 원문 기록,
    memory / skill_manage 직접 호출 시 해당 리뷰 카운터 리셋
- after_agent / aafter_agent: 누락분 보정 기록(가드레일 차단 턴은 kind=blocked), 그리고 두 작업을 따로 판단:
    ① Background Review (메모리·스킬 리뷰) — nudge 조건(_background_review_due)을 만족하면 즉시 실행
    ② 에피소드 추출 (finalize_session) — 세션 종료 조건(IdleReviewQueue.session_end_reason)까지 대기 후 실행

Hermes 대응 (두 작업은 별개의 경로):
- Background Review = agent/background_review.py: 턴 종료 시 nudge 조건(사용자 턴 / 도구 호출 응답 10회)
- 에피소드 추출 = 외부 메모리 provider의 on_session_end: 세션 경계(/new·종료·압축·게이트웨이 1시간 유휴)
교육용 설정(configs/memory.config): memory_nudge_interval=1(매 턴), 세션 종료 유휴 15초

원문 기록 스위치는 메모리 학습/주입 스위치와 독립입니다:
- memory.config의 transcript.enabled (배포 정책) + AgentContext.transcript_enabled (요청별)
- episodic_memory_enabled: 과거 세션 요약을 검색해 프롬프트에 주입할지
- episodic_learning_enabled: 원문을 LLM으로 요약해 episodic.db에 학습할지

Tool Factory:
- self.tools: [memory(), session_recall()] 도구를 소유하며, create_agent()가 생성 시 자동으로 에이전트 도구로 주입
  - memory(): 에이전트가 직접 MEMORY.md / USER.md를 관리 (add/replace/remove)
  - session_recall(): Anchor 기반 과거 세션 원문 인출 (transcript.db)
  - skill_view() / skill_manage(): procedural_store가 있을 때만 추가 (skill_tools.py)
  - session_search는 도구로 제공하지 않음 (before_agent가 자동 인출)

Background Review (Hermes _spawn_background_review 패턴):
- 메인 스레드를 블로킹하지 않음 (ThreadPoolExecutor + 세션별 cancel_event, 새 턴이 오면 취소)
- Semantic: 별도 LLM 호출로 대화 리뷰 → memory() 파싱 실행. add만 바로 반영하고 replace/remove는
  pending.json에 제안으로만 남김 (Hermes _background_delete_gate — 사용자 없이 기억을 지우지 않음)
- Procedural: 스킬 도구만 가진 리뷰 에이전트 실행 (skill_review.py)

에피소드 추출 (Hermes MemoryProvider.on_session_end 대응):
- 세션 종료로 판단되면 원문을 LLM으로 주제별 요약 → EpisodicStore(episodic.db) 색인
- 같은 세션이 다시 이어지면 다음 세션 종료 때 원문 전체를 다시 요약해 같은 행을 갱신
===============================================================================
"""

import re
import json
import time
import logging
import asyncio
import threading
import concurrent.futures
from collections import OrderedDict
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain.tools import tool as langchain_tool
from app.utils.message_utils import normalize_content
from app.middleware.memory.semantic_store import SemanticMemoryStore
from app.middleware.memory.procedural_store import ProceduralMemoryStore
from app.middleware.memory.skill_tools import SKILL_MANAGE_TOOL, SKILL_VIEW_TOOL, build_skill_tools
from app.middleware.memory.skill_review import run_skill_review
from app.middleware.memory.review_queue import IdleReviewQueue
from app.utils.message_markers import last_user_turn_start
from app.middleware.memory.transcript_store import (
    TranscriptStore,
    ensure_message_id,
    format_tool_call,
    is_blocked_message,
    render_transcript_for_summary,
)

logger = logging.getLogger(__name__)

_DEFAULT_USER_ID = "default"
_memory_user_id_var: ContextVar[str] = ContextVar("_memory_user_id_var", default=_DEFAULT_USER_ID)

# ── Background Review Prompts ──

COMBINED_REVIEW_PROMPT = """Review the conversation above and update two things:

**Memory** — who the user is.
Did the user reveal personal information, preferences, communication style, or workflow habits?
If so, save them using: memory(action="add", target="user", content="...")
Write entries in concise English. Keep each entry to 1-2 sentences.

**Agent Notes** — what you learned.
Did you discover new environment facts, project conventions, tool quirks, or useful patterns?
If so, save them using: memory(action="add", target="memory", content="...")
Write entries in concise English. Keep each entry to 1-2 sentences.

IMPORTANT:
- Only save genuinely NEW information. Check existing entries first and do not add duplicates.
- Do NOT save task-specific data, logs, conversation summaries, or temporary state.
- If nothing is worth saving, respond with "Nothing to save."
- This review runs without the user present, so it may only ADD entries. If an existing entry is now
  wrong or duplicated, you may propose memory(action="replace", target=..., old_text=..., content=...)
  or memory(action="remove", target=..., old_text=...); those proposals are staged for the user's
  approval and are NOT applied.

Current MEMORY.md entries:
{memory_entries}

Current USER.md entries:
{user_entries}
"""


def _serialize_message_for_review(msg) -> Dict[str, Any]:
    """메시지를 리뷰/원문 저장용 구조화 딕셔너리로 직렬화."""
    if isinstance(msg, BaseMessage):
        msg_id = ensure_message_id(msg)
        data: Dict[str, Any] = {
            "id": msg_id,
            "role": msg.type,
            "content": normalize_content(msg.content) if msg.content else "",
        }
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            data["tool_calls"] = [format_tool_call(tc) for tc in msg.tool_calls]
        if isinstance(msg, ToolMessage):
            data["tool_call_id"] = getattr(msg, "tool_call_id", None)
            data["tool_name"] = getattr(msg, "name", None)
        ak = getattr(msg, "additional_kwargs", None)
        if isinstance(ak, dict) and ak:
            data["additional_kwargs"] = dict(ak)
        return data
    if isinstance(msg, dict):
        return dict(msg)
    return {"role": "unknown", "content": str(msg)}


def _extract_response_messages(response: Any) -> List[BaseMessage]:
    """wrap_model_call 핸들러 반환값(ModelResponse 또는 AIMessage)에서 BaseMessage 목록 추출."""
    if isinstance(response, BaseMessage):
        return [response]
    result = getattr(response, "result", None)
    if isinstance(result, list):
        return [m for m in result if isinstance(m, BaseMessage)]
    if isinstance(response, dict):
        msgs = response.get("messages")
        if isinstance(msgs, list):
            return [m for m in msgs if isinstance(m, BaseMessage)]
    return []


def _sanitize_user_id(user_id: Optional[str]) -> str:
    from app.utils.workspace import sanitize_path_segment
    return sanitize_path_segment(user_id, fallback=_DEFAULT_USER_ID)


class MemoryMiddleware(AgentMiddleware):
    """고응집 메모리 미들웨어 — 스토어, 도구, 훅을 모두 소유.

    Usage:
        memory_mw = MemoryMiddleware(
            semantic_store=semantic_store,
            episodic_store=episodic_store,
            review_llm=llm,
        )
        agent = create_agent(
            model=llm,
            tools=base_tools,          # memory_mw.tools는 create_agent가 자동 등록
            middleware=[memory_mw],
        )
    """

    def __init__(
        self,
        semantic_store: SemanticMemoryStore,
        episodic_store,
        review_llm=None,
        top_k: int = 2,
        auto_recall_episodic: bool = True,
        transcript_store: Optional[TranscriptStore] = None,
        transcript_enabled: bool = True,
        finalize_idle_settle_s: float = 15.0,
        finalize_max_age_s: float = 1800.0,
        procedural_store: Optional[ProceduralMemoryStore] = None,
        skill_nudge_interval: int = 10,
        skill_review_max_iterations: int = 16,
        memory_nudge_interval: int = 10,
    ):
        """
        Args:
            semantic_store: 기본 사용자(default)의 SemanticMemoryStore
            episodic_store: 공유 EpisodicStore (user_id 컬럼으로 사용자 격리)
            review_llm: Background Review(시맨틱 갱신·세션 요약·스킬 리뷰)에 사용할 LLM
            top_k: 매 턴 자동 인출(episodic_context)할 과거 세션(Episodic) 요약 개수
            transcript_store: 원문 전체본 저장소 (TranscriptStore, 미지정 시 episodic_store.transcript 사용)
            transcript_enabled: 대화 원문(transcript.db) 자동 백업 활성화 여부 (메모리 학습 스위치와 독립)
            finalize_idle_settle_s: 세션 종료로 볼 유휴 시간(초, 기본 15초) — 에피소드 추출 시점
            finalize_max_age_s: 연속 대화 시 에피소드 추출을 미룰 최대 시간(초, 기본 1800초=30분)
            procedural_store: 기본 사용자의 ProceduralMemoryStore (None이면 Procedural Memory 비활성)
            skill_nudge_interval: 스킬 리뷰를 예약할 도구 호출 반복 횟수 (Hermes 기본 10, 0이면 끔)
            skill_review_max_iterations: 스킬 리뷰 에이전트의 최대 모델 호출 횟수 (Hermes 기본 16)
            memory_nudge_interval: Semantic 리뷰를 실행할 사용자 턴 수 (Hermes 기본 10, 교육용 config는 1, 0이면 끔)
        """
        self.semantic_store = semantic_store
        self.episodic_store = episodic_store
        self.review_llm = review_llm
        self.top_k = max(0, int(top_k))
        self.transcript_enabled = bool(transcript_enabled)
        self.procedural_store = procedural_store
        self.skill_nudge_interval = max(0, int(skill_nudge_interval))
        self.skill_review_max_iterations = max(1, int(skill_review_max_iterations))
        self.memory_nudge_interval = max(0, int(memory_nudge_interval))

        if transcript_store is not None:
            self.transcript_store: Optional[TranscriptStore] = transcript_store
            if self.episodic_store is not None:
                self.episodic_store.transcript_store = transcript_store
        elif self.episodic_store is not None and hasattr(self.episodic_store, "transcript"):
            self.transcript_store = self.episodic_store.transcript
        else:
            self.transcript_store = None

        from pathlib import Path
        from app.utils.workspace import USERS_BASE_DIR, resolve_user_memory_dir

        default_unified_mem_dir = resolve_user_memory_dir(_DEFAULT_USER_ID, ensure_exists=False).resolve()
        base_sem_dir = Path(semantic_store.memory_dir).resolve()
        self._use_unified_workspace = (
            base_sem_dir == default_unified_mem_dir
            or base_sem_dir == USERS_BASE_DIR.resolve()
        )

        # 사용자(user_id)별 SemanticMemoryStore 캐시 (실시간 쓰기용, 사용자당 1개)
        # (EpisodicStore는 app/database/episodic.db 단일 DB를 공유하며 user_id 컬럼으로 격리)
        self._store_lock = threading.RLock()
        self._user_semantic_stores: Dict[str, SemanticMemoryStore] = {
            _DEFAULT_USER_ID: semantic_store
        }
        # 세션별 Frozen Snapshot: (user_id, session_id) → 그 세션 첫 턴에 렌더링한 프롬프트 블록
        # 같은 사용자가 세션을 오가도 각 세션의 L4 메모리 텍스트는 세션 내내 그대로입니다 (LRU로 개수 제한).
        self._prompt_snapshots: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
        self._max_prompt_snapshots = 256

        # Semantic Memory 리뷰 트리거 (Hermes _turns_since_memory 패턴)
        self._memory_lock = threading.Lock()
        self._memory_turns: Dict[str, int] = {}          # 세션별 사용자 턴 수
        self._memory_review_pending: set = set()          # Semantic 리뷰가 예약됐지만 아직 완료되지 않은 세션

        # Procedural Memory: 사용자별 스킬 스토어 + 세션별 스킬 리뷰 트리거 상태
        self._user_procedural_stores: Dict[str, ProceduralMemoryStore] = {}
        self._procedural_unified = False
        if procedural_store is not None:
            from app.utils.workspace import resolve_user_skills_dir
            self._user_procedural_stores[_DEFAULT_USER_ID] = procedural_store
            self._procedural_unified = (
                Path(procedural_store.skills_dir).resolve()
                == resolve_user_skills_dir(_DEFAULT_USER_ID).resolve()
            )
        self._skill_lock = threading.Lock()
        self._skill_iters: Dict[str, int] = {}          # 세션별 도구 호출 반복 횟수
        self._skill_review_pending: set = set()          # 스킬 리뷰가 예약됐지만 아직 완료되지 않은 세션
        self.last_skill_reviews: Dict[str, Dict[str, Any]] = {}  # 세션별 마지막 스킬 리뷰 결과 (관찰용)

        # 백그라운드 작업 공용 스레드 풀 + 세션별 락 (Semantic 반영과 에피소드 저장이 같은 세션에서 겹치지 않도록)
        self._review_meta_lock = threading.Lock()
        self._review_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="memory-bg-review"
        )
        self._session_review_locks: Dict[str, threading.Lock] = {}
        # ① Background Review (메모리·스킬): 턴 종료 시 즉시 실행, 새 턴이 오면 cancel_event로 취소
        self._review_cancel_events: Dict[str, threading.Event] = {}
        self._review_futures: Dict[str, concurrent.futures.Future] = {}
        # ② 에피소드 추출: 세션 종료 조건까지 대기하는 큐 (취소 이벤트는 큐가 관리)
        self._episode_futures: Dict[str, concurrent.futures.Future] = {}
        self._review_queue = IdleReviewQueue(
            dispatch_fn=self._dispatch_episode_from_queue,
            settle_s=finalize_idle_settle_s,
            max_age_s=finalize_max_age_s,
        )

        self.auto_recall_episodic = auto_recall_episodic
        # 내부에서 도구 생성 (스토어 바인딩을 미들웨어가 소유)
        self._memory_tool = self._create_memory_tool()
        self._session_search_tool = self._create_session_search_tool()
        self._session_recall_tool = self._create_session_recall_tool()
        # 포그라운드 스킬 도구 (review=None → 사용자 지시로 만든 스킬은 사용자 소유)
        self._skill_tools = (
            build_skill_tools(lambda: self._get_procedural_store(self._resolve_current_user_id()))
            if procedural_store is not None else []
        )
        # AgentMiddleware.tools — LangChain create_agent()가 생성 시 한 번 읽어 에이전트 도구로 자동 주입합니다
        # (langchain/agents/factory.py의 middleware_tools 수집 → tools=와 합쳐 등록).
        self.tools = [self._memory_tool, self._session_recall_tool, *self._skill_tools]

    def get_tools(self) -> list:
        """메모리 도구 목록 반환 (create_agent 사용 시에는 self.tools를 통해 자동 등록됨).

        Returns:
            [memory_tool, session_search_tool, session_recall_tool, (skill_view, skill_manage)]
        """
        return [self._memory_tool, self._session_search_tool, self._session_recall_tool, *self._skill_tools]

    def _get_semantic_store(self, user_id: Optional[str] = None) -> SemanticMemoryStore:
        """user_id에 해당하는 SemanticMemoryStore를 반환합니다.
        - 통합 사용자 워크스페이스 모드인 경우: resolve_user_memory_dir(uid) 단일 진실 공급원(SSOT)을 사용
        - 단위 테스트 등에서 임시 디렉터리(sem_root)를 주입한 경우:
          'default'는 기본 스토어를 그대로 사용하고, 그 외 사용자는 {sem_root}/users/{user_id}/memory/ 사용
        """
        from app.utils.workspace import resolve_user_memory_dir

        uid = _sanitize_user_id(user_id or _memory_user_id_var.get())
        with self._store_lock:
            if uid in self._user_semantic_stores:
                return self._user_semantic_stores[uid]

            if self._use_unified_workspace:
                user_dir = resolve_user_memory_dir(uid, ensure_exists=True)
            else:
                user_dir = self.semantic_store.memory_dir / "users" / uid / "memory"

            user_store = SemanticMemoryStore(
                memory_dir=str(user_dir),
                memory_char_limit=self.semantic_store.memory_char_limit,
                user_char_limit=self.semantic_store.user_char_limit,
            )
            user_store.load_from_disk()
            self._user_semantic_stores[uid] = user_store
            return user_store

    def _get_episodic_store(self, user_id: Optional[str] = None):
        """단일 EpisodicStore 인스턴스(app/database/episodic.db)를 반환합니다 (내부 user_id 컬럼으로 격리)."""
        return self.episodic_store

    def _get_procedural_store(self, user_id: Optional[str] = None) -> Optional[ProceduralMemoryStore]:
        """user_id에 해당하는 ProceduralMemoryStore를 반환합니다 (procedural_store 미지정 시 None).
        - 통합 사용자 워크스페이스 모드: artifacts/users/{uid}/skills
        - 테스트용 임시 디렉터리 주입: 'default'는 기본 스토어, 그 외는 {skills_root}/users/{uid}/skills
        """
        if self.procedural_store is None:
            return None
        from app.utils.workspace import resolve_user_skills_dir

        uid = _sanitize_user_id(user_id or _memory_user_id_var.get())
        with self._store_lock:
            if uid in self._user_procedural_stores:
                return self._user_procedural_stores[uid]
            if self._procedural_unified:
                skills_dir = resolve_user_skills_dir(uid, ensure_exists=True)
            else:
                skills_dir = self.procedural_store.skills_dir / "users" / uid / "skills"
            store = ProceduralMemoryStore(
                skills_dir=str(skills_dir),
                max_skill_chars=self.procedural_store.max_skill_chars,
                max_file_bytes=self.procedural_store.max_file_bytes,
            )
            store.load_from_disk()
            self._user_procedural_stores[uid] = store
            return store

    def _resolve_current_user_id(self, runtime: Any = None) -> str:
        """runtime.context 또는 runnable config 또는 ContextVar에서 현재 user_id를 추출합니다."""
        if runtime is not None:
            ctx = getattr(runtime, "context", None)
            if ctx is not None:
                uid = getattr(ctx, "user_id", None)
                if uid and str(uid).strip():
                    return _sanitize_user_id(str(uid))
        try:
            from langchain_core.runnables.config import get_config_from_context
            cfg = get_config_from_context()
            if isinstance(cfg, dict):
                uid = (cfg.get("configurable") or {}).get("user_id")
                if uid and str(uid).strip():
                    return _sanitize_user_id(str(uid))
        except Exception:
            pass
        return _sanitize_user_id(_memory_user_id_var.get())

    def _is_transcript_enabled(self, runtime: Any = None) -> bool:
        """원문 저장소(transcript.db) 기록이 활성화되어 있는지 확인 (메모리 학습 스위치와 독립)."""
        if not self.transcript_enabled or self.transcript_store is None:
            return False
        ctx = getattr(runtime, "context", None) if runtime is not None else None
        if ctx is not None and not getattr(ctx, "transcript_enabled", True):
            return False
        return True

    def _cancel_active_review(self, session_id: str) -> None:
        """새 턴 시작: 실행 중인 Background Review를 취소하고, 에피소드 추출 대기를 보류합니다.

        - Background Review: 사용자가 다시 말을 걸면 진행 중인 리뷰를 멈춥니다
          (Hermes cancel_background_review_for_live_turn). 기준값 도달 표시는 남아 다음 턴 종료 때 다시 실행됩니다.
        - 에피소드 추출: 세션이 아직 끝나지 않았으므로 대기 슬롯을 "사용 중"으로 돌립니다. 유휴로 실행 중이던
          추출은 취소하고, max_age로 실행된 추출은 끝까지 진행합니다 (판단은 IdleReviewQueue).
        """
        if not session_id or session_id == "unknown":
            return
        with self._review_meta_lock:
            cancel_event = self._review_cancel_events.get(session_id)
        if cancel_event is not None:
            cancel_event.set()
        self._review_queue.mark_active(session_id)

    def _spawn_background_review(self, payload: Dict[str, Any]) -> None:
        """nudge 조건을 만족한 턴이 끝나면 즉시 스레드 풀로 Background Review(메모리·스킬)를 실행합니다."""
        session_id = payload["session_id"]
        cancel_event = threading.Event()
        with self._review_meta_lock:
            prev_event = self._review_cancel_events.get(session_id)
            if prev_event is not None:
                prev_event.set()   # 같은 세션의 이전 리뷰는 더 최신 대화를 보는 이번 리뷰가 대체
            self._review_cancel_events[session_id] = cancel_event
            self._session_review_locks.setdefault(session_id, threading.Lock())
            self._review_futures[session_id] = self._review_executor.submit(
                self._background_review,
                payload["messages_snapshot"],
                session_id,
                payload["user_id"],
                payload["semantic_learning"],
                payload["procedural_learning"],
                cancel_event,
                payload["transcript_enabled"],
            )
        logger.info(
            "[BG-Review] Started for session %s (semantic=%s, procedural=%s)",
            session_id, payload["semantic_learning"], payload["procedural_learning"],
        )

    def _dispatch_episode_from_queue(
        self, payload: Dict[str, Any], cancel_event: threading.Event
    ) -> None:
        """세션 종료 조건을 만족하면(IdleReviewQueue 타이머 또는 flush) 스레드 풀로 에피소드 추출을 실행합니다."""
        session_id = payload["session_id"]
        with self._review_meta_lock:
            self._session_review_locks.setdefault(session_id, threading.Lock())
            future = self._review_executor.submit(
                self._extract_episode,
                payload["messages_snapshot"],
                session_id,
                payload["user_id"],
                cancel_event,
                payload["transcript_enabled"],
            )
            self._episode_futures[session_id] = future
            future.add_done_callback(
                lambda _f, sid=session_id, ce=cancel_event: self._review_queue.finished(sid, ce)
            )

    def wait_for_reviews(self, timeout: float = 10.0) -> None:
        """대기 중인 에피소드 추출을 즉시 실행(flush)하고, 실행 중인 Background Review와 에피소드 추출이
        모두 끝날 때까지 기다립니다 (노트북·테스트·서버 종료용)."""
        self._review_queue.flush()
        with self._review_meta_lock:
            futures = [*self._review_futures.values(), *self._episode_futures.values()]
        if futures:
            concurrent.futures.wait(futures, timeout=timeout)

    # ══════════════════════════════════════════════════════════════════
    # Tool Factory
    # ══════════════════════════════════════════════════════════════════

    def _create_memory_tool(self):
        """@tool memory(action, target, content, old_text) 생성."""
        mw = self

        @langchain_tool
        def memory(
            action: str,
            target: str = "memory",
            content: str = "",
            old_text: str = "",
        ) -> str:
            """Manage the agent's long-term semantic memory (MEMORY.md / USER.md).

            Two stores:
            - target="memory": Agent's personal notes (environment facts, project conventions, tool quirks).
            - target="user": What the agent knows about the user (preferences, communication style).

            Actions:
            - action="add": Append a new entry. Content must be concise English.
            - action="replace": Find entry containing old_text, replace with content.
            - action="remove": Find entry containing old_text, delete it.

            Guidelines:
            - Write entries in English for search efficiency.
            - Keep entries short (1-2 sentences). Merge overlapping facts.
            - Do NOT store task-specific data, logs, or temporary state.
            - Current MEMORY.md and USER.md contents for the active user are ALREADY preloaded in your session memory snapshot (`# memorySnapshot` in `<system-reminder>`). Do NOT use file_read, glob_search, or bash_command to search for USER.md or MEMORY.md on disk.

            Args:
                action: "add" | "replace" | "remove"
                target: "memory" (MEMORY.md) | "user" (USER.md)
                content: New entry text (required for add/replace)
                old_text: Unique substring to match existing entry (required for replace/remove)

            Returns:
                JSON string with success status, usage info, and current entry count.
            """
            uid = mw._resolve_current_user_id()
            store = mw._get_semantic_store(uid)
            if action == "add":
                result = store.add(target, content)
            elif action == "replace":
                result = store.replace(target, old_text, content)
            elif action == "remove":
                result = store.remove(target, old_text)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown action '{action}'. Use add, replace, or remove.",
                }
            return json.dumps(result, ensure_ascii=False, indent=2)

        return memory

    def _create_session_search_tool(self):
        """@tool session_search(query, limit) 생성 — 1단계 세션 탐색 및 브라우즈."""
        mw = self

        @langchain_tool
        def session_search(
            query: str = "",
            limit: int = 3,
        ) -> str:
            """Search past conversation sessions to recall relevant context.

            Two modes (inferred from arguments):
            1. DISCOVERY: Pass a query string to search for related past sessions via FTS5.
               Returns session summaries, keywords, and match snippets.
            2. BROWSE: Call with no arguments (or empty query) to list recent sessions chronologically.

            Args:
                query: Search query in English (empty = browse recent sessions)
                limit: Max sessions to return (default 3)

            Returns:
                JSON with session list: [{session_id, summary, keywords, snippet}, ...]
            """
            uid = mw._resolve_current_user_id()
            epi_store = mw._get_episodic_store(uid)
            if query.strip():
                results = _run_async_safe(
                    epi_store.search_sessions(query, top_k=limit, user_id=uid)
                )
                # episodic.db 요약에 아직 반영되지 않았거나 정확한 식별자가 원문(transcript.db)에만 있는 경우 보충
                if len(results) < limit and mw.transcript_store is not None:
                    seen_sids = {r.get("session_id") for r in results}
                    raw_hits = mw.transcript_store.search_sync(
                        query=query, top_k=limit, user_id=uid
                    )
                    for hit in raw_hits:
                        sid = hit.get("session_id")
                        if sid and sid not in seen_sids:
                            seen_sids.add(sid)
                            results.append({
                                "session_id": sid,
                                "summary": f"(Transcript match) {hit.get('snippet', '')}",
                                "topics": [],
                                "keywords": [],
                                "tool_names": [],
                                "snippet": hit.get("snippet", ""),
                            })
                            if len(results) >= limit:
                                break
            else:
                results = _run_async_safe(
                    epi_store.browse_recent(limit=limit, user_id=uid)
                )

            if not results:
                return json.dumps({
                    "results": [],
                    "message": "No matching sessions found." if query else "No past sessions recorded.",
                })

            return json.dumps({
                "query": query or "(browse recent)",
                "results": results,
                "count": len(results),
            }, ensure_ascii=False, indent=2)

        return session_search

    def _create_session_recall_tool(self):
        """@tool session_recall(session_id, anchor_message, window) 생성."""
        mw = self

        def _format_recall_result(
            session_id: str, anchor_message: str, messages: List[Dict[str, Any]]
        ) -> str:
            if not messages:
                return json.dumps({
                    "session_id": session_id,
                    "messages": [],
                    "message": f"No messages found for session '{session_id}'.",
                })
            max_chars = (
                mw.transcript_store.recall_max_chars
                if mw.transcript_store is not None
                else 2000
            )
            formatted_msgs = []
            for m in messages:
                # transcript.db 결과는 이미 키워드 주변으로 잘려 있음(잘림 안내 문구 포함) → 여유분을 두고 상한만 적용
                item: Dict[str, Any] = {
                    "role": m.get("role", "unknown"),
                    "content": (m.get("content") or "")[:max_chars + 200],
                }
                if m.get("tool_calls"):
                    item["tool_calls"] = m["tool_calls"]
                if m.get("tool_name"):
                    item["tool_name"] = m["tool_name"]
                formatted_msgs.append(item)
            return json.dumps({
                "session_id": session_id,
                "anchor": anchor_message or "(tail view)",
                "message_count": len(formatted_msgs),
                "messages": formatted_msgs,
            }, ensure_ascii=False, indent=2)

        @langchain_tool
        def session_recall(
            session_id: str,
            anchor_message: str = "",
            window: int = 5,
        ) -> str:
            """Recall messages from a specific past session using anchor-based retrieval.

            Anchor-based retrieval (Hermes Lineage pattern):
            1. Verifies that the target session belongs to the current user (user_id isolation)
            2. Finds the message best matching anchor_message keywords in the full session transcript
            3. Returns ±window messages around the anchor (including tool calls and outputs)
            4. Always includes bookends (first 3 + last 3 messages of the session)

            If anchor_message is empty, returns the last `window` messages of the session.

            Use this tool after seeing a relevant session hint in EPISODIC MEMORY (`# episodicMemory` in `<system-reminder>`).

            Args:
                session_id: The session ID to recall messages from
                anchor_message: Keywords to locate the most relevant message (optional)
                window: Number of messages before/after the anchor (default 5)

            Returns:
                JSON with messages: [{role, content, tool_calls?, tool_name?}, ...]
            """
            uid = mw._resolve_current_user_id()
            epi_store = mw._get_episodic_store(uid)
            messages = _run_async_safe(
                epi_store.get_anchored_view(
                    session_id=session_id,
                    anchor_keyword=anchor_message,
                    window=window,
                    user_id=uid,
                )
            )
            return _format_recall_result(session_id, anchor_message, messages)

        async def _asession_recall(
            session_id: str,
            anchor_message: str = "",
            window: int = 5,
        ) -> str:
            uid = mw._resolve_current_user_id()
            epi_store = mw._get_episodic_store(uid)
            messages = await epi_store.get_anchored_view(
                session_id=session_id,
                anchor_keyword=anchor_message,
                window=window,
                user_id=uid,
            )
            return _format_recall_result(session_id, anchor_message, messages)

        session_recall.coroutine = _asession_recall
        return session_recall

    # ══════════════════════════════════════════════════════════════════
    # 도구 컨텍스트 전파, 게이팅 및 실시간 원문(transcript.db) 기록
    # ══════════════════════════════════════════════════════════════════

    def _disabled_tool_names(self, runtime: Any) -> set:
        """명시적으로 꺼진 메모리 기능의 도구 이름."""
        ctx = getattr(runtime, "context", None)
        if not ctx:
            return set()
        disabled = set()
        if not getattr(ctx, "semantic_memory_enabled", True):
            disabled.add(self._memory_tool.name)
        if not getattr(ctx, "episodic_memory_enabled", True):
            disabled.add(self._session_recall_tool.name)
        if not getattr(ctx, "procedural_memory_enabled", True):
            disabled.update(t.name for t in self._skill_tools)
        return disabled

    # ── Semantic Memory 트리거 (Hermes _turns_since_memory 패턴) ──

    def _count_memory_turn(self, session_id: str, messages: List[Any]) -> None:
        """사용자 턴 1회 = 1. 처음 보는 세션(서버 재시작 등)은 이전 사용자 턴 수 % 기준값에서 시작합니다."""
        if self.memory_nudge_interval <= 0 or session_id == "unknown":
            return
        with self._memory_lock:
            if session_id not in self._memory_turns:
                prior = sum(1 for m in messages if getattr(m, "type", None) == "human") - 1
                self._memory_turns[session_id] = max(0, prior) % self.memory_nudge_interval
            self._memory_turns[session_id] += 1

    def _memory_nudge_reached(self, session_id: str) -> bool:
        """nudge 조건 (Semantic): 사용자 턴 카운터가 기준값에 닿았거나, 닿은 뒤 리뷰가 아직 끝나지 않았는가."""
        if self.memory_nudge_interval <= 0:
            return False
        with self._memory_lock:
            return (
                self._memory_turns.get(session_id, 0) >= self.memory_nudge_interval
                or session_id in self._memory_review_pending
            )

    def _finish_memory_review(self, session_id: str) -> None:
        """Semantic 리뷰가 취소되지 않고 끝났을 때만 카운터를 리셋합니다 (취소되면 다음 리뷰로 이월)."""
        with self._memory_lock:
            self._memory_review_pending.discard(session_id)
            self._memory_turns[session_id] = 0

    def _reset_memory_counter_on_tool(self, request: Any) -> None:
        """에이전트가 대화 중 memory 도구를 직접 호출했다면 이미 기록한 것이므로 카운터를 리셋합니다."""
        if request.tool_call.get("name") != self._memory_tool.name:
            return
        session_id = self._get_session_id(getattr(request, "runtime", None), request)
        with self._memory_lock:
            self._memory_turns[session_id] = 0

    # ── Procedural Memory 트리거 (Hermes _iters_since_skill 패턴) ──

    def _count_skill_iteration(self, runtime: Any, request: Any, out_msgs: List[BaseMessage]) -> None:
        """도구를 호출한 모델 응답 1회 = 반복 1회. 병렬 도구 호출은 한 번으로 셉니다."""
        if self.procedural_store is None or self.skill_nudge_interval <= 0:
            return
        if not any(isinstance(m, AIMessage) and getattr(m, "tool_calls", None) for m in out_msgs):
            return
        session_id = self._get_session_id(runtime, request)
        if session_id == "unknown":
            return
        with self._skill_lock:
            self._skill_iters[session_id] = self._skill_iters.get(session_id, 0) + 1

    def _reset_skill_counter_on_manage(self, request: Any) -> None:
        """에이전트가 스스로 skill_manage를 호출했다면 이미 학습한 것이므로 카운터를 리셋합니다."""
        if request.tool_call.get("name") != SKILL_MANAGE_TOOL:
            return
        session_id = self._get_session_id(getattr(request, "runtime", None), request)
        with self._skill_lock:
            self._skill_iters[session_id] = 0

    def _skill_nudge_reached(self, session_id: str) -> bool:
        """nudge 조건 (Procedural): 도구 호출 응답 카운터가 기준값에 닿았거나, 닿은 뒤 리뷰가 아직 끝나지 않았는가."""
        if self.procedural_store is None or self.skill_nudge_interval <= 0:
            return False
        with self._skill_lock:
            return (
                self._skill_iters.get(session_id, 0) >= self.skill_nudge_interval
                or session_id in self._skill_review_pending
            )

    def _background_review_due(self, ctx: Any, session_id: str, base_learning: bool) -> Dict[str, bool]:
        """이번 턴 종료 시 Background Review로 무엇을 리뷰할지 결정합니다 (nudge 조건 + 요청별 스위치).

        조건을 만족한 세션은 pending에 올려, 리뷰가 새 턴 때문에 취소되더라도 다음 턴 종료 때 다시 실행되게 합니다.
        pending은 리뷰가 끝까지 수행된 뒤 _finish_memory_review / _finish_skill_review가 비웁니다.
        """
        due = {"semantic": False, "procedural": False}
        if not self.review_llm:
            return due
        semantic_on = getattr(ctx, "semantic_memory_enabled", True) and getattr(
            ctx, "semantic_learning_enabled", base_learning
        )
        if semantic_on and self._memory_nudge_reached(session_id):
            with self._memory_lock:
                self._memory_review_pending.add(session_id)
            due["semantic"] = True
        procedural_on = getattr(ctx, "procedural_memory_enabled", True) and getattr(
            ctx, "procedural_learning_enabled", base_learning
        )
        if procedural_on and self._skill_nudge_reached(session_id):
            with self._skill_lock:
                self._skill_review_pending.add(session_id)
            due["procedural"] = True
        return due

    @staticmethod
    def _episode_extraction_enabled(ctx: Any, base_learning: bool) -> bool:
        """에피소드 추출을 예약할지 (요청별 스위치). 실행 시점은 세션 종료 조건이 정합니다."""
        return bool(
            getattr(ctx, "episodic_memory_enabled", True)
            and getattr(ctx, "episodic_learning_enabled", base_learning)
        )

    def _finish_skill_review(self, session_id: str) -> None:
        """스킬 리뷰가 취소되지 않고 끝났을 때만 카운터를 리셋합니다 (취소되면 다음 리뷰로 이월)."""
        with self._skill_lock:
            self._skill_review_pending.discard(session_id)
            self._skill_iters[session_id] = 0

    def _filter_tools(self, request):
        disabled = self._disabled_tool_names(request.runtime)
        if not disabled:
            return request
        tools = [t for t in request.tools if getattr(t, "name", None) not in disabled]
        return request.override(tools=tools)

    def wrap_model_call(self, request, handler):
        response = handler(self._filter_tools(request))
        out_msgs = _extract_response_messages(response)
        for m in out_msgs:
            ensure_message_id(m)
        runtime = getattr(request, "runtime", None)
        self._count_skill_iteration(runtime, request, out_msgs)
        if out_msgs and self._is_transcript_enabled(runtime) and self.transcript_store is not None:
            session_id = self._get_session_id(runtime, request)
            if session_id != "unknown":
                user_id = self._resolve_current_user_id(runtime)
                turn_id = self.transcript_store.current_turn(session_id)
                try:
                    self.transcript_store.append(
                        session_id=session_id,
                        user_id=user_id,
                        messages=out_msgs,
                        turn_id=turn_id,
                    )
                except Exception as e:
                    logger.warning("[MemoryMiddleware] Transcript append on model call failed: %s", e)
        return response

    async def awrap_model_call(self, request, handler):
        response = await handler(self._filter_tools(request))
        out_msgs = _extract_response_messages(response)
        for m in out_msgs:
            ensure_message_id(m)
        runtime = getattr(request, "runtime", None)
        self._count_skill_iteration(runtime, request, out_msgs)
        if out_msgs and self._is_transcript_enabled(runtime) and self.transcript_store is not None:
            session_id = self._get_session_id(runtime, request)
            if session_id != "unknown":
                user_id = self._resolve_current_user_id(runtime)
                turn_id = self.transcript_store.current_turn(session_id)
                try:
                    await self.transcript_store.aappend(
                        session_id=session_id,
                        user_id=user_id,
                        messages=out_msgs,
                        turn_id=turn_id,
                    )
                except Exception as e:
                    logger.warning("[MemoryMiddleware] Transcript aappend on model call failed: %s", e)
        return response

    def _blocked_tool_message(self, request) -> Optional[ToolMessage]:
        name = request.tool_call.get("name")
        if name not in self._disabled_tool_names(request.runtime):
            return None
        return ToolMessage(
            content=f"'{name}' 도구는 현재 대화에서 메모리 기능이 꺼져 있어 사용할 수 없습니다.",
            tool_call_id=request.tool_call.get("id", ""),
            name=name,
            status="error",
        )

    def wrap_tool_call(self, request, handler):
        runtime = getattr(request, "runtime", None)
        blocked = self._blocked_tool_message(request)
        if blocked:
            ensure_message_id(blocked)
            if self._is_transcript_enabled(runtime) and self.transcript_store is not None:
                session_id = self._get_session_id(runtime, request)
                if session_id != "unknown":
                    user_id = self._resolve_current_user_id(runtime)
                    turn_id = self.transcript_store.current_turn(session_id)
                    try:
                        self.transcript_store.append(
                            session_id=session_id,
                            user_id=user_id,
                            messages=[blocked],
                            turn_id=turn_id,
                        )
                    except Exception:
                        pass
            return blocked

        self._reset_memory_counter_on_tool(request)
        self._reset_skill_counter_on_manage(request)
        uid = self._resolve_current_user_id(runtime)
        tok = _memory_user_id_var.set(uid)
        try:
            result = handler(request)
        finally:
            _memory_user_id_var.reset(tok)

        if isinstance(result, BaseMessage):
            ensure_message_id(result)
            if self._is_transcript_enabled(runtime) and self.transcript_store is not None:
                session_id = self._get_session_id(runtime, request)
                if session_id != "unknown":
                    turn_id = self.transcript_store.current_turn(session_id)
                    try:
                        self.transcript_store.append(
                            session_id=session_id,
                            user_id=uid,
                            messages=[result],
                            turn_id=turn_id,
                        )
                    except Exception as e:
                        logger.warning("[MemoryMiddleware] Transcript append on tool call failed: %s", e)
        return result

    async def awrap_tool_call(self, request, handler):
        runtime = getattr(request, "runtime", None)
        blocked = self._blocked_tool_message(request)
        if blocked:
            ensure_message_id(blocked)
            if self._is_transcript_enabled(runtime) and self.transcript_store is not None:
                session_id = self._get_session_id(runtime, request)
                if session_id != "unknown":
                    user_id = self._resolve_current_user_id(runtime)
                    turn_id = self.transcript_store.current_turn(session_id)
                    try:
                        await self.transcript_store.aappend(
                            session_id=session_id,
                            user_id=user_id,
                            messages=[blocked],
                            turn_id=turn_id,
                        )
                    except Exception:
                        pass
            return blocked

        self._reset_memory_counter_on_tool(request)
        self._reset_skill_counter_on_manage(request)
        uid = self._resolve_current_user_id(runtime)
        tok = _memory_user_id_var.set(uid)
        try:
            result = await handler(request)
        finally:
            _memory_user_id_var.reset(tok)

        if isinstance(result, BaseMessage):
            ensure_message_id(result)
            if self._is_transcript_enabled(runtime) and self.transcript_store is not None:
                session_id = self._get_session_id(runtime, request)
                if session_id != "unknown":
                    turn_id = self.transcript_store.current_turn(session_id)
                    try:
                        await self.transcript_store.aappend(
                            session_id=session_id,
                            user_id=uid,
                            messages=[result],
                            turn_id=turn_id,
                        )
                    except Exception as e:
                        logger.warning("[MemoryMiddleware] Transcript aappend on tool call failed: %s", e)
        return result

    # ══════════════════════════════════════════════════════════════════
    # AgentMiddleware Hooks
    # ══════════════════════════════════════════════════════════════════

    def _record_turn_start(self, state: Dict[str, Any], runtime: Any) -> tuple[str, str]:
        """턴 시작 시 세션 유휴 리뷰를 취소하고 사용자 입력 메시지를 transcript.db에 즉시 기록합니다."""
        session_id = self._get_session_id(runtime)
        user_id = self._resolve_current_user_id(runtime)
        _memory_user_id_var.set(user_id)

        # 같은 세션에서 새 턴이 시작되면 대기/진행 중이던 이전 턴 Background Review를 즉시 취소
        self._cancel_active_review(session_id)

        messages = state.get("messages", [])
        for m in messages:
            if isinstance(m, BaseMessage):
                ensure_message_id(m)

        if (
            session_id != "unknown"
            and messages
            and self._is_transcript_enabled(runtime)
            and self.transcript_store is not None
        ):
            turn_id = self.transcript_store.begin_turn(session_id)
            try:
                self.transcript_store.append(
                    session_id=session_id,
                    user_id=user_id,
                    messages=messages,
                    turn_id=turn_id,
                )
            except Exception as e:
                logger.warning("[MemoryMiddleware] Transcript append on before_agent failed: %s", e)

        return session_id, user_id

    def _session_snapshot(self, user_id: str, session_id: str, kind: str, capture) -> Any:
        """세션별 Frozen Snapshot. 그 세션에서 처음 요청될 때만 capture()로 디스크를 읽어 렌더링합니다.

        스냅샷 키가 (user_id, session_id)이므로 같은 사용자가 세션 A → B → A로 오가도
        A로 돌아왔을 때 A의 첫 턴 스냅샷을 그대로 씁니다 (세션 중 프롬프트 불변 → 프리픽스 캐시 보존).
        """
        if session_id == "unknown":
            return capture()   # 세션을 알 수 없으면 고정할 단위가 없으므로 매번 새로 읽음
        key = (user_id, session_id)
        with self._store_lock:
            entry = self._prompt_snapshots.get(key)
            if entry is None:
                entry = self._prompt_snapshots[key] = {}
                while len(self._prompt_snapshots) > self._max_prompt_snapshots:
                    self._prompt_snapshots.popitem(last=False)   # 가장 오래 안 쓴 세션부터 정리
            self._prompt_snapshots.move_to_end(key)
            if kind not in entry:
                entry[kind] = capture()
            return entry[kind]

    def _prepare_turn_and_semantic_memory(
        self, state: Dict[str, Any], runtime: Any
    ) -> Optional[tuple[Any, str, str, List[str]]]:
        """턴 시작 시 원문 기록, 이전 리뷰 취소 및 L3 Semantic Memory 프리페치를 수행합니다."""
        session_id, user_id = self._record_turn_start(state, runtime)
        ctx = getattr(runtime, "context", None)
        if not ctx:
            return None

        if session_id != "unknown":
            ctx.session_id = session_id

        recalled_parts: List[str] = []

        # L3: Semantic Memory 프리페치 (세션별 Frozen Snapshot, user_id 격리)
        if getattr(ctx, "semantic_memory_enabled", True):
            self._count_memory_turn(session_id, state.get("messages", []))
            store = self._get_semantic_store(user_id)

            def _capture_semantic():
                store.load_from_disk()   # 세션의 첫 턴에만 디스크에서 최신 마크다운 로드
                logger.info(
                    "[MemoryMiddleware] Captured semantic memory snapshot (user=%s, session=%s).",
                    user_id, session_id,
                )
                return store.format_for_prompt("memory"), store.format_for_prompt("user")

            memory_block, user_block = self._session_snapshot(user_id, session_id, "semantic", _capture_semantic)
            mem_limit = store.memory_char_limit
            usr_limit = store.user_char_limit
            if memory_block:
                recalled_parts.append(memory_block)
            else:
                recalled_parts.append(
                    "══════════════════════════════════════════════\n"
                    f"MEMORY (agent's personal notes) [0% — 0/{mem_limit:,} chars]\n"
                    "══════════════════════════════════════════════\n"
                    "(No entries recorded yet in MEMORY.md for this user.)"
                )
            if user_block:
                recalled_parts.append(user_block)
            else:
                recalled_parts.append(
                    "══════════════════════════════════════════════\n"
                    f"USER PROFILE (who the user is) [0% — 0/{usr_limit:,} chars]\n"
                    "══════════════════════════════════════════════\n"
                    "(No entries recorded yet in USER.md for this user.)"
                )

        # Procedural: 학습 스킬 카탈로그 프리페치 (이름 + 설명만, 세션별 Frozen Snapshot)
        if self.procedural_store is not None and getattr(ctx, "procedural_memory_enabled", True):
            pstore = self._get_procedural_store(user_id)

            def _capture_skills():
                pstore.load_from_disk()
                return pstore.format_for_prompt()

            skills_block = self._session_snapshot(user_id, session_id, "skills", _capture_skills)
            recalled_parts.append(skills_block or (
                "══════════════════════════════════════════════\n"
                "PROCEDURAL MEMORY (learned skills) [0 skills]\n"
                "══════════════════════════════════════════════\n"
                "(No learned skills yet. After a difficult, multi-step task, offer to save the approach "
                "as a skill with skill_manage.)"
            ))

        return ctx, session_id, user_id, recalled_parts

    @staticmethod
    def _assign_recalled_context(ctx: Any, snapshot_parts: List[str], episodic_block: str = "") -> None:
        """세션 동결 스냅샷(memory_snapshot)과 턴별 에피소드 기억(episodic_context)을 분리 기록하고,
        단독 @dynamic_prompt 하위 호환용 합본(recalled_memory)도 함께 채웁니다."""
        snapshot_text = "\n\n".join(snapshot_parts) if snapshot_parts else ""
        if hasattr(ctx, "memory_snapshot"):
            ctx.memory_snapshot = snapshot_text
        if hasattr(ctx, "episodic_context"):
            ctx.episodic_context = episodic_block
        combined = [*snapshot_parts, episodic_block] if episodic_block else list(snapshot_parts)
        ctx.recalled_memory = "\n\n".join(combined) if combined else ""

    def before_agent(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        """동기 실행 경로용 원문 기록 + L2+L3 메모리 인출 → memory_snapshot / episodic_context / recalled_memory 주입."""
        prepared = self._prepare_turn_and_semantic_memory(state, runtime)
        if not prepared:
            return None
        ctx, session_id, user_id, recalled_parts = prepared
        episodic_block = ""

        # L2: Episodic Memory 자동 인출 (동기 호출자용 _run_async_safe 경로)
        if getattr(ctx, "episodic_memory_enabled", True) and self.top_k > 0:
            user_query = self._extract_last_user_query(state)
            if user_query:
                try:
                    epi_store = self._get_episodic_store(user_id)
                    sessions = _run_async_safe(
                        epi_store.search_sessions(
                            user_query,
                            top_k=self.top_k,
                            exclude_session_id=session_id,
                            user_id=user_id,
                        )
                    )
                    if sessions:
                        episodic_block = self._format_episodic(sessions)
                except Exception as e:
                    logger.warning("Episodic memory retrieval failed: %s", e)

        self._assign_recalled_context(ctx, recalled_parts, episodic_block)
        return None

    async def abefore_agent(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        """비동기 실행 경로(FastAPI ainvoke/astream_events)용 네이티브 논블로킹 원문 기록 + L2+L3 메모리 인출."""
        prepared = await asyncio.to_thread(self._prepare_turn_and_semantic_memory, state, runtime)
        if not prepared:
            return None
        ctx, session_id, user_id, recalled_parts = prepared
        _memory_user_id_var.set(user_id)
        episodic_block = ""

        # L2: Episodic Memory 자동 인출 (이벤트 루프를 블로킹하지 않는 네이티브 await 경로)
        if getattr(ctx, "episodic_memory_enabled", True) and self.top_k > 0:
            user_query = self._extract_last_user_query(state)
            if user_query:
                try:
                    epi_store = self._get_episodic_store(user_id)
                    sessions = await epi_store.search_sessions(
                        user_query,
                        top_k=self.top_k,
                        exclude_session_id=session_id,
                        user_id=user_id,
                    )
                    if sessions:
                        episodic_block = self._format_episodic(sessions)
                except Exception as e:
                    logger.warning("Episodic memory retrieval failed: %s", e)

        self._assign_recalled_context(ctx, recalled_parts, episodic_block)
        return None

    def after_agent(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        """턴 종료 시 누락 메시지 원문 보정 기록 + Background Review 실행 + 에피소드 추출 예약.

        1. wrap 훅을 거치지 않은 메시지(예: 가드레일 조기 차단 AIMessage 또는 테스트 직접 호출)를
           transcript.db에 멱등(idempotent) 보정 기록합니다.
        2. 현재 턴이 가드레일에 의해 차단(guardrail_blocked=True)된 경우 해당 턴 전체를
           kind='blocked'로 마킹하고 메모리 학습/요약을 건너뜁니다.
        3. 정상 턴이면 두 작업을 따로 판단합니다.
           ① Background Review: nudge 조건(_background_review_due)을 만족하면 지금 바로 실행
           ② 에피소드 추출: IdleReviewQueue에 예약 → 세션 종료 조건(기본 유휴 15초 / 최대 30분)에 실행
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        for m in messages:
            if isinstance(m, BaseMessage):
                ensure_message_id(m)

        session_id = self._get_session_id(runtime)
        user_id = self._resolve_current_user_id(runtime)
        turn_blocked = is_blocked_message(messages[-1])
        transcript_on = self._is_transcript_enabled(runtime)

        if session_id != "unknown" and transcript_on and self.transcript_store is not None:
            try:
                if turn_blocked:
                    # 가드레일이 before_agent에서 턴을 끝내면 이 미들웨어의 before_agent가 실행되지 않아
                    # 새 턴 번호가 없습니다. 차단된 질문·응답만 골라 별도 턴으로 기록합니다.
                    start = last_user_turn_start(messages)
                    start = start if start is not None else len(messages) - 1
                    self.transcript_store.append(
                        session_id=session_id,
                        user_id=user_id,
                        messages=messages[:start],
                        turn_id=self.transcript_store.current_turn(session_id),
                    )
                    self.transcript_store.record_blocked_exchange(session_id, user_id, messages[start:])
                else:
                    self.transcript_store.append(
                        session_id=session_id,
                        user_id=user_id,
                        messages=messages,
                        turn_id=self.transcript_store.current_turn(session_id),
                    )
            except Exception as e:
                logger.warning("[MemoryMiddleware] Transcript append on after_agent failed: %s", e)

        if turn_blocked or len(messages) < 2:
            return None

        ctx = getattr(runtime, "context", None)
        if not ctx:
            return None

        base_learning = getattr(ctx, "memory_learning_enabled", True)
        review_due = self._background_review_due(ctx, session_id, base_learning)
        extract_episode = self._episode_extraction_enabled(ctx, base_learning)
        if not (review_due["semantic"] or review_due["procedural"] or extract_episode):
            return None

        # 대화 스냅샷 복제 (transcript.db 비활성화 시 폴백용)
        # 원문 기록이 꺼진 턴이면 두 작업 모두 원문 저장소를 읽거나 쓰지 않고 이 스냅샷만 사용합니다.
        base_payload = {
            "messages_snapshot": [_serialize_message_for_review(m) for m in messages],
            "session_id": session_id,
            "user_id": user_id,
            "transcript_enabled": transcript_on,
        }

        # ① Background Review (메모리·스킬): nudge 조건을 만족했으면 지금 바로 실행 (새 턴이 오면 취소)
        if review_due["semantic"] or review_due["procedural"]:
            self._spawn_background_review({
                **base_payload,
                "semantic_learning": review_due["semantic"],
                "procedural_learning": review_due["procedural"],
            })

        # ② 에피소드 추출: 세션 종료 조건까지 대기 (세션당 최신 예약만 유지)
        if extract_episode:
            self._review_queue.schedule(session_id=session_id, user_id=user_id, payload=base_payload)
            logger.info("Episode extraction queued for session %s (user=%s)", session_id, user_id)
        return None

    async def aafter_agent(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self.after_agent, state, runtime)

    # ══════════════════════════════════════════════════════════════════
    # 백그라운드 작업: ① Background Review (메모리·스킬)  ② 에피소드 추출
    # ══════════════════════════════════════════════════════════════════

    def _load_session_messages(
        self, messages_snapshot: List[Dict[str, Any]], session_id: str, user_id: str, transcript_enabled: bool
    ) -> List[Dict[str, Any]]:
        """원문 기록이 켜져 있으면 transcript.db의 세션 전체 원문(압축 손실 없음), 아니면 턴 종료 시점 스냅샷."""
        if transcript_enabled and self.transcript_store is not None and session_id and session_id != "unknown":
            try:
                rows = self.transcript_store.read_session_sync(
                    session_id=session_id, user_id=user_id, include_hidden=False,
                )
                if rows:
                    return rows
            except Exception as e:
                logger.warning("[BG-Review] Failed to read transcript for %s: %s", session_id, e)
        return messages_snapshot

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._review_meta_lock:
            return self._session_review_locks.setdefault(session_id, threading.Lock())

    def _background_review(
        self,
        messages_snapshot: List[Dict[str, Any]],
        session_id: str,
        user_id: str = _DEFAULT_USER_ID,
        semantic_learning: bool = True,
        procedural_learning: bool = False,
        cancel_event: Optional[threading.Event] = None,
        transcript_enabled: bool = True,
    ) -> None:
        """Background Review 워커 (Hermes agent/background_review.py 대응) — nudge 조건을 만족한 턴 종료 직후 실행.

        1. semantic_learning: LLM에게 대화 리뷰 요청 → memory() 호출 파싱 실행 (add만 반영, 수정·삭제는 제안)
        2. procedural_learning: 스킬 도구만 가진 리뷰 에이전트가 스킬 생성/패치 (skill_review.py)
        같은 세션에서 새 턴이 시작되어 cancel_event가 설정되면 중단하고, 기준값 도달 표시는 남겨 다음 턴에 다시 실행합니다.
        """
        if cancel_event is not None and cancel_event.is_set():
            logger.info("[BG-Review] Skipped cancelled review for session %s", session_id)
            return
        try:
            full_messages = self._load_session_messages(messages_snapshot, session_id, user_id, transcript_enabled)

            if semantic_learning and self.review_llm:
                if self._review_semantic_memory(
                    full_messages, user_id=user_id, cancel_event=cancel_event,
                    session_lock=self._session_lock(session_id),
                ):
                    self._finish_memory_review(session_id)

            if procedural_learning and self.review_llm:
                if cancel_event is not None and cancel_event.is_set():
                    return
                self._review_procedural_memory(
                    full_messages, session_id, user_id=user_id, cancel_event=cancel_event
                )

            logger.info("[BG-Review] Completed for session %s", session_id)
        except Exception as e:
            logger.error("[BG-Review] Failed for session %s: %s", session_id, e)

    def _extract_episode(
        self,
        messages_snapshot: List[Dict[str, Any]],
        session_id: str,
        user_id: str = _DEFAULT_USER_ID,
        cancel_event: Optional[threading.Event] = None,
        transcript_enabled: bool = True,
    ) -> None:
        """에피소드 추출 워커 (Hermes MemoryProvider.on_session_end 대응) — 세션 종료 조건을 만족하면 실행.

        세션 원문을 LLM으로 주제별 요약해 EpisodicStore(episodic.db)에 색인합니다.
        유휴로 실행된 추출은 새 턴이 시작되면(cancel_event) 중단합니다.
        """
        if cancel_event is not None and cancel_event.is_set():
            logger.info("[Episode] Skipped cancelled extraction for session %s", session_id)
            return
        try:
            full_messages = self._load_session_messages(messages_snapshot, session_id, user_id, transcript_enabled)
            self._review_episodic_memory(
                full_messages, session_id, user_id=user_id, cancel_event=cancel_event,
                session_lock=self._session_lock(session_id),
            )
            logger.info("[Episode] Extracted episode for session %s", session_id)
        except Exception as e:
            logger.error("[Episode] Failed for session %s: %s", session_id, e)

    def _review_semantic_memory(
        self,
        messages: List[Dict[str, Any]],
        user_id: str = _DEFAULT_USER_ID,
        cancel_event: Optional[threading.Event] = None,
        session_lock: Optional[threading.Lock] = None,
    ) -> bool:
        """LLM으로 대화를 리뷰하여 사용자별 MEMORY.md/USER.md 갱신.

        Returns:
            리뷰가 끝까지 수행되었으면 True (취소·실패 시 False → 트리거 카운터를 리셋하지 않음)
        """
        if not self.review_llm:
            return False
        if cancel_event is not None and cancel_event.is_set():
            return False

        store = self._get_semantic_store(user_id)

        # 현재 엔트리를 프롬프트에 포함
        memory_entries = "\n".join(
            f"- {e}" for e in store.memory_entries
        ) or "(empty)"
        user_entries = "\n".join(
            f"- {e}" for e in store.user_entries
        ) or "(empty)"

        # 대화 텍스트 구성 (원문 및 도구 호출 내역 포함)
        conversation_text = render_transcript_for_summary(messages, max_chars=4000)
        if not conversation_text:
            conversation_text = "\n".join(
                f"{m.get('role', 'unknown')}: {m.get('content', '')}"
                for m in messages
                if m.get("content") and m.get("role") != "tool"
            )[:4000]

        review_prompt = (
            f"Conversation:\n{conversation_text}\n\n"
            + COMBINED_REVIEW_PROMPT.format(
                memory_entries=memory_entries,
                user_entries=user_entries,
            )
        )

        try:
            response = self.review_llm.invoke(review_prompt)
            if cancel_event is not None and cancel_event.is_set():
                logger.info("[BG-Review] Cancelled stale semantic review before applying ops (user=%s)", user_id)
                return False

            content = normalize_content(getattr(response, "content", response))

            # LLM 응답에서 memory() 호출 파싱 및 실행 (세션 락 하에서 취소 재확인 후 반영)
            if session_lock is not None:
                with session_lock:
                    if cancel_event is not None and cancel_event.is_set():
                        return False
                    self._parse_and_execute_memory_ops(content, store=store)
            else:
                self._parse_and_execute_memory_ops(content, store=store)
            return True

        except Exception as e:
            logger.warning("[BG-Review] Semantic review LLM call failed: %s", e)
            return False

    def _review_procedural_memory(
        self,
        messages: List[Dict[str, Any]],
        session_id: str,
        user_id: str = _DEFAULT_USER_ID,
        cancel_event: Optional[threading.Event] = None,
    ) -> Optional[Dict[str, Any]]:
        """스킬 리뷰 에이전트를 실행합니다. 취소되지 않고 끝났을 때만 트리거 카운터를 리셋합니다."""
        store = self._get_procedural_store(user_id)
        if store is None:
            return None
        try:
            result = run_skill_review(
                self.review_llm,
                store,
                messages,
                cancel_event=cancel_event,
                max_iterations=self.skill_review_max_iterations,
            )
        except Exception as e:
            logger.warning("[BG-Review] Skill review failed: %s", e)
            return None
        self.last_skill_reviews[session_id] = result
        if not result.get("cancelled"):
            self._finish_skill_review(session_id)
        return result

    def _parse_and_execute_memory_ops(
        self, llm_response: str, store: Optional[SemanticMemoryStore] = None
    ) -> None:
        """LLM 응답에서 memory 작업을 파싱하여 실행.

        지원 형식 (모델마다 답하는 모양이 다름):
        - 함수 호출 형태: memory(action="add", target="user", content="...")
        - XML 태그 형태: <call:memory action="add" ... />
        - JSON 형태: [{"action": "add", "target": "user", "content": "..."}] (```json 코드 펜스 포함)
        """
        target_store = store or self.semantic_store

        json_ops = self._parse_json_memory_ops(llm_response)
        if json_ops is not None:
            for op in json_ops:
                self._apply_memory_op(
                    target_store,
                    action=str(op.get("action") or ""),
                    target=str(op.get("target") or "memory"),
                    content=str(op.get("content") or ""),
                    old_text=str(op.get("old_text") or ""),
                )
            return

        patterns = [
            r'''memory\(((?:[^)"']|"[^"]*"|'[^']*')*)\)''',
            r'<call:memory\s+([^/>]+)/?>',
            r'<memory\s+([^/>]+)/?>',
        ]

        matches = []
        for p in patterns:
            matches.extend(re.findall(p, llm_response))

        for match in matches:
            try:
                # 인자 파싱
                self._apply_memory_op(
                    target_store,
                    action=self._extract_kwarg(match, "action") or "",
                    target=self._extract_kwarg(match, "target") or "memory",
                    content=self._extract_kwarg(match, "content") or "",
                    old_text=self._extract_kwarg(match, "old_text") or "",
                )
            except Exception as e:
                logger.warning("[BG-Review] Failed to parse memory op '%s': %s", match, e)

    @staticmethod
    def _parse_json_memory_ops(llm_response: str) -> Optional[List[Dict[str, Any]]]:
        """JSON으로 답한 memory 작업 목록을 꺼냅니다. JSON 형태가 아니면 None.

        코드 펜스(```json ... ```) 안의 JSON을 먼저 보고, 없으면 응답 전체를 JSON으로 읽어 봅니다.
        배열, 단일 객체, {"operations": [...]} 모양을 모두 받습니다.
        """
        text = (llm_response or "").strip()
        candidates = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL) or [text]
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate.startswith(("[", "{")):
                continue
            try:
                data = json.loads(candidate)
            except (json.JSONDecodeError, ValueError):
                continue
            if isinstance(data, dict):
                data = data.get("operations", [data])
            if isinstance(data, list):
                ops = [op for op in data if isinstance(op, dict) and op.get("action")]
                if ops or not data:
                    return ops
        return None

    @staticmethod
    def _apply_memory_op(
        target_store: SemanticMemoryStore, action: str, target: str, content: str, old_text: str
    ) -> None:
        """파싱된 memory 작업 하나를 저장소에 반영합니다 (필수 인자가 빠진 작업은 건너뜀).

        Background Delete Gate (Hermes _background_delete_gate): 이 경로는 사용자가 없는 백그라운드
        리뷰 전용이므로 add만 바로 반영하고, replace/remove는 pending.json에 제안으로만 남깁니다.
        대화 중 에이전트가 직접 부르는 memory 도구는 이 게이트를 거치지 않습니다.
        """
        if action == "add" and content:
            result = target_store.add(target, content)
        elif (action == "replace" and old_text and content) or (action == "remove" and old_text):
            result = target_store.stage_pending(action, target, content=content, old_text=old_text)
            logger.info("[BG-Review] Memory %s staged for approval (%s): %s",
                        action, result.get("pending_id"), old_text[:50])
            return
        else:
            return

        if result.get("success"):
            logger.info("[BG-Review] Memory %s: %s → %s",
                        action, target, content[:50] if content else old_text[:50])
        else:
            logger.warning("[BG-Review] Memory %s failed: %s",
                           action, result.get("error", "unknown"))

    @staticmethod
    def _extract_kwarg(args_str: str, key: str) -> Optional[str]:
        """키워드 인자 문자열에서 특정 키의 값을 추출.

        여는 따옴표와 같은 종류의 따옴표에서만 끝냅니다 (content="User's ..."의 아포스트로피에서 잘리지 않도록).
        """
        pattern = rf'\b{key}\s*=\s*(["\'])(.*?)(?<!\\)\1'
        m = re.search(pattern, args_str, re.DOTALL)
        return m.group(2) if m else None

    def _review_episodic_memory(
        self,
        messages: List[Dict[str, Any]],
        session_id: str,
        user_id: str = _DEFAULT_USER_ID,
        cancel_event: Optional[threading.Event] = None,
        session_lock: Optional[threading.Lock] = None,
    ) -> None:
        """세션 메시지를 EpisodicStore에 저장 + 요약 생성 (세션 락 및 취소 이벤트 적용)."""
        if cancel_event is not None and cancel_event.is_set():
            return
        epi_store = self._get_episodic_store(user_id)
        try:
            if session_lock is not None:
                with session_lock:
                    if cancel_event is not None and cancel_event.is_set():
                        return
                    _run_async_safe(
                        epi_store.finalize_session(
                            session_id=session_id,
                            messages=messages,
                            llm=self.review_llm,
                            user_id=user_id,
                            cancel_event=cancel_event,
                            record_transcript=False,
                        )
                    )
            else:
                # record_transcript=False: 원문 기록은 훅이 담당하므로 요약 단계에서는 원문 저장소에 쓰지 않음
                _run_async_safe(
                    epi_store.finalize_session(
                        session_id=session_id,
                        messages=messages,
                        llm=self.review_llm,
                        user_id=user_id,
                        cancel_event=cancel_event,
                        record_transcript=False,
                    )
                )
        except Exception as e:
            logger.warning("[BG-Review] Episodic finalize failed: %s", e)

    # ══════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _extract_last_user_query(state: Dict[str, Any]) -> str:
        """state에서 마지막 유저 메시지 추출."""
        messages = state.get("messages", [])
        idx = last_user_turn_start(messages)
        if idx is None:
            return ""
        msg = messages[idx]
        if isinstance(msg, dict):
            return normalize_content(msg.get("content", "") or "")
        return normalize_content(getattr(msg, "content", "") or "")

    @staticmethod
    def _get_session_id(runtime: Any, request: Any = None) -> str:
        """runtime 및 request에서 session_id(thread_id)를 지능적으로 추출."""
        # 1. langchain_core의 get_config_from_context() 시도
        try:
            from langchain_core.runnables.config import get_config_from_context
            cfg = get_config_from_context()
            if cfg and isinstance(cfg, dict):
                tid = cfg.get("configurable", {}).get("thread_id")
                if tid:
                    return str(tid)
        except Exception:
            pass

        ctx = getattr(runtime, "context", None)
        if not ctx and request:
            ctx = getattr(getattr(request, "runtime", None), "context", None)

        if ctx:
            sid = getattr(ctx, "session_id", None)
            if sid and sid != "unknown":
                return str(sid)

        if request:
            configurable = getattr(request, "configurable", None)
            if isinstance(configurable, dict) and configurable.get("thread_id"):
                return str(configurable["thread_id"])

        config = getattr(runtime, "config", None)
        if not config and request:
            config = getattr(getattr(request, "runtime", None), "config", None)

        if config:
            if isinstance(config, dict):
                tid = config.get("configurable", {}).get("thread_id")
                if tid:
                    return str(tid)
            elif hasattr(config, "get"):
                configurable = config.get("configurable", {})
                if isinstance(configurable, dict) and configurable.get("thread_id"):
                    return str(configurable["thread_id"])
            elif hasattr(config, "configurable"):
                configurable = getattr(config, "configurable", {})
                if isinstance(configurable, dict) and configurable.get("thread_id"):
                    return str(configurable["thread_id"])

        return "unknown"

    @staticmethod
    def _format_episodic(sessions: List[Dict[str, Any]]) -> str:
        """에피소드 검색 결과를 프롬프트 주입용 텍스트로 포맷."""
        if not sessions:
            return ""

        separator = "═" * 46
        header = "EPISODIC MEMORY (relevant past sessions)"
        lines = [separator, header, separator]

        for i, s in enumerate(sessions, 1):
            lines.append(f"[Session {i}] {s.get('session_id', 'unknown')}")
            lines.append(f"  Summary: {s.get('summary', 'N/A')}")
            topics = s.get("topics") or []
            if isinstance(topics, list) and topics:
                topic_titles = []
                for t in topics:
                    if isinstance(t, str) and t.strip():
                        topic_titles.append(t.strip())
                    elif isinstance(t, dict) and (t.get("title") or t.get("outcome")):
                        topic_titles.append(str(t.get("title") or t.get("outcome") or "").strip())
                if topic_titles:
                    lines.append(f"  Topics: {'; '.join(topic_titles[:4])}")
            tool_names = s.get("tool_names") or []
            if tool_names:
                lines.append(f"  Tools: {', '.join(tool_names)}")
            keywords = s.get("keywords", [])
            if keywords:
                lines.append(f"  Keywords: {', '.join(keywords)}")
            lines.append("")

        return "\n".join(lines)


# ── Module-level helper ──

def _run_async_safe(coro):
    """비동기 코루틴을 안전하게 동기 실행."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)
