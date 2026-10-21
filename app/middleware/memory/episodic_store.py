"""
===============================================================================
Episodic Memory Store — episodic.db (L2 Episodic: 요약 색인 계층, SQLite + FTS5)
===============================================================================
Source: frontier-agent-lab/modules/hermes/session_store.py

세션 데이터 이원화 구조에서 "일화 기억 색인"을 담당합니다.
- 원문(전체본)은 transcript.db(TranscriptStore)에 추가 전용으로 저장됩니다.
- episodic.db에는 세션별 요약·주제(topics)·키워드·사용 도구 목록만 저장합니다.
  - sessions 테이블: session_summary, topics_json, keywords, tool_names
  - sessions_fts: 요약 + 주제 제목/요청/결과 + 키워드 전문 검색 (한/영 이중 색인)

finalize_session()이 호출되면 (MemoryMiddleware의 대기 후 실행 큐가 호출):
1. 세션 원문을 준비
   - record_transcript=True(기본, 노트북·직접 호출): messages를 transcript.db에 추가한 뒤 원문 전체를 읽음
   - record_transcript=False(MemoryMiddleware): 전달된 messages만 요약에 사용하고 원문 저장소에는 쓰지 않음
     (원문 기록 스위치가 꺼진 턴이 요약 단계에서 transcript.db로 새지 않도록)
2. 로컬 규칙 기반으로 원문 키워드 추출 (1단계)
3. LLM이 있으면 주제별(topic-segmented) 요약 + 한/영 키워드 추출 후 병합 (2단계)
4. sessions 테이블과 sessions_fts 갱신
→ 다음 세션에서 FTS5로 검색되어 L4 힌트로 주입됩니다.
===============================================================================
"""

import os
import re
import json
import time
import logging
import asyncio
import aiosqlite
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from app.utils.message_utils import normalize_content
from app.utils.message_markers import (
    ORIGIN_STOP_HOOK,
    TRANSCRIPT_SKIP_KEY,
    is_compaction_artifact,
    message_origin,
)
from app.middleware.memory.transcript_store import (
    HIDDEN_KINDS,
    TranscriptStore,
    classify_message,
    render_transcript_for_summary,
)

logger = logging.getLogger(__name__)

# ── SQL Schema ──
# 원문 메시지는 transcript.db에 저장하므로 episodic.db는 세션 요약 색인만 가집니다.
# (구버전 DB의 messages 테이블은 남겨 두고 session_recall 하위 호환 조회에만 사용합니다)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT DEFAULT 'default',
    summary TEXT,
    keywords TEXT,
    topics_json TEXT,
    tool_names TEXT,
    message_count INTEGER DEFAULT 0,
    created_at REAL,
    updated_at REAL
);
"""

FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    summary, keywords
);
"""

# ── Summary Generation Prompt (Topic-segmented, Hermes 템플릿 참고) ──

SUMMARY_PROMPT = """You are an episodic memory indexer. Read the session transcript below (user messages, assistant replies, tool calls and abbreviated tool results, in chronological order) and build a structured index that lets an AI agent find and resume this session later.

Segment the session into TOPICS. A new topic starts when the user's goal changes, not at every turn. Keep topics in chronological order (most sessions have 1-4 topics).

For each topic:
- title: 3-7 words naming what the user wanted DONE (not "user asked about..."). Keep technical terms, file names, numbers and error codes exact.
- request: the user's request in one sentence.
- actions: concrete actions taken, each formatted as "ACTION target — outcome [tool: name]".
  Example: "READ config.py:45 — found `==` should be `!=` [tool: file_read]". Use [tool: none] for reasoning-only steps. Max 8 per topic.
- decisions: important decisions and WHY, formatted as "decision — reason".
- corrections: corrections the USER gave, quoted, and what changed as a result ("\\"<user quote>\\" → <change>").
- outcome: start with exactly one of resolved | partial | open, then a short sentence.
- artifacts: files or outputs created or modified (paths).
- keywords: 3-8 atomic search terms for this topic.

Rules:
1. Be CONCRETE: file paths, commands, numbers, error messages. Never invent anything that is not in the transcript.
2. Write session_summary, title, request and outcome in the same language the user used.
3. NEVER include API keys, tokens or passwords — write [REDACTED] instead.
4. Keywords: SHORT, ATOMIC terms (1-2 words each). Preserve ORIGINAL words from the conversation. Include BOTH Korean and English forms of each key concept as separate entries. Nouns and technical terms only. Session-level keywords: 8-20 total.
5. Use empty lists when a field does not apply.

Keyword example:
Conversation: "Kubernetes 클러스터에서 Pod 오토스케일링 설정을 논의했고, HPA 메트릭으로 CPU 사용률 70%를 기준으로 확정했습니다."
Good keywords: ["Kubernetes", "클러스터", "Pod", "오토스케일링", "autoscaling", "HPA", "CPU", "메트릭"]
Bad keywords: ["쿠버네티스 클러스터 오토스케일링 설정", "pod autoscaling configuration"] ← too long, compound

Transcript:
{conversation}

Respond with JSON only, exactly in this shape:
{{"session_summary": "2-3 sentences covering ALL topics, focusing on decisions and outcomes", "topics": [{{"title": "...", "request": "...", "actions": ["..."], "decisions": ["..."], "corrections": ["..."], "outcome": "resolved | partial | open — ...", "artifacts": ["..."], "keywords": ["..."]}}], "keywords": ["keyword1", "keyword2"]}}
"""

# ── 한국어/영어 불용어 (키워드 추출 시 필터링) ──

_STOPWORDS_KO = frozenset([
    "그리고", "그래서", "하지만", "그런데", "또한", "그러나", "따라서", "때문에",
    "어떻게", "무엇을", "있습니다", "합니다", "됩니다", "입니다", "했습니다",
    "하겠습니다", "해주세요", "시작합시다", "진행합시다", "확정할까요",
    "네", "예", "아니요", "좋습니다", "감사합니다",
    "이것", "저것", "여기", "거기", "우리", "오늘",
    "어떤", "어디", "위해", "대신", "방식", "방식을", "방식으로",
])

_STOPWORDS_EN = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "and", "but", "or", "nor", "not", "so", "yet", "for", "with",
    "from", "into", "about", "that", "this", "these", "those",
    "what", "which", "who", "whom", "how", "where", "when", "why",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "than", "too", "very", "just", "also",
    "then", "there", "here", "only", "yes", "no",
    "use", "using", "used", "uses",
])


def _serialize_message(msg) -> Dict[str, str]:
    """LangChain 메시지 또는 dict를 직렬화 (content 문자열화 100% 보장)."""
    if isinstance(msg, BaseMessage):
        content = normalize_content(msg.content or "")
        return {"role": msg.type, "content": content}
    if isinstance(msg, dict):
        content = normalize_content(msg.get("content", ""))
        return {
            "role": msg.get("role", msg.get("type", "unknown")),
            "content": content,
        }
    return {"role": "unknown", "content": str(msg)}


def _records_from_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    """원문 저장소 없이 메시지 리스트로 요약할 때 transcript 레코드 형식으로 변환합니다."""
    records: List[Dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict):
            # 압축 요약·복원 블록은 원문이 아니므로 제외, StopHooks 피드백은 internal(요약 제외)
            if (m.get("additional_kwargs") or {}).get(TRANSCRIPT_SKIP_KEY) or is_compaction_artifact(m):
                continue
            role = m.get("role", m.get("type", ""))
            kind = m.get("kind") or {"user": "user", "human": "user", "assistant": "assistant", "ai": "assistant",
                                     "tool": "tool_result"}.get(role)
            if kind is None:
                continue
            if kind == "user" and message_origin(m) == ORIGIN_STOP_HOOK:
                kind = "internal"
            records.append({"kind": kind, "content": normalize_content(m.get("content", "") or ""),
                            "tool_calls": m.get("tool_calls") or [],
                            "tool_name": m.get("tool_name") or m.get("name"),
                            "status": m.get("status")})
            continue
        kind = classify_message(m)
        if kind is None:
            continue
        records.append({
            "kind": kind,
            "content": normalize_content(getattr(m, "content", "") or ""),
            "tool_calls": [{"name": tc.get("name"), "args": tc.get("args")}
                           for tc in (getattr(m, "tool_calls", None) or [])],
            "tool_name": getattr(m, "name", None),
            "status": getattr(m, "status", None),
        })
    return records


def _strip_json_fence(content: str) -> str:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if not cleaned.startswith("{"):
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end > start:
            cleaned = cleaned[start:end + 1]
    return cleaned


class EpisodicStore:
    """L2 에피소드 메모리 색인 — 세션 요약·주제·키워드를 SQLite FTS5로 저장하고 검색합니다.

    DB: app/database/episodic.db (user_id 컬럼으로 사용자별 격리)

    Args:
        db_path: episodic.db 경로
        transcript_store: 원문 저장소(TranscriptStore). None이고 auto_transcript=True이면
            episodic.db와 같은 폴더의 transcript.db를 자동으로 사용합니다 (노트북 단독 실행 호환).
        auto_transcript: False면 원문 저장소 없이 동작합니다 (finalize_session에 전달된
            메시지만으로 요약하며, session_recall 원문 조회는 구버전 테이블만 사용).
    """

    def __init__(self, db_path: str = "./app/database/episodic.db",
                 transcript_store: Optional[TranscriptStore] = None,
                 auto_transcript: bool = True):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None
        self._transcript = transcript_store
        self._auto_transcript = auto_transcript

    @property
    def transcript(self) -> Optional[TranscriptStore]:
        """원문 저장소 (필요 시 episodic.db 옆 transcript DB를 지연 생성).

        - episodic.db → 같은 폴더의 transcript.db
        - 그 외 이름(예: 테스트용 my_test.db) → my_test_transcript.db (실제 transcript.db 오염 방지)
        """
        if self._transcript is None and self._auto_transcript:
            self._transcript = TranscriptStore(db_path=self.default_transcript_path(self.db_path))
        return self._transcript

    @property
    def transcript_store(self) -> Optional[TranscriptStore]:
        return self.transcript

    @transcript_store.setter
    def transcript_store(self, value: Optional[TranscriptStore]) -> None:
        self._transcript = value

    @staticmethod
    def default_transcript_path(episodic_db_path: str) -> str:
        folder = os.path.dirname(episodic_db_path) or "."
        stem = os.path.splitext(os.path.basename(episodic_db_path))[0]
        name = "transcript.db" if stem == "episodic" else f"{stem}_transcript.db"
        return os.path.join(folder, name)

    async def setup(self) -> None:
        """DB 연결 + WAL 모드 + 테이블 생성 (+ 구버전 스키마 자동 마이그레이션)."""
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        conn = aiosqlite.connect(self.db_path, check_same_thread=False)
        conn.daemon = True
        if hasattr(conn, "_thread"):
            conn._thread.daemon = True
        self._conn = await conn
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        await self._conn.execute("PRAGMA busy_timeout=5000;")
        await self._conn.executescript(SCHEMA_SQL)
        try:
            cols = await self._conn.execute_fetchall("PRAGMA table_info(sessions)")
            col_names = {c[1] for c in cols}
            for col, ddl in (
                ("user_id", "ALTER TABLE sessions ADD COLUMN user_id TEXT DEFAULT 'default'"),
                ("topics_json", "ALTER TABLE sessions ADD COLUMN topics_json TEXT"),
                ("tool_names", "ALTER TABLE sessions ADD COLUMN tool_names TEXT"),
            ):
                if col not in col_names:
                    await self._conn.execute(ddl)
        except Exception as mig_err:
            logger.debug("sessions schema migration check: %s", mig_err)
        try:
            await self._conn.executescript(FTS_SCHEMA_SQL)
        except Exception as e:
            logger.warning("FTS5 setup warning (may already exist): %s", e)
        await self._conn.commit()
        logger.info("EpisodicStore initialized: %s", self.db_path)

    async def _ensure_conn(self) -> None:
        """setup()이 아직 호출되지 않았으면 첫 사용 시점에 연결합니다."""
        if not self._conn:
            await self.setup()

    async def close(self) -> None:
        """DB 연결 종료 (자동 생성한 원문 저장소도 함께 종료)."""
        if self._conn:
            await self._conn.close()
            self._conn = None
        if self._transcript is not None and self._auto_transcript:
            await self._transcript.aclose()

    async def _has_legacy_messages(self) -> bool:
        rows = await self._conn.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        )
        return bool(rows)

    # ── 메시지 저장 (하위 호환) ──

    async def save_messages(
        self, session_id: str, messages: List[Any], user_id: str = "default"
    ) -> int:
        """[하위 호환] 메시지를 원문 저장소(transcript.db)에 추가합니다 (이미 저장된 메시지는 건너뜀)."""
        store = self.transcript
        if store is None:
            return 0
        return await store.aappend(session_id, user_id, messages)

    # ── 세션 종료 (요약 생성) ──

    async def finalize_session(
        self,
        session_id: str,
        messages: Optional[List[Any]] = None,
        llm=None,
        user_id: str = "default",
        cancel_event=None,
        record_transcript: bool = True,
    ) -> str:
        """세션 요약 색인 생성: 원문 읽기 → 주제별 요약 + 키워드 → sessions/FTS 갱신.

        Args:
            session_id: 세션 식별자
            messages: (선택) 대화 메시지 리스트. 원문 저장소가 있으면 먼저 추가한 뒤 원문 전체를 요약합니다.
                      None이면 transcript.db에 이미 기록된 원문을 사용합니다.
            llm: 요약 생성에 사용할 LLM (None이면 로컬 규칙 기반 요약)
            user_id: 사용자 식별자 (기본값 'default')
            cancel_event: threading.Event — 설정되어 있으면 요약/DB 반영을 중단
            record_transcript: False면 messages를 원문 저장소에 쓰지 않고 그대로 요약합니다
                (MemoryMiddleware가 사용 — 원문 기록은 훅이 담당하고 스위치를 따릅니다).

        Returns:
            생성된 세션 요약 문자열 (취소되었거나 원문이 없으면 빈 문자열)
        """
        if cancel_event is not None and cancel_event.is_set():
            logger.info("Episodic finalize cancelled before start for session %s", session_id)
            return ""

        await self._ensure_conn()
        uid = (user_id or "default").strip() or "default"

        store = self.transcript if (record_transcript or not messages) else None
        if store is not None:
            if messages:
                await store.aappend(session_id, uid, messages)
            records = await store.aread_session(session_id, user_id=uid)
            tool_names = await asyncio.to_thread(store.session_tool_names, session_id)
        else:
            records = _records_from_messages(messages or [])
            tool_names = []
            for r in records:
                for tc in r.get("tool_calls") or []:
                    # 원문 행은 {"name": ...}, 스냅샷은 "name(args)" 문자열
                    name = tc.get("name") if isinstance(tc, dict) else str(tc).split("(", 1)[0].strip()
                    if name and name not in tool_names:
                        tool_names.append(name)

        records = [r for r in records if r.get("kind") not in HIDDEN_KINDS]
        if not any(r.get("kind") == "user" and r.get("content") for r in records):
            logger.info("Episodic finalize skipped for session %s: no user messages", session_id)
            return ""

        # 1. 요약 생성 (LLM 시간이 걸리는 작업을 먼저 수행하여 취소 시 DB 덮어쓰기 방지)
        summary, keywords, topics = await self._generate_summary(records, llm, tool_names=tool_names)

        if cancel_event is not None and cancel_event.is_set():
            logger.info("Episodic finalize cancelled after LLM summary for session %s (skipping DB write)", session_id)
            return ""

        # 2. sessions 테이블 upsert
        now = time.time()
        await self._conn.execute(
            """INSERT INTO sessions (session_id, user_id, summary, keywords, topics_json, tool_names,
                                     message_count, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   user_id = excluded.user_id,
                   summary = excluded.summary,
                   keywords = excluded.keywords,
                   topics_json = excluded.topics_json,
                   tool_names = excluded.tool_names,
                   message_count = excluded.message_count,
                   updated_at = excluded.updated_at""",
            (session_id, uid, summary, json.dumps(keywords, ensure_ascii=False),
             json.dumps(topics, ensure_ascii=False), json.dumps(tool_names, ensure_ascii=False),
             len(records), now, now),
        )

        # 3. FTS 인덱스 갱신
        await self._update_fts(session_id, summary, keywords, topics)
        await self._conn.commit()

        logger.info(
            "Session %s finalized (user=%s): %d messages, %d topics, summary=%d chars, keywords=%s",
            session_id, uid, len(records), len(topics), len(summary), keywords,
        )
        return summary

    @staticmethod
    def _extract_local_keywords(
        messages: List[Dict[str, str]], max_keywords: int = 15
    ) -> List[str]:
        """대화 원문에서 규칙 기반으로 한국어/영어 키워드를 추출합니다.

        Progressive Keyword Pipeline 1단계:
        - 대화 전체 텍스트에서 의미 있는 명사/기술어를 추출
        - 불용어(조사, 어미, 접속사 등) 필터링
        - 한글 2자 이상, 영어 3자 이상 단어만 선별
        - 빈도순으로 상위 max_keywords개 반환

        이 단계의 핵심 역할:
        유저가 실제로 말한 원문 단어("쿠키", "보안", "토큰" 등)를
        FTS5 인덱스에 100% 보존하여 검색 누락을 원천 차단합니다.
        """
        all_text = " ".join(
            m["content"] for m in messages
            if m.get("content") and m.get("role") != "tool" and m.get("kind") != "tool_result"
        )
        if not all_text.strip():
            return []

        # 한글 단어 추출 (2자 이상 연속 한글)
        ko_words = re.findall(r'[가-힣]{2,}', all_text)
        # 영어 단어 추출 (하이픈 포함 기술어, 3자 이상)
        en_words = re.findall(r'[A-Za-z][A-Za-z0-9\-]{2,}', all_text)

        # 불용어 필터링 + 빈도 집계
        freq: Dict[str, int] = {}
        for w in ko_words:
            if w not in _STOPWORDS_KO:
                freq[w] = freq.get(w, 0) + 1
        for w in en_words:
            w_lower = w.lower()
            if w_lower not in _STOPWORDS_EN:
                # 원래 대소문자 유지 (JWT, XSS 등 보존)
                key = w if w.isupper() or w[0].isupper() else w_lower
                freq[key] = freq.get(key, 0) + 1

        # 빈도순 정렬 후 상위 N개 반환
        sorted_words = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return [w for w, _ in sorted_words[:max_keywords]]

    async def _generate_summary(
        self, records: List[Dict[str, Any]], llm=None, tool_names: Optional[List[str]] = None
    ) -> tuple:
        """Progressive Keyword Pipeline으로 주제별 요약 + 키워드 추출.

        1단계: _extract_local_keywords()로 원문 키워드 추출 (항상 실행)
        2단계: LLM이 있으면 주제별 요약 + 한/영 이중 키워드 추출 후 1단계와 병합
               LLM이 없거나 실패하면 1단계 키워드 기반 Fallback 요약 생성

        Returns:
            (session_summary, keywords, topics)
        """
        local_keywords = self._extract_local_keywords(records)

        if llm is None:
            return self._fallback_summary(records, local_keywords, tool_names)

        conversation_text = render_transcript_for_summary(records)
        prompt = SUMMARY_PROMPT.format(conversation=conversation_text)

        try:
            loop = asyncio.get_running_loop()
            try:
                response = await loop.run_in_executor(None, llm.invoke, prompt)
            except Exception:
                if hasattr(llm, "ainvoke"):
                    response = await llm.ainvoke(prompt)
                else:
                    raise
            raw_content = response.content if hasattr(response, "content") else str(response)
            result = json.loads(_strip_json_fence(normalize_content(raw_content)))

            summary = result.get("session_summary") or result.get("summary") or "No summary available."
            topics = [t for t in (result.get("topics") or []) if isinstance(t, dict)]
            raw_kw = result.get("keywords") or []
            if isinstance(raw_kw, str):
                llm_keywords = [k.strip() for k in raw_kw.split(",") if k.strip()]
            else:
                llm_keywords = [str(k).strip() for k in raw_kw if str(k).strip()]
            for t in topics:
                t_kw = t.get("keywords") or []
                if isinstance(t_kw, str):
                    llm_keywords.extend([k.strip() for k in t_kw.split(",") if k.strip()])
                else:
                    llm_keywords.extend([str(k).strip() for k in t_kw if str(k).strip()])

            # 핵심: LLM 키워드(지능형 확장) + 로컬 키워드(원문 보존) 병합
            merged = self._merge_keywords(local_keywords, llm_keywords)
            return summary, merged, topics
        except Exception as e:
            logger.warning("LLM summary generation failed: %s. Using fallback.", e)
            return self._fallback_summary(records, local_keywords, tool_names)

    @staticmethod
    def _merge_keywords(
        local_keywords: List[str],
        llm_keywords: List[str],
        max_total: int = 24,
    ) -> List[str]:
        """로컬 키워드와 LLM 키워드를 병합합니다.

        - LLM 키워드를 우선 배치 (의미적으로 정제된 키워드)
        - 로컬 키워드 중 LLM이 놓친 원문 단어를 뒤에 추가
        - 대소문자 무시 중복 제거
        """
        seen_lower = set()
        merged = []

        for kw in llm_keywords:
            kw_str = str(kw).strip()
            if kw_str and kw_str.lower() not in seen_lower:
                seen_lower.add(kw_str.lower())
                merged.append(kw_str)

        for kw in local_keywords:
            if kw.lower() not in seen_lower:
                seen_lower.add(kw.lower())
                merged.append(kw)

        return merged[:max_total]

    def _fallback_summary(
        self, records: List[Dict[str, Any]], local_keywords: List[str] = None,
        tool_names: Optional[List[str]] = None,
    ) -> tuple:
        """LLM 없이 규칙 기반 요약 생성 (첫 질문 + 최근 질문 + 사용 도구)."""
        user_msgs = [r["content"] for r in records
                     if r.get("kind") == "user" or r.get("role") in ("user", "human")]
        user_msgs = [m for m in user_msgs if m]
        if not user_msgs:
            return "Empty conversation.", [], []

        summary = f"Conversation starting with: {user_msgs[0][:200]}"
        if len(user_msgs) > 1:
            summary += f" | Latest request: {user_msgs[-1][:120]}"
        if tool_names:
            summary += f" | Tools used: {', '.join(tool_names[:8])}"

        if local_keywords is None:
            local_keywords = self._extract_local_keywords(records)

        return summary, local_keywords, []

    @staticmethod
    def _format_conversation(messages: List[Dict[str, str]], max_chars: int = 24000) -> str:
        """[하위 호환] 메시지를 요약 입력 텍스트로 포맷합니다."""
        return render_transcript_for_summary(_records_from_messages(messages), budget_chars=max_chars)

    @staticmethod
    def _topic_index_text(summary: str, topics: List[Dict[str, Any]]) -> str:
        parts = [summary or ""]
        for t in topics:
            for key in ("title", "request", "outcome"):
                if t.get(key):
                    parts.append(str(t[key]))
        return "\n".join(p for p in parts if p)

    async def _update_fts(
        self,
        session_id: str,
        summary: str,
        keywords: List[str],
        topics: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """sessions_fts 갱신 (요약 + 주제 제목/요청/결과 + 키워드)."""
        try:
            row = await self._conn.execute_fetchall(
                "SELECT rowid FROM sessions WHERE session_id = ?", (session_id,)
            )
            if row:
                rowid = row[0][0]
                try:
                    await self._conn.execute("DELETE FROM sessions_fts WHERE rowid = ?", (rowid,))
                except Exception:
                    pass
                await self._conn.execute(
                    "INSERT INTO sessions_fts(rowid, summary, keywords) VALUES (?, ?, ?)",
                    (rowid, self._topic_index_text(summary, topics or []),
                     json.dumps(keywords, ensure_ascii=False)),
                )
        except Exception as e:
            logger.warning("FTS update failed: %s", e)

    # ── 검색 ──
    # Extension Point: 향후 Dense Vector(Embedding) 기반 Hybrid Search 추가 시
    # search_sessions()에 strategy="fts" | "dense" | "hybrid" 파라미터를 도입하고,
    # _search_fts()와 _search_dense()를 분리한 뒤 RRF(Reciprocal Rank Fusion)로
    # 결과를 병합하는 구조로 확장할 수 있습니다.

    @staticmethod
    def _row_to_session(row) -> Dict[str, Any]:
        topics = json.loads(row[3]) if row[3] else []
        return {
            "session_id": row[0],
            "summary": row[1] or "",
            "keywords": json.loads(row[2]) if row[2] else [],
            "topics": [t.get("title", "") for t in topics if isinstance(t, dict) and t.get("title")],
            "tool_names": json.loads(row[4]) if row[4] else [],
            "message_count": row[5] or 0,
        }

    async def search_sessions(
        self,
        query: str,
        top_k: int = 3,
        exclude_session_id: str = None,
        user_id: str = "default",
    ) -> List[Dict[str, Any]]:
        """FTS5 기반 세션 요약 검색 (user_id 격리).

        Returns:
            [{session_id, summary, keywords, topics, tool_names, message_count, snippet}, ...]
        """
        if not query.strip():
            return []
        await self._ensure_conn()
        uid = (user_id or "default").strip() or "default"

        try:
            # FTS5 구문 에러(-, &, :, *, / 등) 완전 방지 정제 로직
            cleaned = re.sub(r'[^\w\s\uac00-\ud7a3]', ' ', query)
            words = [w.strip() for w in cleaned.split() if w.strip()]
            fts_query = " OR ".join(words) if words else query

            sql = """
                SELECT s.session_id, s.summary, s.keywords, s.topics_json, s.tool_names, s.message_count,
                       snippet(sessions_fts, 0, '>>>', '<<<', '...', 32) as snippet
                FROM sessions_fts
                JOIN sessions s ON sessions_fts.rowid = s.rowid
                WHERE sessions_fts MATCH ? AND COALESCE(s.user_id, 'default') = ?
                ORDER BY rank
                LIMIT ?
            """
            rows = await self._conn.execute_fetchall(sql, (fts_query, uid, top_k + 5))

            results = []
            for row in rows:
                if exclude_session_id and row[0] == exclude_session_id:
                    continue
                item = self._row_to_session(row)
                item["snippet"] = row[6] or ""
                results.append(item)
                if len(results) >= top_k:
                    break
            return results
        except Exception as e:
            logger.warning("Session search failed: %s", e)
            return []

    async def get_anchored_view(
        self,
        session_id: str,
        anchor_keyword: str = "",
        window: int = 5,
        user_id: str = "default",
        around_message_id: int = 0,
    ) -> List[Dict[str, Any]]:
        """Anchor 기반 원문 인출 (transcript.db 우선, 구버전 episodic.messages 하위 호환).

        1. 요청자의 user_id 소유 세션인지 검증 (타 사용자 세션 접근 차단)
        2. anchor_keyword(또는 around_message_id) 기준 ±window 메시지 반환
        3. 세션의 첫 3개 + 마지막 3개 메시지를 bookend로 항상 포함
        """
        store = self.transcript
        if store is not None:
            view = await store.aget_anchored_view(
                session_id, anchor_keyword=anchor_keyword, window=window,
                user_id=user_id, around_message_id=around_message_id,
            )
            if view:
                return view
        return await self._legacy_anchored_view(session_id, anchor_keyword, window, user_id)

    async def _legacy_anchored_view(self, session_id: str, anchor_keyword: str, window: int,
                                    user_id: str) -> List[Dict[str, Any]]:
        """구버전 episodic.db의 messages 테이블에서 조회 (원문 이관 전 세션용)."""
        await self._ensure_conn()
        uid = (user_id or "default").strip() or "default"
        try:
            if not await self._has_legacy_messages():
                return []
            rows = await self._conn.execute_fetchall(
                "SELECT m.id, m.role, m.content, m.created_at "
                "FROM messages m JOIN sessions s ON m.session_id = s.session_id "
                "WHERE m.session_id = ? AND s.user_id = ? ORDER BY m.id ASC",
                (session_id, uid),
            )
            if not rows:
                return []
            all_messages = [
                {"id": r[0], "seq": i, "role": r[1], "kind": r[1], "content": r[2] or "", "created_at": r[3]}
                for i, r in enumerate(rows)
            ]
            if not anchor_keyword.strip():
                return all_messages[-window:]
            anchor_idx = TranscriptStore._find_anchor(all_messages, anchor_keyword)
            core = all_messages[max(0, anchor_idx - window): anchor_idx + window + 1]
            seen, result = set(), []
            for m in all_messages[:3] + core + all_messages[-3:]:
                if m["id"] not in seen:
                    seen.add(m["id"])
                    result.append(m)
            result.sort(key=lambda m: m["id"])
            return result
        except Exception as e:
            logger.warning("Legacy anchored view failed for session %s: %s", session_id, e)
            return []

    @staticmethod
    def _find_anchor(messages: List[Dict], keyword: str) -> int:
        """[하위 호환] 키워드와 가장 잘 매칭되는 메시지의 인덱스 반환."""
        return TranscriptStore._find_anchor(messages, keyword)

    async def browse_recent(self, limit: int = 5, user_id: str = "default") -> List[Dict[str, Any]]:
        """최근 세션 목록 반환 (Browse 모드, user_id 격리 적용)."""
        await self._ensure_conn()
        uid = (user_id or "default").strip() or "default"

        try:
            rows = await self._conn.execute_fetchall(
                "SELECT session_id, summary, keywords, topics_json, tool_names, message_count, updated_at "
                "FROM sessions WHERE summary IS NOT NULL AND user_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (uid, limit),
            )
            results = []
            for r in rows:
                item = self._row_to_session(r)
                item["updated_at"] = r[6]
                results.append(item)
            return results
        except Exception as e:
            logger.warning("Browse recent failed: %s", e)
            return []
