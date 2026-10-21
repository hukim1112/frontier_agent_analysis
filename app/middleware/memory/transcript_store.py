"""
===============================================================================
Transcript Store — transcript.db (L2 Episodic: 원문 계층, Append-only)
===============================================================================
세션 데이터 이원화(Dual-tier Session Data)의 "전체본"을 담당합니다.

- 압축본 (L1 Short-term / Working): LangGraph checkpointer state["messages"]
  → 모델 입력용. Auto-Compact가 실제로 교체(요약)합니다.
- 전체본 (L2 Episodic 원문): transcript.db (이 모듈)
  → 사용자 질문, AI 답변, 도구 호출(이름·인자), 도구 결과 원문, 압축 이벤트를
     "생성되는 즉시" 추가만 합니다(삭제/덮어쓰기 없음).
- 색인 (L2 Episodic 요약): episodic.db (EpisodicStore)
  → 세션별 주제 요약·키워드. 이 모듈의 원문을 읽어 생성합니다.

설계 포인트:
- message_uid(LangChain 메시지 id) + UNIQUE(session_id, message_uid)로 멱등 추가
  → 여러 훅(before_agent / wrap_model_call / wrap_tool_call / after_agent)에서
     중복 호출해도 한 번만 저장됩니다.
- kind: user / assistant / tool_result / compaction / internal / blocked
  → internal(StopHooks 피드백)·blocked(가드레일 차단 턴)은 저장하되
     검색·회상·요약·학습에서 제외합니다.
- messages_fts: 사용자 질문·AI 답변·도구 이름/인자만 색인합니다.
  도구 결과 원문은 저장만 하고 색인하지 않습니다(대량 텍스트가 검색 결과를
  덮어버리는 recall blindness 방지).
- 표준 sqlite3 + RLock + WAL: 동기 훅, 비동기 훅(asyncio.to_thread), 백그라운드
  리뷰 스레드 어디서 호출해도 안전합니다. 서버 배포 시 같은 인터페이스로
  Postgres 구현으로 교체할 수 있습니다.
===============================================================================
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.utils.message_utils import normalize_content
from app.utils.message_markers import (
    ORIGIN_STOP_HOOK,
    TRANSCRIPT_SKIP_KEY,
    is_compaction_artifact,
    message_origin,
)

logger = logging.getLogger(__name__)

# ── 다른 미들웨어와 공유하는 메시지 마커 (additional_kwargs 키) ──
# TRANSCRIPT_SKIP_KEY: Compactor가 state에 넣는 요약/복원 메시지 — 원문에는 record_compaction()으로 따로 기록
# 가드레일이 차단 응답에 붙이는 표시: 차단된 질문·응답을 kind=blocked로 기록
GUARDRAIL_BLOCKED_KEY = "guardrail_blocked"

# 마커가 없는 구버전 메시지를 위한 내용 기반 판별 (하위 호환)
_BLOCKED_PREFIXES = ("🛡️ [Safety Guard Blocked]", "🛑 [Topic Guard Blocked]")

# 검색·회상·요약에서 제외하는 kind
HIDDEN_KINDS = ("internal", "blocked", "compaction")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    user_id TEXT NOT NULL DEFAULT 'default',
    turn_id INTEGER,
    seq INTEGER NOT NULL,
    message_uid TEXT NOT NULL,
    role TEXT NOT NULL,
    kind TEXT NOT NULL,
    content TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    tool_call_id TEXT,
    status TEXT,
    created_at REAL NOT NULL,
    UNIQUE(session_id, message_uid)
);
CREATE INDEX IF NOT EXISTS idx_transcript_session_seq ON messages(session_id, seq);
CREATE INDEX IF NOT EXISTS idx_transcript_user ON messages(user_id, created_at);
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(content);
"""


def _short(value: Any, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_tool_call(tc: Dict[str, Any], args_limit: int = 200) -> str:
    """도구 호출 1건을 사람이 읽기 좋은 한 줄로 변환합니다: name(key=value, ...)."""
    name = tc.get("name", "tool")
    args = tc.get("args") or {}
    if isinstance(args, dict):
        parts = [f"{k}={_short(v, 80)}" for k, v in args.items()]
        return f"{name}({_short(', '.join(parts), args_limit)})"
    return f"{name}({_short(args, args_limit)})"


def _fts_query(query: str) -> str:
    """FTS5 구문 에러를 막기 위해 특수문자를 제거하고 OR 질의로 변환합니다."""
    cleaned = re.sub(r"[^\w\s\uac00-\ud7a3]", " ", query or "")
    words = [w for w in cleaned.split() if w.strip()]
    return " OR ".join(f'"{w}"' for w in words)


def classify_message(msg: Any) -> Optional[str]:
    """LangChain 메시지의 transcript kind를 판별합니다. None이면 기록하지 않습니다."""
    kwargs = getattr(msg, "additional_kwargs", None) or {}
    if kwargs.get(TRANSCRIPT_SKIP_KEY) or is_compaction_artifact(msg):
        return None
    if isinstance(msg, SystemMessage):
        return None
    content = normalize_content(getattr(msg, "content", "") or "")
    if isinstance(msg, HumanMessage):
        return "internal" if message_origin(msg) == ORIGIN_STOP_HOOK else "user"
    if isinstance(msg, AIMessage):
        if kwargs.get(GUARDRAIL_BLOCKED_KEY) or content.startswith(_BLOCKED_PREFIXES):
            return "blocked"
        return "assistant"
    if isinstance(msg, ToolMessage):
        return "tool_result"
    msg_type = getattr(msg, "type", "")
    return {"human": "user", "ai": "assistant", "tool": "tool_result"}.get(msg_type)


def is_blocked_message(msg: Any) -> bool:
    return classify_message(msg) == "blocked"


def ensure_message_id(msg: Any) -> Any:
    """state에 들어가기 전 메시지에 id가 없으면 부여합니다 (transcript와 state가 같은 id를 쓰도록)."""
    if isinstance(msg, BaseMessage) and not getattr(msg, "id", None):
        try:
            msg.id = str(uuid.uuid4())
        except Exception:
            pass
    return msg


class _AwaitableNone:
    """동기 호출(store.setup())과 비동기 호출(await store.setup())을 모두 허용하는 헬퍼."""

    def __await__(self):
        if False:  # pragma: no cover
            yield None
        return None


_AWAITABLE_NONE = _AwaitableNone()


class TranscriptStore:
    """세션 원문(전체본)을 transcript.db에 추가 전용으로 저장하고 조회합니다.

    Args:
        db_path: SQLite 파일 경로 (기본 ./app/database/transcript.db)
        recall_max_chars: session_recall 등 조회 시 메시지당 최대 반환 길이
    """

    def __init__(self, db_path: str = "./app/database/transcript.db", recall_max_chars: int = 2000,
                 max_cached_sessions: int = 256):
        self.db_path = db_path
        self.recall_max_chars = recall_max_chars
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None
        # 세션별 이미 기록한 message_uid 캐시 (매 훅마다 DB를 조회하지 않기 위함)
        self._known_uids: "OrderedDict[str, set]" = OrderedDict()
        self._max_cached_sessions = max_cached_sessions

    # ── Connection ──

    def setup(self) -> _AwaitableNone:
        with self._lock:
            if self._conn is not None:
                return _AWAITABLE_NONE
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA busy_timeout=5000;")
            conn.executescript(SCHEMA_SQL)
            conn.commit()
            self._conn = conn
            logger.info("TranscriptStore initialized: %s", self.db_path)
        return _AWAITABLE_NONE

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            self.setup()
        return self._conn  # type: ignore[return-value]

    def close(self) -> _AwaitableNone:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
            self._known_uids.clear()
        return _AWAITABLE_NONE

    async def aclose(self) -> None:
        await asyncio.to_thread(self.close)

    # ── uid helpers ──

    def _known(self, session_id: str) -> set:
        with self._lock:
            known = self._known_uids.get(session_id)
            if known is None:
                rows = self._db().execute(
                    "SELECT message_uid FROM messages WHERE session_id = ?", (session_id,)
                ).fetchall()
                known = {r[0] for r in rows}
                self._known_uids[session_id] = known
                while len(self._known_uids) > self._max_cached_sessions:
                    self._known_uids.popitem(last=False)
            else:
                self._known_uids.move_to_end(session_id)
            return known

    @staticmethod
    def _uid_for(msg: Any, session_id: str, ordinal: int) -> str:
        if isinstance(msg, dict):
            mid = msg.get("message_uid") or msg.get("id")
        else:
            mid = getattr(msg, "id", None)
        if mid:
            return str(mid)
        # id가 없는 메시지(노트북의 dict 메시지 등): 세션·순번·내용 기반의 안정적인 uid
        if isinstance(msg, dict):
            role, content = msg.get("role", msg.get("type", "")), msg.get("content", "")
        else:
            role, content = getattr(msg, "type", ""), getattr(msg, "content", "")
        raw = f"{session_id}|{ordinal}|{role}|{normalize_content(content or '')}"
        return "h-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:24]

    # ── Write ──

    def begin_turn(self, session_id: str) -> int:
        """새 사용자 턴 번호를 발급합니다 (세션 내 최대 turn_id + 1)."""
        with self._lock:
            row = self._db().execute(
                "SELECT COALESCE(MAX(turn_id), 0) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
            return int(row[0] or 0) + 1

    def current_turn(self, session_id: str) -> int:
        with self._lock:
            row = self._db().execute(
                "SELECT COALESCE(MAX(turn_id), 0) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
            return max(1, int(row[0] or 0))

    def append(
        self,
        session_id: str,
        user_id: str,
        messages: Iterable[Any],
        turn_id: Optional[int] = None,
        kind_override: Optional[str] = None,
    ) -> int:
        """새 메시지만 추가합니다 (이미 기록된 message_uid는 건너뜀). 추가된 건수를 반환."""
        if not session_id or session_id == "unknown":
            return 0
        uid_user = (user_id or "default").strip() or "default"
        inserted = 0
        with self._lock:
            db = self._db()
            known = self._known(session_id)
            seq_row = db.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
            seq = int(seq_row[0] or 0)
            if turn_id is None:
                turn_id = self.current_turn(session_id)
            now = time.time()
            for ordinal, msg in enumerate(messages):
                row = self._to_row(msg, session_id, ordinal)
                if row is None:
                    continue
                if row["message_uid"] in known:
                    continue
                kind = kind_override or row["kind"]
                seq += 1
                try:
                    cur = db.execute(
                        "INSERT INTO messages (session_id, user_id, turn_id, seq, message_uid, role, kind, content, "
                        "tool_calls, tool_name, tool_call_id, status, created_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (session_id, uid_user, turn_id, seq, row["message_uid"], row["role"], kind, row["content"],
                         row["tool_calls"], row["tool_name"], row["tool_call_id"], row["status"], now),
                    )
                except sqlite3.IntegrityError:
                    known.add(row["message_uid"])
                    seq -= 1
                    continue
                known.add(row["message_uid"])
                inserted += 1
                fts_text = self._fts_text(kind, row)
                if fts_text:
                    db.execute("INSERT INTO messages_fts(rowid, content) VALUES (?, ?)", (cur.lastrowid, fts_text))
            db.commit()
        return inserted

    async def aappend(self, *args, **kwargs) -> int:
        return await asyncio.to_thread(self.append, *args, **kwargs)

    def _to_row(self, msg: Any, session_id: str, ordinal: int) -> Optional[Dict[str, Any]]:
        if isinstance(msg, dict):
            if (msg.get("additional_kwargs") or {}).get(TRANSCRIPT_SKIP_KEY) or is_compaction_artifact(msg):
                return None
            role = msg.get("role", msg.get("type", "unknown"))
            kind = {"user": "user", "human": "user", "assistant": "assistant", "ai": "assistant",
                    "tool": "tool_result"}.get(role)
            if kind is None:
                return None
            if kind == "user" and message_origin(msg) == ORIGIN_STOP_HOOK:
                kind = "internal"
            return {
                "message_uid": self._uid_for(msg, session_id, ordinal),
                "role": {"user": "human", "assistant": "ai", "tool": "tool"}.get(role, role),
                "kind": kind,
                "content": normalize_content(msg.get("content", "") or ""),
                "tool_calls": json.dumps(msg.get("tool_calls"), ensure_ascii=False) if msg.get("tool_calls") else None,
                "tool_name": msg.get("tool_name") or msg.get("name"),
                "tool_call_id": msg.get("tool_call_id"),
                "status": msg.get("status"),
            }
        kind = classify_message(msg)
        if kind is None:
            return None
        tool_calls = None
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            tool_calls = json.dumps(
                [{"name": tc.get("name"), "args": tc.get("args"), "id": tc.get("id")} for tc in msg.tool_calls],
                ensure_ascii=False, default=str,
            )
        return {
            "message_uid": self._uid_for(msg, session_id, ordinal),
            "role": getattr(msg, "type", "unknown"),
            "kind": kind,
            "content": normalize_content(getattr(msg, "content", "") or ""),
            "tool_calls": tool_calls,
            "tool_name": getattr(msg, "name", None) if isinstance(msg, ToolMessage) else None,
            "tool_call_id": getattr(msg, "tool_call_id", None) if isinstance(msg, ToolMessage) else None,
            "status": getattr(msg, "status", None) if isinstance(msg, ToolMessage) else None,
        }

    @staticmethod
    def _fts_text(kind: str, row: Dict[str, Any]) -> str:
        if kind not in ("user", "assistant"):
            return ""
        parts = [row.get("content") or ""]
        if row.get("tool_calls"):
            try:
                for tc in json.loads(row["tool_calls"]):
                    parts.append(format_tool_call(tc))
            except Exception:
                pass
        return "\n".join(p for p in parts if p).strip()

    def record_blocked_exchange(self, session_id: str, user_id: str, messages: List[Any]) -> int:
        """가드레일 차단 턴(사용자 질문 + 차단 응답)을 kind=blocked로 기록합니다.

        가드레일은 MemoryMiddleware.before_agent보다 먼저 턴을 끝내므로 새 턴 번호가 발급되지 않은
        상태일 수 있습니다. 이미 기록된 메시지가 있으면 그 턴 번호를, 없으면 새 턴 번호를 사용해
        직전 정상 턴과 섞이지 않게 합니다. 바꾸는 대상은 전달된 메시지뿐입니다.

        Returns:
            사용한 turn_id
        """
        if not session_id or session_id == "unknown" or not messages:
            return 0
        with self._lock:
            db = self._db()
            uids = [self._uid_for(m, session_id, i) for i, m in enumerate(messages)]
            placeholders = ",".join("?" * len(uids))
            rows = db.execute(
                f"SELECT id, turn_id FROM messages WHERE session_id = ? AND message_uid IN ({placeholders})",
                (session_id, *uids),
            ).fetchall()
            turn_id = int(rows[0][1]) if rows and rows[0][1] else self.begin_turn(session_id)
            self.append(session_id, user_id, messages, turn_id=turn_id, kind_override="blocked")
            # before_agent가 먼저 user로 기록해 둔 질문 등은 blocked로 바꾸고 검색 색인에서 제거
            for rid, _ in rows:
                db.execute("DELETE FROM messages_fts WHERE rowid = ?", (rid,))
                db.execute("UPDATE messages SET kind = 'blocked' WHERE id = ?", (rid,))
            db.commit()
        return turn_id

    def mark_turn_blocked(self, session_id: str, turn_id: int) -> None:
        """가드레일 차단 턴의 이미 기록된 메시지를 kind=blocked로 바꾸고 검색 색인에서 뺍니다."""
        with self._lock:
            db = self._db()
            rows = db.execute(
                "SELECT id FROM messages WHERE session_id = ? AND turn_id = ? AND kind NOT IN ('blocked')",
                (session_id, turn_id),
            ).fetchall()
            for (rid,) in rows:
                db.execute("DELETE FROM messages_fts WHERE rowid = ?", (rid,))
            db.execute(
                "UPDATE messages SET kind = 'blocked' WHERE session_id = ? AND turn_id = ?", (session_id, turn_id)
            )
            db.commit()

    def record_compaction(
        self,
        session_id: str,
        user_id: str,
        summary: str,
        compacted_uids: Optional[List[str]] = None,
        turn_id: Optional[int] = None,
        replaced_count: Optional[int] = None,
    ) -> None:
        """Auto-Compact 이벤트를 기록합니다 (요약 원문 + 압축된 메시지 uid 목록/개수)."""
        if not session_id or session_id == "unknown":
            return
        uid = f"compaction-{uuid.uuid4().hex[:12]}"
        meta: Dict[str, Any] = {"compacted_uids": compacted_uids or []}
        if replaced_count is not None:
            meta["replaced_count"] = replaced_count
        with self._lock:
            db = self._db()
            seq = int(db.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()[0] or 0) + 1
            db.execute(
                "INSERT INTO messages (session_id, user_id, turn_id, seq, message_uid, role, kind, content, "
                "tool_calls, created_at) VALUES (?, ?, ?, ?, ?, 'system', 'compaction', ?, ?, ?)",
                (session_id, (user_id or "default"), turn_id or self.current_turn(session_id), seq, uid,
                 summary, json.dumps(meta), time.time()),
            )
            db.commit()
            self._known(session_id).add(uid)

    async def arecord_compaction(self, *args, **kwargs) -> None:
        await asyncio.to_thread(self.record_compaction, *args, **kwargs)

    # ── Read ──

    def _row_to_dict(self, r, max_chars: Optional[int]) -> Dict[str, Any]:
        content = r[6] or ""
        limit = self.recall_max_chars if max_chars is None else max_chars
        truncated = bool(limit and len(content) > limit)
        item: Dict[str, Any] = {
            "id": r[0], "turn_id": r[1], "seq": r[2], "role": r[3], "kind": r[4],
            "message_uid": r[11],
            "content": content[:limit] + (f"\n... [{len(content) - limit} chars truncated]" if truncated else "")
            if limit else content,
            "created_at": r[10],
        }
        if r[7]:
            try:
                item["tool_calls"] = json.loads(r[7])
            except Exception:
                item["tool_calls"] = r[7]
        if r[8]:
            item["tool_name"] = r[8]
        if r[9]:
            item["status"] = r[9]
        return item

    _SELECT = (
        "SELECT id, turn_id, seq, role, kind, session_id, content, tool_calls, tool_name, status, created_at, message_uid "
        "FROM messages"
    )

    def read_session(self, session_id: str, user_id: Optional[str] = None,
                     include_hidden: bool = False, max_chars: Optional[int] = 0) -> List[Dict[str, Any]]:
        """세션 원문 전체를 시간순으로 반환합니다 (max_chars=0이면 자르지 않음)."""
        sql = self._SELECT + " WHERE session_id = ?"
        params: List[Any] = [session_id]
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if not include_hidden:
            sql += f" AND kind NOT IN ({','.join('?' * len(HIDDEN_KINDS))})"
            params.extend(HIDDEN_KINDS)
        sql += " ORDER BY seq ASC"
        with self._lock:
            rows = self._db().execute(sql, params).fetchall()
        return [self._row_to_dict(r, max_chars) for r in rows]

    async def aread_session(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.read_session, *args, **kwargs)

    read_session_sync = read_session

    def get_anchored_view(self, session_id: str, anchor_keyword: str = "", window: int = 5,
                          user_id: str = "default", around_message_id: int = 0) -> List[Dict[str, Any]]:
        """기준 메시지 앞뒤 ±window + 세션 앞/뒤 3개(bookend)를 반환합니다 (user_id 소유 검증 포함).

        기준 메시지는 자르기 전 원문 전체에서 찾습니다. 반환할 때만 메시지당 recall_max_chars로 자르며,
        긴 메시지에서 키워드가 뒤쪽에 있으면 키워드 주변을 잘라 보여 줍니다.
        """
        uid = (user_id or "default").strip() or "default"
        messages = self.read_session(session_id, user_id=uid, max_chars=0)
        if not messages:
            return []
        words = [w for w in (anchor_keyword or "").lower().split() if w]
        if around_message_id:
            idx = next((i for i, m in enumerate(messages) if m["id"] == around_message_id), len(messages) - 1)
        elif words:
            idx = self._find_anchor(messages, anchor_keyword)
        else:
            return [self._clip_for_recall(m, words) for m in messages[-window:]]
        core = messages[max(0, idx - window): idx + window + 1]
        seen, result = set(), []
        for m in messages[:3] + core + messages[-3:]:
            if m["id"] not in seen:
                seen.add(m["id"])
                result.append(m)
        result.sort(key=lambda m: m["seq"])
        return [self._clip_for_recall(m, words) for m in result]

    def _clip_for_recall(self, item: Dict[str, Any], words: List[str]) -> Dict[str, Any]:
        """회상 결과용으로 메시지 본문을 recall_max_chars 이내로 자릅니다 (키워드가 뒤쪽이면 그 주변을 보여 줌)."""
        limit = self.recall_max_chars
        content = item.get("content") or ""
        if not limit or len(content) <= limit:
            return item
        lowered = content.lower()
        hit = min((p for p in (lowered.find(w) for w in words) if p >= 0), default=-1)
        if hit < 0 or hit < limit - 200:
            clipped = content[:limit] + f"\n... [{len(content) - limit} chars truncated]"
        else:
            start = max(0, min(hit - limit // 2, len(content) - limit))
            end = start + limit
            clipped = (f"[... {start} chars omitted ...]\n" if start else "") + content[start:end]
            if end < len(content):
                clipped += f"\n... [{len(content) - end} chars truncated]"
        return {**item, "content": clipped}

    async def aget_anchored_view(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.get_anchored_view, *args, **kwargs)

    @staticmethod
    def _find_anchor(messages: List[Dict[str, Any]], keyword: str) -> int:
        words = [w for w in keyword.lower().split() if w]
        best_idx, best_score = len(messages) - 1, 0
        for i, m in enumerate(messages):
            text = (m.get("content") or "").lower()
            if m.get("tool_calls"):
                text += " " + json.dumps(m["tool_calls"], ensure_ascii=False).lower()
            score = sum(1 for w in words if w in text)
            if score > best_score:
                best_idx, best_score = i, score
        return best_idx

    def search(self, query: str, user_id: str = "default", limit: int = 5,
               exclude_session_id: Optional[str] = None, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """여러 세션에 걸친 원문 전문 검색 (사용자 질문·AI 답변·도구 호출만 대상)."""
        if top_k is not None:
            limit = top_k
        fts = _fts_query(query)
        if not fts:
            return []
        uid = (user_id or "default").strip() or "default"
        sql = (
            "SELECT m.session_id, m.id, m.role, m.created_at, "
            "snippet(messages_fts, 0, '>>>', '<<<', '...', 24) "
            "FROM messages_fts JOIN messages m ON messages_fts.rowid = m.id "
            "WHERE messages_fts MATCH ? AND m.user_id = ? "
            f"AND m.kind NOT IN ({','.join('?' * len(HIDDEN_KINDS))}) "
            "ORDER BY rank LIMIT ?"
        )
        try:
            with self._lock:
                rows = self._db().execute(sql, (fts, uid, *HIDDEN_KINDS, limit * 5)).fetchall()
        except sqlite3.Error as e:
            logger.warning("Transcript search failed: %s", e)
            return []
        results, seen = [], set()
        for sid, mid, role, created, snip in rows:
            if sid == exclude_session_id or sid in seen:
                continue
            seen.add(sid)
            results.append({"session_id": sid, "message_id": mid, "role": role,
                            "created_at": created, "snippet": snip or ""})
            if len(results) >= limit:
                break
        return results

    async def asearch(self, *args, **kwargs) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self.search, *args, **kwargs)

    search_sync = search

    def session_tool_names(self, session_id: str) -> List[str]:
        """세션에서 사용한 도구 이름 목록 (등장 순서, 중복 제거)."""
        with self._lock:
            rows = self._db().execute(
                "SELECT tool_calls FROM messages WHERE session_id = ? AND kind = 'assistant' "
                "AND tool_calls IS NOT NULL ORDER BY seq", (session_id,)
            ).fetchall()
        names: List[str] = []
        for (raw,) in rows:
            try:
                for tc in json.loads(raw):
                    n = tc.get("name")
                    if n and n not in names:
                        names.append(n)
            except Exception:
                continue
        return names

    def turn_kinds(self, session_id: str, turn_id: int) -> List[str]:
        with self._lock:
            rows = self._db().execute(
                "SELECT kind FROM messages WHERE session_id = ? AND turn_id = ?", (session_id, turn_id)
            ).fetchall()
        return [r[0] for r in rows]


def render_transcript_for_summary(messages: List[Dict[str, Any]], budget_chars: int = 24000,
                                  per_message_chars: int = 1500, max_chars: Optional[int] = None) -> str:
    """요약 LLM 입력용으로 원문을 시간순 텍스트로 렌더링합니다.

    - 사용자/AI 메시지는 메시지당 per_message_chars로 줄이고, 도구 호출은 한 줄 요약으로 표시합니다.
    - 전체가 budget_chars를 넘으면 앞부분 30% + 최근 70%를 남기고 가운데를 생략합니다.
    """
    if max_chars is not None:
        budget_chars = max_chars
    lines: List[str] = []
    for m in messages:
        if (m.get("additional_kwargs") or {}).get(TRANSCRIPT_SKIP_KEY) or is_compaction_artifact(m):
            continue
        kind = m.get("kind") or {
            "user": "user", "human": "user",
            "assistant": "assistant", "ai": "assistant",
            "tool": "tool_result",
        }.get(m.get("role", ""))
        if kind in HIDDEN_KINDS:
            continue
        content = (m.get("content") or "").strip()
        if kind == "user":
            lines.append(f"USER: {_short(content, per_message_chars)}")
        elif kind == "assistant":
            if content:
                lines.append(f"ASSISTANT: {_short(content, per_message_chars)}")
            for tc in m.get("tool_calls") or []:
                if isinstance(tc, dict):
                    lines.append(f"  -> [tool call] {format_tool_call(tc)}")
                elif isinstance(tc, str):
                    lines.append(f"  -> [tool call] {tc}")
        elif kind == "tool_result":
            status = m.get("status") or "success"
            lines.append(f"  <- [tool result: {m.get('tool_name') or 'tool'}, {status}] {_short(content, 200)}")
    total = sum(len(l) + 1 for l in lines)
    if total <= budget_chars:
        return "\n".join(lines)
    head_budget, tail_budget = int(budget_chars * 0.3), int(budget_chars * 0.7)
    head, used = [], 0
    for l in lines:
        if used + len(l) + 1 > head_budget:
            break
        head.append(l)
        used += len(l) + 1
    tail, used = [], 0
    for l in reversed(lines[len(head):]):
        if used + len(l) + 1 > tail_budget:
            break
        tail.append(l)
        used += len(l) + 1
    tail.reverse()
    omitted = len(lines) - len(head) - len(tail)
    return "\n".join(head + [f"... [{omitted} lines omitted for length] ..."] + tail)

