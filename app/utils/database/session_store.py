"""
session_store.py — FastAPI용 비동기 세션 메타데이터 저장소 및 공유 Checkpointer(SSOT) 팩토리

설계 특징 (Phase 1 — DB 통합):
1. 별도 `chat_history.db` 및 `chat_messages` 이중 기록(Dual-Write)을 전면 폐기하고,
   `app/database/checkpoints.db` 단일 SQLite 파일로 세션 메타데이터(`chat_sessions`, `session_options`)와
   LangGraph 대화 상태(`checkpoints`, `writes`)를 통합
2. 대화 메시지 조회(`get_messages`)는 LangGraph `AsyncSqliteSaver`(`aget_tuple`)를 단일 진실 공급원(SSOT)으로 직접 읽음
3. 세션 삭제(`delete_session`) 시 메타데이터와 체크포인트(`checkpoints`, `writes`)를 원자적으로 함께 정리
4. SQLite WAL(Write-Ahead Logging) 모드 및 busy_timeout(5000ms) 적용으로 동시성 락 경합 해소
"""

import sqlite3
import os
import json
import time
import asyncio
import threading
import weakref
import aiosqlite
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# ── DB 경로 설정 (checkpoints.db 단일 통합) ───────────────────
BASE_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "database",
)
_DEFAULT_DB_PATH = os.path.join(BASE_DB_DIR, "checkpoints.db")
CHECKPOINTS_DB_PATH = _DEFAULT_DB_PATH
DB_PATH = _DEFAULT_DB_PATH

_init_lock = threading.Lock()
_initialized_db_paths: set = set()


def _resolve_db_path(db_path: Optional[str] = None) -> str:
    """명시 인자 > 패치된 CHECKPOINTS_DB_PATH > 패치된 DB_PATH > 기본값 순으로 DB 경로를 결정합니다."""
    if db_path:
        return os.path.abspath(db_path)
    if CHECKPOINTS_DB_PATH != _DEFAULT_DB_PATH:
        return os.path.abspath(CHECKPOINTS_DB_PATH)
    if DB_PATH != _DEFAULT_DB_PATH:
        return os.path.abspath(DB_PATH)
    return os.path.abspath(_DEFAULT_DB_PATH)


def _connect_sync(db_path: Optional[str] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(_resolve_db_path(db_path), timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def init_db(force: bool = False):
    """세션 메타데이터(`chat_sessions`) 및 실행 옵션(`session_options`) 테이블을 최초 1회 생성하고 WAL 모드를 활성화합니다."""
    target_db = _resolve_db_path()
    if not force and target_db in _initialized_db_paths and os.path.exists(target_db):
        return

    with _init_lock:
        if not force and target_db in _initialized_db_paths and os.path.exists(target_db):
            return

        os.makedirs(os.path.dirname(target_db) or ".", exist_ok=True)
        with _connect_sync(target_db) as conn:
            cursor = conn.cursor()
            # 1. 대화방(세션) 메타데이터 테이블 (user_id 포함)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL DEFAULT 'default',
                    agent_name TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            # 기존 테이블에 user_id 컬럼이 없는 경우 자동 마이그레이션
            cols = {row[1] for row in cursor.execute("PRAGMA table_info(chat_sessions)").fetchall()}
            if "user_id" not in cols:
                cursor.execute("ALTER TABLE chat_sessions ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'")

            # 2. 부모 세션 ↔ 서브에이전트 스레드 명시적 소속 링크 테이블 (정규식/접두사 추측 배제)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS thread_links (
                    thread_id TEXT PRIMARY KEY,
                    parent_thread_id TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
            """)
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_thread_links_parent ON thread_links(parent_thread_id)"
            )
            conn.commit()
        _initialized_db_paths.add(target_db)


# =============================================================================
# 공유 LangGraph Checkpointer (AsyncSqliteSaver + WAL, 대화 메시지 SSOT)
# =============================================================================
_checkpointer_by_key: Dict[Tuple[int, str], AsyncSqliteSaver] = {}
_checkpointer_conn_by_key: Dict[Tuple[int, str], aiosqlite.Connection] = {}
_checkpointer_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def _is_conn_alive(conn: Optional[aiosqlite.Connection]) -> bool:
    """aiosqlite 커넥션이 아직 열려 실행 중인지 확인합니다."""
    return conn is not None and bool(getattr(conn, "_running", False))


def _get_loop_lock(loop: asyncio.AbstractEventLoop) -> asyncio.Lock:
    """이벤트 루프별 asyncio.Lock을 반환하며, 루프 GC 시 자동 해제됩니다."""
    lock = _checkpointer_locks.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _checkpointer_locks[loop] = lock
    return lock


def _is_entry_usable(cache_key: Tuple[int, str], loop: asyncio.AbstractEventLoop) -> bool:
    """캐시된 checkpointer와 커넥션이 현재 이벤트 루프에서 재사용 가능한지 확인합니다."""
    saver = _checkpointer_by_key.get(cache_key)
    conn = _checkpointer_conn_by_key.get(cache_key)
    if saver is None or not _is_conn_alive(conn):
        return False
    saver_loop = getattr(saver, "loop", None)
    if saver_loop is not None and (saver_loop is not loop or saver_loop.is_closed()):
        return False
    return True


def _prune_stale_entries(current_loop: asyncio.AbstractEventLoop) -> None:
    """이미 닫힌 이전 이벤트 루프나 종료된 커넥션에 묶인 캐시 항목을 정리합니다."""
    current_loop_id = id(current_loop)
    for k in list(_checkpointer_by_key.keys()):
        conn = _checkpointer_conn_by_key.get(k)
        saver = _checkpointer_by_key.get(k)
        saver_loop = getattr(saver, "loop", None)
        if saver_loop is not None and saver_loop.is_closed():
            stale = True
        elif k[0] == current_loop_id:
            stale = saver_loop is not None and saver_loop is not current_loop
        else:
            stale = not _is_conn_alive(conn)
        if stale:
            _checkpointer_by_key.pop(k, None)
            _checkpointer_conn_by_key.pop(k, None)


async def get_shared_checkpointer(db_path: Optional[str] = None) -> AsyncSqliteSaver:
    """현재 비동기 이벤트 루프에서 공유되는 단일 AsyncSqliteSaver 인스턴스를 반환합니다.
    에이전트마다 별도 커넥션을 열지 않고 WAL 모드가 활성화된 단일 커넥션을 공유합니다.
    """
    target_path = _resolve_db_path(db_path)
    loop = asyncio.get_running_loop()
    loop_id = id(loop)
    cache_key = (loop_id, target_path)

    async with _get_loop_lock(loop):
        _prune_stale_entries(loop)
        if _is_entry_usable(cache_key, loop):
            return _checkpointer_by_key[cache_key]

        _checkpointer_by_key.pop(cache_key, None)
        _checkpointer_conn_by_key.pop(cache_key, None)

        os.makedirs(os.path.dirname(target_path) or ".", exist_ok=True)
        conn = await aiosqlite.connect(target_path, check_same_thread=False)
        await conn.execute("PRAGMA journal_mode=WAL;")
        await conn.execute("PRAGMA synchronous=NORMAL;")
        await conn.execute("PRAGMA busy_timeout=5000;")
        await conn.commit()

        saver = AsyncSqliteSaver(conn)
        await saver.setup()

        _checkpointer_by_key[cache_key] = saver
        _checkpointer_conn_by_key[cache_key] = conn
        return saver


async def close_shared_checkpointer() -> None:
    """현재 이벤트 루프에 생성된 공유 Checkpointer DB 커넥션을 안전하게 닫습니다."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        for k in list(_checkpointer_conn_by_key.keys()):
            _checkpointer_by_key.pop(k, None)
            conn = _checkpointer_conn_by_key.pop(k, None)
            if _is_conn_alive(conn):
                await conn.close()
        return

    loop_id = id(loop)
    async with _get_loop_lock(loop):
        keys_to_close = [k for k in list(_checkpointer_conn_by_key.keys()) if k[0] == loop_id]
        for k in keys_to_close:
            _checkpointer_by_key.pop(k, None)
            conn = _checkpointer_conn_by_key.pop(k, None)
            if _is_conn_alive(conn):
                await conn.close()
        _prune_stale_entries(loop)


# =============================================================================
# 비동기 CRUD (FastAPI server.py 단일 논블로킹 경로)
# =============================================================================

async def _connect_async() -> aiosqlite.Connection:
    init_db()
    conn = await aiosqlite.connect(_resolve_db_path(), timeout=10.0)
    await conn.execute("PRAGMA journal_mode=WAL;")
    await conn.execute("PRAGMA synchronous=NORMAL;")
    await conn.execute("PRAGMA busy_timeout=5000;")
    return conn


async def register_subagent_thread(thread_id: str, parent_thread_id: str) -> None:
    """서브에이전트 스레드(`thread_id`)와 부모 세션(`parent_thread_id`)의 명시적 소속 관계를 기록합니다."""
    tid = (thread_id or "").strip()
    ptid = (parent_thread_id or "").strip()
    if not tid or not ptid or tid == ptid:
        return
    conn = await _connect_async()
    try:
        await conn.execute(
            "INSERT OR IGNORE INTO thread_links (thread_id, parent_thread_id, created_at) VALUES (?, ?, ?)",
            (tid, ptid, time.time()),
        )
        await conn.commit()
    finally:
        await conn.close()


async def create_session(session_id: str, agent_name: str, title: str, user_id: str = "default") -> Dict[str, Any]:
    """대화 세션을 생성합니다.
    이미 존재하는 session_id인 경우 created_at과 최초 소유자(user_id)는 보존하고 title만 갱신합니다.
    """
    created_at = time.time()
    uid = (user_id or "default").strip() or "default"
    conn = await _connect_async()
    try:
        conn.row_factory = aiosqlite.Row
        await conn.execute(
            "INSERT INTO chat_sessions (session_id, user_id, agent_name, title, created_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET title = excluded.title",
            (session_id, uid, agent_name, title, created_at),
        )
        await conn.commit()
        rows = await conn.execute_fetchall(
            "SELECT session_id, user_id, agent_name, title, created_at FROM chat_sessions WHERE session_id = ?",
            (session_id,),
        )
        if rows:
            return dict(rows[0])
    finally:
        await conn.close()
    return {
        "session_id": session_id,
        "user_id": uid,
        "agent_name": agent_name,
        "title": title,
        "created_at": created_at,
    }


async def get_sessions(agent_name: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = await _connect_async()
    try:
        conn.row_factory = aiosqlite.Row
        query = "SELECT * FROM chat_sessions WHERE 1=1"
        params: list = []
        if agent_name:
            query += " AND agent_name = ?"
            params.append(agent_name)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        query += " ORDER BY created_at DESC"
        rows = await conn.execute_fetchall(query, tuple(params))
        return [dict(row) for row in rows]
    finally:
        await conn.close()


async def delete_session(session_id: str) -> None:
    """세션 메타데이터(`chat_sessions`, `session_options`), 부모-자식 링크(`thread_links`),
    그리고 LangGraph 체크포인트 상태(메인 `session_id` 및 `thread_links`에 명시적으로 등록된 하위 서브에이전트 스레드)를 함께 삭제합니다.
    이름 접두사(`LIKE`/`startswith`)로 추측하지 않고 `thread_links` 테이블의 명시적 관계만 사용합니다.

    보존 정책(Retention Policy):
    - 사용자 장기 기억(`app/database/episodic.db`, `USER.md`, `MEMORY.md`)과
      디스크 산출물(`artifacts/users/{uid}/sessions/{session_id}/`)은 대화방을 삭제하더라도 보존됩니다.
    """
    sub_thread_ids: List[str] = []
    conn = await _connect_async()
    try:
        queue = [session_id]
        seen = {session_id}
        while queue:
            current_parent = queue.pop(0)
            rows = await conn.execute_fetchall(
                "SELECT thread_id FROM thread_links WHERE parent_thread_id = ?",
                (current_parent,),
            )
            for r in rows:
                child_tid = str(r[0]) if r and r[0] else ""
                if child_tid and child_tid not in seen:
                    seen.add(child_tid)
                    sub_thread_ids.append(child_tid)
                    queue.append(child_tid)

        all_tids = [session_id, *sub_thread_ids]
        await conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        for tid in all_tids:
            await conn.execute(
                "DELETE FROM thread_links WHERE parent_thread_id = ? OR thread_id = ?",
                (tid, tid),
            )
        await conn.commit()
    finally:
        await conn.close()

    saver = await get_shared_checkpointer()
    for tid in [session_id, *sub_thread_ids]:
        await saver.adelete_thread(tid)


async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    conn = await _connect_async()
    try:
        conn.row_factory = aiosqlite.Row
        rows = await conn.execute_fetchall("SELECT * FROM chat_sessions WHERE session_id = ?", (session_id,))
        return dict(rows[0]) if rows else None
    finally:
        await conn.close()


async def get_messages(session_id: str) -> List[Dict[str, str]]:
    """LangGraph Checkpointer(`checkpoints.db`)의 최신 체크포인트 상태에서
    해당 `session_id`(`thread_id`)의 대화 메시지 목록(`[{"role": "user"|"assistant", "content": ...}]`)을 반환합니다 (SSOT).

    필터링 규칙:
    - 도구 호출이 포함된 중간 AI 발화(`tool_calls`가 비어 있지 않은 `AIMessage`)는 제외하고 최종 응답만 포함합니다.
    - 백그라운드 잡 완료 시 서버가 내부적으로 주입하는 `[SYSTEM NOTIFICATION:` 트리거 메시지는 제외합니다.
    """
    saver = await get_shared_checkpointer()
    tup = await saver.aget_tuple({"configurable": {"thread_id": session_id}})
    if not tup or not tup.checkpoint:
        return []

    raw_messages = (tup.checkpoint.get("channel_values") or {}).get("messages", [])
    if not raw_messages:
        return []

    from app.utils.message_utils import normalize_content, sanitize_text

    formatted: List[Dict[str, str]] = []
    for msg in raw_messages:
        if isinstance(msg, dict):
            msg_type = msg.get("type") or msg.get("role")
            raw_content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")
        else:
            msg_type = getattr(msg, "type", None)
            raw_content = getattr(msg, "content", "")
            tool_calls = getattr(msg, "tool_calls", None)

        text = sanitize_text(normalize_content(raw_content or "")).strip()
        if not text:
            continue

        if msg_type in ("human", "user"):
            if text.startswith("[SYSTEM NOTIFICATION:"):
                continue
            formatted.append({"role": "user", "content": text})
        elif msg_type in ("ai", "assistant"):
            if tool_calls:
                continue
            formatted.append({"role": "assistant", "content": text})

    return formatted


# 명시적 a* 별칭 (하위 호환 유지)
acreate_session = create_session
aget_sessions = get_sessions
adelete_session = delete_session
aget_session = get_session
aget_messages = get_messages

