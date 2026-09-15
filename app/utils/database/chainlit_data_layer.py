"""
chainlit_data_layer.py — Chainlit 전용 SQLite Data Layer

Chainlit의 BaseDataLayer를 상속하여 SQLite 기반으로 구현한 데이터 레이어입니다.
사이드바 대화 목록, 세션 복원, 메시지/도구 호출 단계(Step) 저장을 담당합니다.

사용처: app/chainlit_ui.py 에서 @cl.data_layer 로 등록
"""

import os
import json
import uuid
import aiosqlite
from datetime import datetime
from typing import Dict, List, Optional

from chainlit.data.base import BaseDataLayer
from chainlit.types import (
    Feedback,
    PaginatedResponse,
    Pagination,
    PageInfo,
    ThreadDict,
    ThreadFilter,
)
from chainlit.user import PersistedUser, User
from chainlit.step import StepDict
from chainlit.element import Element, ElementDict

# ── DB 경로 설정 ──────────────────────────────────────────────
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "database")
CHAINLIT_DB_PATH = os.path.join(DB_DIR, "chainlit_history.db")


class ChainlitSQLiteDataLayer(BaseDataLayer):
    """Chainlit 사이드바·히스토리·Step 영구 저장을 위한 SQLite Data Layer.
    
    Chainlit의 BaseDataLayer 인터페이스를 구현하며,
    공식 SQLAlchemyDataLayer가 테이블 자동 생성을 지원하지 않아
    직접 스키마 초기화(_init_tables)를 포함합니다.
    """

    def __init__(self, db_path: str = CHAINLIT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_tables()

    # ── 헬퍼 ──────────────────────────────────────────────────

    def _connect(self):
        """aiosqlite 비동기 DB 연결을 반환합니다. async with 와 함께 사용."""
        return aiosqlite.connect(self.db_path)

    def _init_tables(self):
        """최초 실행 시 필요한 테이블을 생성합니다 (CREATE IF NOT EXISTS)."""
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    identifier TEXT NOT NULL UNIQUE,
                    createdAt TEXT,
                    metadata TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    createdAt TEXT,
                    name TEXT,
                    userId TEXT,
                    userIdentifier TEXT,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS steps (
                    id TEXT PRIMARY KEY,
                    name TEXT,
                    type TEXT,
                    threadId TEXT,
                    parentId TEXT,
                    streaming INTEGER DEFAULT 0,
                    waitForAnswer INTEGER DEFAULT 0,
                    isError INTEGER DEFAULT 0,
                    metadata TEXT,
                    tags TEXT,
                    input TEXT,
                    output TEXT,
                    createdAt TEXT,
                    start TEXT,
                    end TEXT,
                    generation TEXT,
                    showInput TEXT,
                    language TEXT,
                    feedback TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS elements (
                    id TEXT PRIMARY KEY,
                    threadId TEXT,
                    type TEXT,
                    chainlitKey TEXT,
                    url TEXT,
                    objectKey TEXT,
                    name TEXT,
                    display TEXT,
                    size TEXT,
                    language TEXT,
                    autoPlay INTEGER,
                    playerConfig TEXT,
                    page INTEGER,
                    props TEXT,
                    forId TEXT,
                    mime TEXT
                )
            """)
            c.execute("""
                CREATE TABLE IF NOT EXISTS feedbacks (
                    id TEXT PRIMARY KEY,
                    forId TEXT NOT NULL,
                    threadId TEXT,
                    value INTEGER,
                    comment TEXT
                )
            """)
            conn.commit()

    # ── 사용자(User) CRUD ─────────────────────────────────────

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        """로그인 식별자(identifier)로 사용자를 조회합니다."""
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM users WHERE identifier = ?", (identifier,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return PersistedUser(
                        id=row["id"],
                        identifier=row["identifier"],
                        createdAt=row["createdAt"],
                        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                    )
        return None

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        """사용자를 생성하거나, 이미 존재하면 metadata를 갱신합니다."""
        user_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat() + "Z"
        meta_str = json.dumps(user.metadata or {})

        async with self._connect() as db:
            await db.execute(
                """INSERT INTO users (id, identifier, createdAt, metadata)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(identifier) DO UPDATE SET metadata=excluded.metadata""",
                (user_id, user.identifier, created_at, meta_str),
            )
            await db.commit()

        return await self.get_user(user.identifier)

    # ── 피드백(Feedback) ──────────────────────────────────────

    async def upsert_feedback(self, feedback: Feedback) -> str:
        """피드백을 저장하거나 갱신합니다."""
        feedback_id = feedback.id or str(uuid.uuid4())
        async with self._connect() as db:
            await db.execute(
                """INSERT INTO feedbacks (id, forId, threadId, value, comment)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET value=excluded.value, comment=excluded.comment""",
                (feedback_id, feedback.forId, feedback.threadId, feedback.value, feedback.comment),
            )
            await db.commit()
        return feedback_id

    async def delete_feedback(self, feedback_id: str) -> bool:
        """피드백을 삭제합니다."""
        async with self._connect() as db:
            await db.execute("DELETE FROM feedbacks WHERE id = ?", (feedback_id,))
            await db.commit()
        return True

    # ── 요소(Element: 이미지/파일 등) ─────────────────────────

    async def create_element(self, element: Element):
        """Element(이미지, 파일, CustomElement 등)를 저장합니다."""
        props_val = getattr(element, "props", None)
        props_str = json.dumps(props_val) if props_val is not None else None
        mime_val = getattr(element, "mime", None) or "application/octet-stream"

        async with self._connect() as db:
            await db.execute(
                """INSERT INTO elements (id, threadId, type, chainlitKey, url, objectKey,
                                         name, display, size, language, mime, forId, props)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET 
                       url=excluded.url, display=excluded.display, props=excluded.props, mime=excluded.mime""",
                (
                    element.id, element.thread_id, element.type,
                    getattr(element, "chainlit_key", None),
                    element.url, getattr(element, "object_key", None),
                    element.name, element.display,
                    getattr(element, "size", None), getattr(element, "language", None),
                    mime_val, getattr(element, "for_id", None),
                    props_str,
                ),
            )
            await db.commit()

    async def get_element(self, thread_id: str, element_id: str) -> Optional[ElementDict]:
        """특정 쓰레드의 Element를 조회합니다."""
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM elements WHERE id = ? AND threadId = ?",
                (element_id, thread_id),
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    props_dict = json.loads(row["props"]) if row["props"] else None
                    return ElementDict(
                        id=row["id"], threadId=row["threadId"], type=row["type"],
                        name=row["name"], url=row["url"], display=row["display"],
                        forId=row["forId"], mime=row["mime"] or "application/octet-stream",
                        props=props_dict,
                    )
        return None

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        """Element를 삭제합니다."""
        async with self._connect() as db:
            await db.execute("DELETE FROM elements WHERE id = ?", (element_id,))
            await db.commit()

    # ── 단계(Step: 사용자 메시지, 도구 호출 등) ───────────────

    async def create_step(self, step_dict: StepDict):
        """Step(메시지·도구 호출 단계)을 저장합니다.
        
        Note: 쓰레드 자동 생성은 하지 않습니다.
              Chainlit이 update_thread()를 통해 userId와 함께 쓰레드를 먼저 생성합니다.
        """
        thread_id = step_dict.get("threadId")
        if not thread_id:
            return

        created_at = datetime.utcnow().isoformat() + "Z"

        async with self._connect() as db:
            await db.execute(
                """INSERT INTO steps
                   (id, name, type, threadId, parentId, streaming, waitForAnswer,
                    isError, metadata, tags, input, output, createdAt, start, end,
                    generation, showInput, language)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       output=excluded.output, input=excluded.input,
                       end=excluded.end, metadata=excluded.metadata,
                       isError=excluded.isError""",
                (
                    step_dict.get("id"),
                    step_dict.get("name"),
                    step_dict.get("type"),
                    thread_id,
                    step_dict.get("parentId"),
                    1 if step_dict.get("streaming") else 0,
                    1 if step_dict.get("waitForAnswer") else 0,
                    1 if step_dict.get("isError") else 0,
                    json.dumps(step_dict.get("metadata") or {}),
                    json.dumps(step_dict.get("tags") or []),
                    str(step_dict.get("input") or ""),
                    str(step_dict.get("output") or ""),
                    step_dict.get("createdAt") or created_at,
                    step_dict.get("start"),
                    step_dict.get("end"),
                    json.dumps(step_dict.get("generation") or {}),
                    str(step_dict.get("showInput") or ""),
                    step_dict.get("language"),
                ),
            )
            await db.commit()

    async def update_step(self, step_dict: StepDict):
        """Step을 갱신합니다 (create_step과 동일한 UPSERT 로직)."""
        await self.create_step(step_dict)

    async def delete_step(self, step_id: str):
        """Step을 삭제합니다."""
        async with self._connect() as db:
            await db.execute("DELETE FROM steps WHERE id = ?", (step_id,))
            await db.commit()

    # ── 쓰레드(Thread: 대화방) ────────────────────────────────

    async def get_thread_author(self, thread_id: str) -> str:
        """쓰레드 소유자의 식별자를 반환합니다.
        
        Chainlit이 chat resume 시 호출하여 현재 로그인 사용자와 비교합니다.
        불일치 시 'Authorization for the thread failed' 에러가 발생합니다.
        """
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT userIdentifier, userId FROM threads WHERE id = ?", (thread_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return row["userIdentifier"] or row["userId"] or ""
        return ""

    async def delete_thread(self, thread_id: str):
        """쓰레드와 관련된 모든 데이터(steps, elements)를 삭제합니다."""
        async with self._connect() as db:
            await db.execute("DELETE FROM steps WHERE threadId = ?", (thread_id,))
            await db.execute("DELETE FROM elements WHERE threadId = ?", (thread_id,))
            await db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            await db.commit()

    async def list_threads(
        self, pagination: Pagination, filters: ThreadFilter
    ) -> PaginatedResponse[ThreadDict]:
        """사이드바에 표시할 대화방 목록을 조회합니다.
        
        userId 또는 userIdentifier가 일치하는 쓰레드만 반환합니다.
        """
        user_id = getattr(filters, "userId", None)
        search = getattr(filters, "search", None)

        query = "SELECT * FROM threads WHERE 1=1"
        params: list = []

        if user_id:
            query += " AND (userId = ? OR userIdentifier = ?)"
            params.extend([user_id, user_id])
        if search:
            query += " AND name LIKE ?"
            params.append(f"%{search}%")

        query += " ORDER BY createdAt DESC"

        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(query, params) as cursor:
                rows = await cursor.fetchall()

        threads: List[ThreadDict] = []
        for r in rows:
            threads.append(
                ThreadDict(
                    id=r["id"],
                    createdAt=r["createdAt"],
                    name=r["name"] or "새 대화",
                    userId=r["userId"],
                    userIdentifier=r["userIdentifier"],
                    tags=json.loads(r["tags"]) if r["tags"] else [],
                    metadata=json.loads(r["metadata"]) if r["metadata"] else {},
                    steps=[],
                    elements=[],
                )
            )

        return PaginatedResponse(
            data=threads,
            pageInfo=PageInfo(
                hasNextPage=False,
                startCursor=threads[0]["id"] if threads else None,
                endCursor=threads[-1]["id"] if threads else None,
            ),
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        """특정 쓰레드의 전체 데이터(steps, elements 포함)를 조회합니다.
        
        사이드바에서 기존 대화 클릭 시(chat resume) Chainlit이 호출합니다.
        """
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row

            # 1. 쓰레드 기본 정보 조회
            async with db.execute(
                "SELECT * FROM threads WHERE id = ?", (thread_id,)
            ) as cursor:
                t_row = await cursor.fetchone()
                if not t_row:
                    return None

            # 2. 해당 쓰레드의 모든 Steps 조회 (시간순 정렬)
            async with db.execute(
                "SELECT * FROM steps WHERE threadId = ? ORDER BY createdAt ASC",
                (thread_id,),
            ) as cursor:
                s_rows = await cursor.fetchall()

            # 3. 해당 쓰레드의 모든 Elements 조회
            async with db.execute(
                "SELECT * FROM elements WHERE threadId = ?", (thread_id,)
            ) as cursor:
                e_rows = await cursor.fetchall()

        # Step 데이터 변환
        steps: List[StepDict] = []
        for s in s_rows:
            steps.append(
                StepDict(
                    id=s["id"],
                    name=s["name"] or "step",
                    type=s["type"] or "run",
                    threadId=s["threadId"],
                    parentId=s["parentId"],
                    streaming=bool(s["streaming"]),
                    waitForAnswer=bool(s["waitForAnswer"]),
                    isError=bool(s["isError"]),
                    metadata=json.loads(s["metadata"]) if s["metadata"] else {},
                    tags=json.loads(s["tags"]) if s["tags"] else [],
                    input=s["input"] or "",
                    output=s["output"] or "",
                    createdAt=s["createdAt"],
                    start=s["start"],
                    end=s["end"],
                    language=s["language"],
                )
            )

        # Element 데이터 변환
        elements: List[ElementDict] = []
        for e in e_rows:
            props_dict = json.loads(e["props"]) if ("props" in e.keys() and e["props"]) else None
            mime_str = e["mime"] if ("mime" in e.keys() and e["mime"]) else "application/octet-stream"
            elements.append(
                ElementDict(
                    id=e["id"], threadId=e["threadId"], type=e["type"],
                    name=e["name"], url=e["url"], display=e["display"],
                    forId=e["forId"], mime=mime_str,
                    props=props_dict,
                )
            )

        return ThreadDict(
            id=t_row["id"],
            createdAt=t_row["createdAt"],
            name=t_row["name"] or "새 대화",
            userId=t_row["userId"],
            userIdentifier=t_row["userIdentifier"],
            tags=json.loads(t_row["tags"]) if t_row["tags"] else [],
            metadata=json.loads(t_row["metadata"]) if t_row["metadata"] else {},
            steps=steps,
            elements=elements,
        )

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        """쓰레드를 생성하거나 갱신합니다.
        
        Chainlit이 새 대화 시작 시 userId(UUID)와 함께 호출합니다.
        userIdentifier에는 사용자 식별자 문자열(예: 'admin')을 저장해야
        get_thread_author()의 인증 비교가 정상 동작합니다.
        """
        created_at = datetime.utcnow().isoformat() + "Z"
        tags_str = json.dumps(tags or [])
        meta_str = json.dumps(metadata or {})

        # user_id(UUID)로부터 실제 사용자 식별자(identifier) 문자열을 조회
        user_identifier = None
        if user_id:
            async with self._connect() as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(
                    "SELECT identifier FROM users WHERE id = ?", (user_id,)
                ) as cursor:
                    row = await cursor.fetchone()
                    if row:
                        user_identifier = row["identifier"]
            # 조회 실패 시 user_id를 그대로 사용 (폴백)
            if not user_identifier:
                user_identifier = user_id

        async with self._connect() as db:
            await db.execute(
                """INSERT INTO threads (id, createdAt, name, userId, userIdentifier, tags, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       name = CASE WHEN ? IS NOT NULL THEN ? ELSE threads.name END,
                       userId = CASE WHEN ? IS NOT NULL THEN ? ELSE threads.userId END,
                       userIdentifier = CASE WHEN ? IS NOT NULL THEN ? ELSE threads.userIdentifier END,
                       tags = CASE WHEN ? IS NOT NULL THEN ? ELSE threads.tags END,
                       metadata = CASE WHEN ? IS NOT NULL THEN ? ELSE threads.metadata END""",
                (
                    thread_id, created_at, name or "새 대화",
                    user_id, user_identifier,
                    tags_str, meta_str,
                    # ON CONFLICT UPDATE 파라미터
                    name, name,
                    user_id, user_id,
                    user_identifier, user_identifier,
                    json.dumps(tags) if tags is not None else None,
                    json.dumps(tags) if tags is not None else None,
                    json.dumps(metadata) if metadata is not None else None,
                    json.dumps(metadata) if metadata is not None else None,
                ),
            )
            await db.commit()

    # ── 기타 인터페이스 구현 ──────────────────────────────────

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        pass

    async def get_favorite_steps(self, user_id: str) -> List[StepDict]:
        return []
