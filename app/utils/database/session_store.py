"""
session_store.py — Streamlit/FastAPI용 세션·메시지 SQLite 저장소

FastAPI server.py에서 사용하는 대화 세션(chat_sessions) 및
메시지(chat_messages) 테이블의 CRUD 함수를 제공합니다.

사용처: app/server.py
"""

import sqlite3
import os
import time
from typing import List, Dict, Any, Optional

# ── DB 경로 설정 ──────────────────────────────────────────────
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "database", "chat_history.db"
)


def init_db():
    """세션·메시지 테이블이 없으면 생성합니다."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # 1. 대화방(세션) 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                agent_name TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at REAL NOT NULL
            )
        """)
        # 2. 메시지 테이블
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
            )
        """)
        conn.commit()


def create_session(session_id: str, agent_name: str, title: str) -> Dict[str, Any]:
    """새 대화 세션을 생성합니다."""
    init_db()
    created_at = time.time()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_sessions (session_id, agent_name, title, created_at) VALUES (?, ?, ?, ?)",
            (session_id, agent_name, title, created_at)
        )
        conn.commit()
    return {"session_id": session_id, "agent_name": agent_name, "title": title, "created_at": created_at}


def get_sessions(agent_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """대화 세션 목록을 조회합니다 (최신순 정렬)."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if agent_name:
            cursor.execute("SELECT * FROM chat_sessions WHERE agent_name = ? ORDER BY created_at DESC", (agent_name,))
        else:
            cursor.execute("SELECT * FROM chat_sessions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def delete_session(session_id: str):
    """세션과 관련 메시지를 삭제합니다."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        conn.commit()


def add_message(session_id: str, role: str, content: str):
    """세션에 메시지를 추가합니다."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time())
        )
        conn.commit()


def get_messages(session_id: str) -> List[Dict[str, Any]]:
    """세션의 메시지 목록을 조회합니다 (시간순 정렬)."""
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT role, content, created_at FROM chat_messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
