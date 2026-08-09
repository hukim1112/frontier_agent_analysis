import sqlite3
import os
import time
from typing import List, Dict, Any, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database", "chat_history.db")

def init_db():
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
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # SQLite foreign keys are off by default, we delete both manually or rely on cascading if enabled
        cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
        conn.commit()

def add_message(session_id: str, role: str, content: str):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, role, content, time.time())
        )
        conn.commit()

def get_messages(session_id: str) -> List[Dict[str, Any]]:
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT role, content, created_at FROM chat_messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
