"""
init_schema.py — Chainlit SQLite DB 스키마 초기화 스크립트

독립 실행 가능: python -m app.utils.database.init_schema
chainlit_data_layer.py의 _init_tables()와 동일한 스키마를 생성합니다.
배포/마이그레이션 시 수동으로 실행할 수 있습니다.
"""

import sqlite3
import os

DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "database"
)
DB_PATH = os.path.join(DB_DIR, "chainlit_history.db")


def init_chainlit_db():
    """Chainlit 히스토리 DB의 전체 스키마를 초기화합니다."""
    os.makedirs(DB_DIR, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()

        # 1. users — 로그인 사용자
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                identifier TEXT NOT NULL UNIQUE,
                createdAt TEXT,
                metadata TEXT
            );
        """)

        # 2. threads — 대화방
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                createdAt TEXT,
                name TEXT,
                userId TEXT,
                userIdentifier TEXT,
                tags TEXT,
                metadata TEXT
            );
        """)

        # 3. steps — 메시지 및 도구 호출 단계
        cursor.execute("""
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
            );
        """)

        # 4. elements — 이미지/파일 등 첨부 요소
        cursor.execute("""
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
            );
        """)

        # 5. feedbacks — 사용자 피드백
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedbacks (
                id TEXT PRIMARY KEY,
                forId TEXT NOT NULL,
                threadId TEXT,
                value INTEGER,
                comment TEXT
            );
        """)
        conn.commit()
    print(f"✅ Chainlit SQLite DB initialized at: {DB_PATH}")


if __name__ == "__main__":
    init_chainlit_db()
