"""
app.utils.database — 데이터베이스 관련 유틸리티 패키지

하위 모듈:
- session_store: Streamlit/FastAPI용 세션·메시지 CRUD (chat_history.db)
- chainlit_data_layer: Chainlit Data Layer (chainlit_history.db)
- init_schema: Chainlit DB 스키마 초기화 스크립트
"""

# ── session_store (FastAPI server.py에서 사용) ────────────────
from .session_store import (
    create_session,
    get_sessions,
    delete_session,
    add_message,
    get_messages,
    DB_PATH as SESSION_DB_PATH,
)

# ── chainlit_data_layer (chainlit_ui.py에서 사용) ─────────────
from .chainlit_data_layer import (
    ChainlitSQLiteDataLayer,
    CHAINLIT_DB_PATH,
)

# ── init_schema (수동 실행용) ─────────────────────────────────
from .init_schema import init_chainlit_db

__all__ = [
    # session_store
    "create_session", "get_sessions", "delete_session",
    "add_message", "get_messages", "SESSION_DB_PATH",
    # chainlit_data_layer
    "ChainlitSQLiteDataLayer", "CHAINLIT_DB_PATH",
    # init_schema
    "init_chainlit_db",
]
