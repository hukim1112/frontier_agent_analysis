"""
app.utils.database — 데이터베이스 관련 유틸리티 패키지

하위 모듈:
- session_store: FastAPI용 세션·옵션 메타데이터 및 공유 Checkpointer SSOT (checkpoints.db)
- chainlit_data_layer: Chainlit Data Layer (chainlit.db)
- init_schema: Chainlit DB 스키마 초기화 스크립트
"""

# ── session_store (FastAPI server.py에서 사용) ────────────────
from .session_store import (
    init_db,
    create_session,
    get_sessions,
    delete_session,
    get_messages,
    get_session,
    register_subagent_thread,
    acreate_session,
    aget_sessions,
    adelete_session,
    aget_messages,
    aget_session,
    get_shared_checkpointer,
    close_shared_checkpointer,
    DB_PATH as SESSION_DB_PATH,
    CHECKPOINTS_DB_PATH,
)

# ── chainlit_data_layer (chainlit_ui.py에서 사용) ─────────────
from .chainlit_data_layer import (
    ChainlitSQLiteDataLayer,
    CHAINLIT_DB_PATH,
)

# ── init_schema (수동 실행용) ─────────────────────────────────
from .init_schema import init_chainlit_db

__all__ = [
    # session_store (async + checkpointer SSOT)
    "init_db",
    "create_session", "get_sessions", "delete_session",
    "get_messages", "SESSION_DB_PATH", "CHECKPOINTS_DB_PATH",
    "get_session", "register_subagent_thread",
    "acreate_session", "aget_sessions", "adelete_session",
    "aget_messages", "aget_session",
    "get_shared_checkpointer", "close_shared_checkpointer",
    # chainlit_data_layer
    "ChainlitSQLiteDataLayer", "CHAINLIT_DB_PATH",
    # init_schema
    "init_chainlit_db",
]
