"""
===============================================================================
scripts/migrate_episodic_to_transcript.py — Legacy episodic.db → transcript.db 이관
===============================================================================
구버전 app/database/episodic.db의 messages 테이블(세션별 마지막 스냅샷)을
app/database/transcript.db로 멱등(idempotent) 이관합니다.

- episodic.db는 읽기 전용으로 엽니다 (내용을 바꾸지 않음).
  sessions 테이블의 topics_json / tool_names 컬럼은 앱 시작 시 EpisodicStore.setup()이 자동으로 추가합니다.
- 실행 전에 두 DB의 백업(<파일명>.bak.<YYYYmmdd-HHMMSS>)을 만듭니다 (transcript.db는 이미 있을 때만).
- 이미 이관한 메시지는 건너뛰므로 여러 번 실행해도 중복되지 않습니다.
- 이관이 끝나도 episodic.db의 messages / messages_fts 테이블은 그대로 남겨 둡니다
  (session_recall은 transcript.db에 없는 세션만 이 테이블에서 읽습니다).

Usage:
    python scripts/migrate_episodic_to_transcript.py
    python scripts/migrate_episodic_to_transcript.py --episodic ./app/database/episodic.db --transcript ./app/database/transcript.db
===============================================================================
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.middleware.memory.transcript_store import TranscriptStore


def backup_sqlite(path: Path, stamp: Optional[str] = None) -> Optional[Path]:
    """SQLite 백업 API로 일관된 사본을 만듭니다 (WAL에 남은 변경분 포함). 파일이 없으면 None."""
    if not path.exists():
        return None
    stamp = stamp or time.strftime("%Y%m%d-%H%M%S")
    dst = path.with_name(f"{path.name}.bak.{stamp}")
    src_conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    dst_conn = sqlite3.connect(str(dst))
    try:
        src_conn.backup(dst_conn)
    finally:
        dst_conn.close()
        src_conn.close()
    return dst


def migrate(
    episodic_db_path: str = "./app/database/episodic.db",
    transcript_db_path: str = "./app/database/transcript.db",
    backup: bool = True,
) -> dict:
    """구버전 episodic.db의 messages 테이블 데이터를 transcript.db로 이관합니다."""
    epi_path = Path(episodic_db_path)
    tr_path = Path(transcript_db_path)

    if not epi_path.exists():
        return {"migrated_sessions": 0, "migrated_messages": 0, "status": "no_episodic_db", "backups": []}

    conn = sqlite3.connect(f"{epi_path.resolve().as_uri()}?mode=ro", uri=True, timeout=5.0)
    try:
        has_messages = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='messages'"
        ).fetchone()
        if not has_messages:
            return {"migrated_sessions": 0, "migrated_messages": 0, "status": "already_clean", "backups": []}

        # 구버전 스키마: messages(id, session_id, role, content, tool_name, created_at)
        rows = conn.execute(
            """
            SELECT m.id, m.session_id, COALESCE(s.user_id, 'default') AS user_id,
                   m.role, m.content, m.tool_name
            FROM messages m
            LEFT JOIN sessions s ON s.session_id = m.session_id
            ORDER BY m.session_id, m.id
            """
        ).fetchall()
    finally:
        conn.close()

    backups = []
    if backup and rows:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for path in (epi_path, tr_path):
            dst = backup_sqlite(path, stamp)
            if dst is not None:
                backups.append(str(dst))

    sessions_map: dict[tuple[str, str], list[dict]] = {}
    for msg_id, session_id, user_id, role, content, tool_name in rows:
        sessions_map.setdefault((session_id, user_id or "default"), []).append({
            "id": f"legacy:{session_id}:{msg_id}",
            "role": role,
            "content": content or "",
            "tool_name": tool_name,
        })

    tr_store = TranscriptStore(db_path=str(tr_path))
    try:
        migrated_msgs = 0
        for (session_id, user_id), msgs in sessions_map.items():
            migrated_msgs += tr_store.append(
                session_id=session_id,
                user_id=user_id,
                messages=msgs,
                turn_id=1,
            )
    finally:
        tr_store.close()

    return {
        "migrated_sessions": len(sessions_map),
        "migrated_messages": migrated_msgs,
        "status": "migrated",
        "backups": backups,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate legacy episodic.db messages into transcript.db")
    parser.add_argument("--episodic", default="./app/database/episodic.db")
    parser.add_argument("--transcript", default="./app/database/transcript.db")
    parser.add_argument("--no-backup", action="store_true", help="백업 파일을 만들지 않음")
    args = parser.parse_args()
    result = migrate(args.episodic, args.transcript, backup=not args.no_backup)
    print(f"[migrate_episodic_to_transcript] {result}")
