"""
Episodic Memory Store (L2) — episodic.db

세션 대화를 SQLite에 저장하고, FTS5로 검색합니다.
- sessions 테이블: 세션 요약 (영어) + 키워드
- messages 테이블: 세션 내 메시지 원문 (원어 유지)
- sessions_fts: 세션 요약/키워드 전문 검색
- messages_fts: 메시지 내용 전문 검색 (Anchor 인출용)

세션 종료 시 finalize_session()을 호출하면:
1. 메시지를 messages 테이블에 저장
2. LLM으로 영어 요약 생성
3. sessions 테이블에 요약 + 키워드 저장
→ 다음 세션에서 FTS5로 검색 가능
"""

import os
import json
import time
import logging
import aiosqlite
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from app.utils.message_utils import normalize_content

logger = logging.getLogger(__name__)

# ── SQL Schema ──

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    summary TEXT,
    keywords TEXT,
    message_count INTEGER DEFAULT 0,
    created_at REAL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT,
    tool_name TEXT,
    created_at REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
"""

FTS_SCHEMA_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    summary, keywords
);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content
);
"""

# ── Summary Generation Prompt ──

SUMMARY_PROMPT = """Analyze the following conversation and produce a concise English summary.
Also extract 3-8 English keywords that would help find this conversation later.

Conversation:
{conversation}

Respond in this exact JSON format:
{{"summary": "A 2-3 sentence English summary of the conversation.", "keywords": ["keyword1", "keyword2", "keyword3"]}}
"""


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


class EpisodicStore:
    """L2 에피소드 메모리 — 세션 대화를 SQLite에 저장하고 검색합니다.

    DB: episodic.db
    """

    def __init__(self, db_path: str = "./app/database/episodic.db"):
        self.db_path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    async def setup(self) -> None:
        """DB 연결 + 테이블 생성."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = await aiosqlite.connect(self.db_path, check_same_thread=False)
        await self._conn.executescript(SCHEMA_SQL)
        # FTS5 테이블은 별도 처리 (이미 존재하면 스킵)
        try:
            await self._conn.executescript(FTS_SCHEMA_SQL)
        except Exception as e:
            logger.warning("FTS5 setup warning (may already exist): %s", e)
        await self._conn.commit()
        logger.info("EpisodicStore initialized: %s", self.db_path)

    async def close(self) -> None:
        """DB 연결 종료."""
        if self._conn:
            await self._conn.close()
            self._conn = None

    # ── 메시지 저장 ──

    async def save_messages(
        self, session_id: str, messages: List[Any]
    ) -> int:
        """현재 세션의 메시지를 DB에 저장.

        기존 메시지가 있으면 삭제 후 재삽입 (upsert 대체).
        Returns: 저장된 메시지 수
        """
        if not self._conn:
            raise RuntimeError("EpisodicStore not initialized. Call setup() first.")

        now = time.time()
        serialized = [_serialize_message(m) for m in messages]

        # 기존 메시지 삭제 (해당 세션)
        await self._conn.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )

        # 메시지 삽입
        for msg in serialized:
            await self._conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (session_id, msg["role"], msg["content"], now),
            )

        # sessions 테이블에 세션 레코드 upsert
        await self._conn.execute(
            """INSERT INTO sessions (session_id, message_count, created_at, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   message_count = excluded.message_count,
                   updated_at = excluded.updated_at""",
            (session_id, len(serialized), now, now),
        )
        await self._conn.commit()
        logger.info("Saved %d messages for session %s", len(serialized), session_id)
        return len(serialized)

    # ── 세션 종료 (요약 생성) ──

    async def finalize_session(
        self,
        session_id: str,
        messages: List[Any],
        llm=None,
    ) -> str:
        """세션 종료: 메시지 저장 + LLM으로 영어 요약 생성 + sessions 테이블 업데이트.

        교육 시연용으로 명시적 호출 가능.

        Args:
            session_id: 세션 식별자
            messages: 대화 메시지 리스트
            llm: 요약 생성에 사용할 LLM (None이면 간단 요약)

        Returns:
            생성된 요약 문자열
        """
        if not self._conn:
            raise RuntimeError("EpisodicStore not initialized. Call setup() first.")

        # 1. 메시지 저장
        await self.save_messages(session_id, messages)

        # 2. 요약 생성
        serialized = [_serialize_message(m) for m in messages]
        summary, keywords = await self._generate_summary(serialized, llm)

        # 3. sessions 테이블 업데이트
        now = time.time()
        await self._conn.execute(
            """UPDATE sessions SET summary = ?, keywords = ?, updated_at = ?
               WHERE session_id = ?""",
            (summary, json.dumps(keywords, ensure_ascii=False), now, session_id),
        )

        # 4. FTS 인덱스 갱신
        await self._update_fts(session_id, summary, keywords, serialized)
        await self._conn.commit()

        logger.info(
            "Session %s finalized: %d messages, summary=%d chars, keywords=%s",
            session_id, len(serialized), len(summary), keywords,
        )
        return summary

    async def _generate_summary(
        self, messages: List[Dict[str, str]], llm=None
    ) -> tuple:
        """LLM으로 영어 요약 + 키워드 추출.

        LLM이 없으면 간단한 규칙 기반 요약 생성.
        """
        if llm is None:
            return self._fallback_summary(messages)

        # 대화를 텍스트로 변환 (최대 2000자)
        conversation_text = self._format_conversation(messages, max_chars=2000)
        prompt = SUMMARY_PROMPT.format(conversation=conversation_text)

        try:
            response = await llm.ainvoke(prompt)
            raw_content = response.content if hasattr(response, "content") else str(response)
            content = normalize_content(raw_content)

            # 마크다운 코드블록 제거
            cleaned = content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                cleaned = "\n".join(lines).strip()

            result = json.loads(cleaned)
            summary = result.get("summary", "No summary available.")
            keywords = result.get("keywords", [])
            return summary, keywords
        except Exception as e:
            logger.warning("LLM summary generation failed: %s. Using fallback.", e)
            return self._fallback_summary(messages)

    def _fallback_summary(self, messages: List[Dict[str, str]]) -> tuple:
        """LLM 없이 간단한 규칙 기반 요약 생성."""
        user_msgs = [m["content"] for m in messages if m["role"] in ("user", "human") and m["content"]]
        if not user_msgs:
            return "Empty conversation.", []

        # 첫 유저 메시지를 요약으로 사용
        first_msg = user_msgs[0][:200]
        summary = f"Conversation starting with: {first_msg}"

        # 단순 키워드 추출 (영어 단어 + 긴 단어)
        all_text = " ".join(user_msgs)
        words = set(all_text.lower().split())
        keywords = [w for w in words if len(w) > 4 and w.isalpha()][:8]

        return summary, keywords

    @staticmethod
    def _format_conversation(messages: List[Dict[str, str]], max_chars: int = 2000) -> str:
        """대화를 텍스트로 포맷."""
        lines = []
        total = 0
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not content or role == "tool":
                continue
            line = f"{role}: {content}"
            if total + len(line) > max_chars:
                lines.append("... [truncated]")
                break
            lines.append(line)
            total += len(line)
        return "\n".join(lines)

    async def _update_fts(
        self,
        session_id: str,
        summary: str,
        keywords: List[str],
        messages: List[Dict[str, str]],
    ) -> None:
        """FTS5 인덱스 갱신."""
        try:
            # sessions_fts: rowid 기반이므로 sessions 테이블의 rowid 사용
            row = await self._conn.execute_fetchall(
                "SELECT rowid FROM sessions WHERE session_id = ?", (session_id,)
            )
            if row:
                rowid = row[0][0]
                # 기존 FTS 엔트리 삭제 시도
                try:
                    await self._conn.execute(
                        "INSERT INTO sessions_fts(sessions_fts, rowid, summary, keywords) VALUES('delete', ?, ?, ?)",
                        (rowid, summary, json.dumps(keywords)),
                    )
                except Exception:
                    pass
                # 새 FTS 엔트리 삽입
                await self._conn.execute(
                    "INSERT INTO sessions_fts(rowid, summary, keywords) VALUES (?, ?, ?)",
                    (rowid, summary, json.dumps(keywords)),
                )

            # messages_fts: 메시지 내용 인덱싱
            msg_rows = await self._conn.execute_fetchall(
                "SELECT id, content FROM messages WHERE session_id = ? AND role != 'tool'",
                (session_id,),
            )
            for msg_id, content in msg_rows:
                if content:
                    try:
                        await self._conn.execute(
                            "INSERT INTO messages_fts(rowid, content) VALUES (?, ?)",
                            (msg_id, content),
                        )
                    except Exception:
                        pass
        except Exception as e:
            logger.warning("FTS update failed: %s", e)

    # ── 검색 ──

    async def search_sessions(
        self,
        query: str,
        top_k: int = 3,
        exclude_session_id: str = None,
    ) -> List[Dict[str, Any]]:
        """FTS5 기반 세션 검색.

        Args:
            query: 검색 쿼리 (영어 추천)
            top_k: 반환할 최대 세션 수
            exclude_session_id: 현재 세션 제외

        Returns:
            [{session_id, summary, keywords, message_count, snippet}, ...]
        """
        if not self._conn or not query.strip():
            return []

        try:
            import re
            # FTS5 구문 에러(-, &, :, *, / 등) 완전 방지 정제 로직
            cleaned = re.sub(r'[^\w\s\uac00-\ud7a3]', ' ', query)
            words = [w.strip() for w in cleaned.split() if w.strip()]
            fts_query = " OR ".join(words) if words else query

            sql = """
                SELECT s.session_id, s.summary, s.keywords, s.message_count,
                       snippet(sessions_fts, 0, '>>>', '<<<', '...', 32) as snippet
                FROM sessions_fts
                JOIN sessions s ON sessions_fts.rowid = s.rowid
                WHERE sessions_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            """
            rows = await self._conn.execute_fetchall(sql, (fts_query, top_k + 5))

            results = []
            for row in rows:
                sid = row[0]
                if exclude_session_id and sid == exclude_session_id:
                    continue
                results.append({
                    "session_id": sid,
                    "summary": row[1] or "",
                    "keywords": json.loads(row[2]) if row[2] else [],
                    "message_count": row[3] or 0,
                    "snippet": row[4] or "",
                })
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
    ) -> List[Dict[str, Any]]:
        """Anchor 기반 인출: anchor_keyword와 매칭되는 메시지 중심 ±window 반환.

        Hermes의 Lineage 기반 인출 방식 단순화:
        1. anchor_keyword로 FTS5 매칭 (또는 LIKE fallback)
        2. anchor 메시지 중심 ±window 범위의 메시지 반환
        3. 세션의 첫 3개 + 마지막 3개 메시지를 bookend로 항상 포함

        anchor_keyword가 비어있으면 세션의 마지막 window 메시지를 반환.
        """
        if not self._conn:
            return []

        try:
            # 세션의 모든 메시지 로드
            rows = await self._conn.execute_fetchall(
                "SELECT id, role, content, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
            if not rows:
                return []

            all_messages = [
                {"id": r[0], "role": r[1], "content": r[2] or "", "created_at": r[3]}
                for r in rows
            ]

            # Anchor 없으면 마지막 window개
            if not anchor_keyword.strip():
                return all_messages[-window:]

            # Anchor 탐색: 키워드 매칭
            anchor_idx = self._find_anchor(all_messages, anchor_keyword)

            # Anchor 주변 ±window 슬라이싱
            start = max(0, anchor_idx - window)
            end = min(len(all_messages), anchor_idx + window + 1)
            core_view = all_messages[start:end]

            # Bookends: 세션 첫 3개 + 마지막 3개
            bookend_start = all_messages[:3]
            bookend_end = all_messages[-3:]

            # 중복 제거하며 병합
            seen_ids = set()
            result = []
            for msg in bookend_start + core_view + bookend_end:
                if msg["id"] not in seen_ids:
                    seen_ids.add(msg["id"])
                    result.append(msg)

            # id 순으로 정렬
            result.sort(key=lambda m: m["id"])
            return result

        except Exception as e:
            logger.warning("Anchored view failed for session %s: %s", session_id, e)
            return []

    @staticmethod
    def _find_anchor(messages: List[Dict], keyword: str) -> int:
        """키워드와 가장 잘 매칭되는 메시지의 인덱스 반환."""
        keyword_lower = keyword.lower()
        best_idx = len(messages) - 1
        best_score = 0

        for i, msg in enumerate(messages):
            content = (msg.get("content") or "").lower()
            if not content:
                continue
            # 간단한 키워드 매칭 스코어
            words = keyword_lower.split()
            score = sum(1 for w in words if w in content)
            if score > best_score:
                best_score = score
                best_idx = i

        return best_idx

    async def browse_recent(self, limit: int = 5) -> List[Dict[str, Any]]:
        """최근 세션 목록 반환 (Browse 모드)."""
        if not self._conn:
            return []

        try:
            rows = await self._conn.execute_fetchall(
                "SELECT session_id, summary, keywords, message_count, updated_at "
                "FROM sessions WHERE summary IS NOT NULL "
                "ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            return [
                {
                    "session_id": r[0],
                    "summary": r[1] or "",
                    "keywords": json.loads(r[2]) if r[2] else [],
                    "message_count": r[3] or 0,
                    "updated_at": r[4],
                }
                for r in rows
            ]
        except Exception as e:
            logger.warning("Browse recent failed: %s", e)
            return []
