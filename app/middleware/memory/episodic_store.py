"""
===============================================================================
Episodic Memory Store — episodic.db (SQLite + FTS5)
===============================================================================
Source: frontier-agent-lab/modules/hermes/session_store.py

세션 대화를 SQLite에 저장하고, FTS5로 검색합니다.
- sessions 테이블: 세션 요약 + 키워드 (한/영 이중 색인)
- messages 테이블: 세션 내 메시지 원문 (원어 유지)
- sessions_fts: 세션 요약/키워드 전문 검색
- messages_fts: 메시지 내용 전문 검색 (Anchor 인출용)

세션 종료 시 finalize_session()을 호출하면:
1. 메시지를 messages 테이블에 저장
2. 로컬 규칙 기반으로 원문 키워드 추출 (1단계)
3. LLM이 있으면 지능형 요약 + 한/영 이중 키워드 추출 후 로컬 키워드와 병합 (2단계)
4. sessions 테이블에 요약 + 병합된 키워드 저장
→ 다음 세션에서 FTS5로 검색 가능
===============================================================================
"""

import os
import re
import json
import time
import logging
import asyncio
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

SUMMARY_PROMPT = """Analyze the following conversation and extract a summary and search keywords.

Rules for keywords:
1. Use SHORT, ATOMIC terms (1-2 words each). Split compound concepts into separate keywords.
2. Preserve ORIGINAL words from the conversation. Do NOT paraphrase or creatively translate.
3. Include BOTH Korean and English forms of each key concept as separate entries.
4. Extract only NOUNS and technical terms. Exclude verbs, adjectives, and sentence fragments.
5. Aim for 6-12 keywords total.

Example:
Conversation: "Kubernetes 클러스터에서 Pod 오토스케일링 설정을 논의했고, HPA 메트릭으로 CPU 사용률 70%를 기준으로 확정했습니다."
Good keywords: ["Kubernetes", "클러스터", "Pod", "오토스케일링", "autoscaling", "HPA", "CPU", "메트릭"]
Bad keywords: ["쿠버네티스 클러스터 오토스케일링 설정", "pod autoscaling configuration"] ← too long, compound

Conversation:
{conversation}

Respond in this exact JSON format:
{{"summary": "A 2-3 sentence summary of the conversation focusing on decisions and outcomes.", "keywords": ["keyword1", "keyword2", "keyword3"]}}
"""

# ── 한국어/영어 불용어 (키워드 추출 시 필터링) ──

_STOPWORDS_KO = frozenset([
    "그리고", "그래서", "하지만", "그런데", "또한", "그러나", "따라서", "때문에",
    "어떻게", "무엇을", "있습니다", "합니다", "됩니다", "입니다", "했습니다",
    "하겠습니다", "해주세요", "시작합시다", "진행합시다", "확정할까요",
    "네", "예", "아니요", "좋습니다", "감사합니다",
    "이것", "저것", "여기", "거기", "우리", "오늘",
    "어떤", "어디", "위해", "대신", "방식", "방식을", "방식으로",
])

_STOPWORDS_EN = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could",
    "and", "but", "or", "nor", "not", "so", "yet", "for", "with",
    "from", "into", "about", "that", "this", "these", "those",
    "what", "which", "who", "whom", "how", "where", "when", "why",
    "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "than", "too", "very", "just", "also",
    "then", "there", "here", "only", "yes", "no",
    "use", "using", "used", "uses",
])


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

    def __init__(self, db_path: str = "./artifacts/memory/episodic.db"):
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

    @staticmethod
    def _extract_local_keywords(
        messages: List[Dict[str, str]], max_keywords: int = 15
    ) -> List[str]:
        """대화 원문에서 규칙 기반으로 한국어/영어 키워드를 추출합니다.

        Progressive Keyword Pipeline 1단계:
        - 대화 전체 텍스트에서 의미 있는 명사/기술어를 추출
        - 불용어(조사, 어미, 접속사 등) 필터링
        - 한글 2자 이상, 영어 3자 이상 단어만 선별
        - 빈도순으로 상위 max_keywords개 반환

        이 단계의 핵심 역할:
        유저가 실제로 말한 원문 단어("쿠키", "보안", "토큰" 등)를
        FTS5 인덱스에 100% 보존하여 검색 누락을 원천 차단합니다.
        """
        all_text = " ".join(
            m["content"] for m in messages
            if m.get("content") and m.get("role") != "tool"
        )
        if not all_text.strip():
            return []

        # 한글 단어 추출 (2자 이상 연속 한글)
        ko_words = re.findall(r'[가-힣]{2,}', all_text)
        # 영어 단어 추출 (하이픈 포함 기술어, 3자 이상)
        en_words = re.findall(r'[A-Za-z][A-Za-z0-9\-]{2,}', all_text)

        # 불용어 필터링 + 빈도 집계
        freq: Dict[str, int] = {}
        for w in ko_words:
            if w not in _STOPWORDS_KO:
                freq[w] = freq.get(w, 0) + 1
        for w in en_words:
            w_lower = w.lower()
            if w_lower not in _STOPWORDS_EN:
                # 원래 대소문자 유지 (JWT, XSS 등 보존)
                key = w if w.isupper() or w[0].isupper() else w_lower
                freq[key] = freq.get(key, 0) + 1

        # 빈도순 정렬 후 상위 N개 반환
        sorted_words = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
        return [w for w, _ in sorted_words[:max_keywords]]

    async def _generate_summary(
        self, messages: List[Dict[str, str]], llm=None
    ) -> tuple:
        """Progressive Keyword Pipeline으로 요약 + 키워드 추출.

        1단계: _extract_local_keywords()로 원문 키워드 추출 (항상 실행)
        2단계: LLM이 있으면 지능형 요약 + 한/영 이중 키워드 추출 후 1단계와 병합
               LLM이 없으면 1단계 키워드만으로 Fallback 요약 생성
        """
        # 1단계: 로컬 규칙 기반 키워드 추출 (항상 실행)
        local_keywords = self._extract_local_keywords(messages)

        if llm is None:
            return self._fallback_summary(messages, local_keywords)

        # 2단계: LLM 지능형 요약 + 키워드 추출
        conversation_text = self._format_conversation(messages, max_chars=2000)
        prompt = SUMMARY_PROMPT.format(conversation=conversation_text)

        try:
            if hasattr(llm, "invoke"):
                loop = asyncio.get_running_loop()
                response = await loop.run_in_executor(None, llm.invoke, prompt)
            else:
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
            llm_keywords = result.get("keywords", [])

            # 핵심: 로컬 키워드(원문 보존) + LLM 키워드(지능형 확장) 병합
            merged = self._merge_keywords(local_keywords, llm_keywords)
            return summary, merged
        except Exception as e:
            logger.warning("LLM summary generation failed: %s. Using fallback.", e)
            return self._fallback_summary(messages, local_keywords)

    @staticmethod
    def _merge_keywords(
        local_keywords: List[str],
        llm_keywords: List[str],
        max_total: int = 20,
    ) -> List[str]:
        """로컬 키워드와 LLM 키워드를 병합합니다.

        - LLM 키워드를 우선 배치 (의미적으로 정제된 키워드)
        - 로컬 키워드 중 LLM이 놓친 원문 단어를 뒤에 추가
        - 대소문자 무시 중복 제거
        """
        seen_lower = set()
        merged = []

        # LLM 키워드 우선
        for kw in llm_keywords:
            kw_str = str(kw).strip()
            if kw_str and kw_str.lower() not in seen_lower:
                seen_lower.add(kw_str.lower())
                merged.append(kw_str)

        # 로컬 키워드 보충 (LLM이 놓친 원문 단어 보존)
        for kw in local_keywords:
            if kw.lower() not in seen_lower:
                seen_lower.add(kw.lower())
                merged.append(kw)

        return merged[:max_total]

    def _fallback_summary(
        self, messages: List[Dict[str, str]], local_keywords: List[str] = None
    ) -> tuple:
        """LLM 없이 규칙 기반 요약 생성.

        1단계 로컬 키워드가 제공되면 그대로 사용합니다.
        """
        user_msgs = [m["content"] for m in messages if m["role"] in ("user", "human") and m["content"]]
        if not user_msgs:
            return "Empty conversation.", []

        # 첫 유저 메시지를 요약으로 사용
        first_msg = user_msgs[0][:200]
        summary = f"Conversation starting with: {first_msg}"

        # 로컬 키워드가 있으면 사용, 없으면 추출
        if local_keywords is None:
            local_keywords = self._extract_local_keywords(messages)

        return summary, local_keywords

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
                # 기존 FTS 엔트리 삭제 (rowid 기준 무조건 안전 삭제)
                try:
                    await self._conn.execute(
                        "DELETE FROM sessions_fts WHERE rowid = ?", (rowid,)
                    )
                except Exception:
                    pass
                # 새 FTS 엔트리 삽입
                await self._conn.execute(
                    "INSERT INTO sessions_fts(rowid, summary, keywords) VALUES (?, ?, ?)",
                    (rowid, summary, json.dumps(keywords, ensure_ascii=False)),
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
                            "DELETE FROM messages_fts WHERE rowid = ?", (msg_id,)
                        )
                    except Exception:
                        pass
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
    # Extension Point: 향후 Dense Vector(Embedding) 기반 Hybrid Search 추가 시
    # search_sessions()에 strategy="fts" | "dense" | "hybrid" 파라미터를 도입하고,
    # _search_fts()와 _search_dense()를 분리한 뒤 RRF(Reciprocal Rank Fusion)로
    # 결과를 병합하는 구조로 확장할 수 있습니다.

    async def search_sessions(
        self,
        query: str,
        top_k: int = 3,
        exclude_session_id: str = None,
    ) -> List[Dict[str, Any]]:
        """FTS5 기반 세션 검색.

        Args:
            query: 검색 쿼리 (한국어/영어 모두 지원)
            top_k: 반환할 최대 세션 수
            exclude_session_id: 현재 세션 제외

        Returns:
            [{session_id, summary, keywords, message_count, snippet}, ...]
        """
        if not self._conn or not query.strip():
            return []

        try:
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
