"""
Memory Middleware — AgentMiddleware (before_agent + after_agent + daemon thread)

Hermes 스타일 메모리 미들웨어:
- before_agent: L2(에피소드) 자동 인출 + L3(시멘틱) 프리페치 → recalled_memory 주입
- after_agent: 비동기 데몬 스레드로 Background Review 실행
  - Semantic Memory (MEMORY.md/USER.md) 갱신
  - Episodic Memory (세션 요약) 저장

Background Review는 Hermes의 _spawn_background_review() 패턴:
- 메인 스레드를 블로킹하지 않음 (daemon=True)
- 별도 LLM 호출로 대화 리뷰
- memory(add/replace) 실행으로 MEMORY.md/USER.md 갱신
- _persist_disabled=True 패턴으로 메인 세션 오염 방지
"""

import json
import time
import logging
import asyncio
import threading
from typing import Any, Dict, List, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage
from app.utils.message_utils import normalize_content

logger = logging.getLogger(__name__)

# ── Background Review Prompts ──

COMBINED_REVIEW_PROMPT = """Review the conversation above and update two things:

**Memory** — who the user is.
Did the user reveal personal information, preferences, communication style, or workflow habits?
If so, save them using: memory(action="add", target="user", content="...")
Write entries in concise English. Keep each entry to 1-2 sentences.

**Agent Notes** — what you learned.
Did you discover new environment facts, project conventions, tool quirks, or useful patterns?
If so, save them using: memory(action="add", target="memory", content="...")
Write entries in concise English. Keep each entry to 1-2 sentences.

IMPORTANT:
- Only save genuinely NEW or UPDATED information. Check existing entries first.
- Do NOT save task-specific data, logs, conversation summaries, or temporary state.
- If nothing is worth saving, respond with "Nothing to save."
- Merge overlapping entries using memory(action="replace", target=..., old_text=..., content=...).

Current MEMORY.md entries:
{memory_entries}

Current USER.md entries:
{user_entries}
"""


def _serialize_message_for_review(msg) -> Dict[str, str]:
    """메시지를 리뷰용 텍스트로 직렬화."""
    if isinstance(msg, BaseMessage):
        return {"role": msg.type, "content": msg.content or ""}
    if isinstance(msg, dict):
        return {
            "role": msg.get("role", msg.get("type", "unknown")),
            "content": msg.get("content", ""),
        }
    return {"role": "unknown", "content": str(msg)}


class MemoryMiddleware(AgentMiddleware):
    """Hermes 스타일 메모리 미들웨어.

    Hooks:
    - before_agent: L2+L3 메모리 인출 → AgentContext.recalled_memory에 주입
    - after_agent: 비동기 데몬 스레드로 Background Review 실행

    Usage:
        memory_mw = MemoryMiddleware(
            semantic_store=semantic_store,
            episodic_store=episodic_store,
            review_llm=llm,
        )
        agent = create_agent(..., middleware=[memory_mw])
    """

    def __init__(
        self,
        semantic_store,
        episodic_store,
        review_llm=None,
    ):
        self.semantic_store = semantic_store
        self.episodic_store = episodic_store
        self.review_llm = review_llm
        self._last_loaded_session_id: Optional[str] = None

    def before_agent(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        """L2+L3 메모리 인출 → AgentContext.recalled_memory에 주입."""
        ctx = getattr(runtime, "context", None)
        if not ctx:
            return None

        recalled_parts = []

        # L3: Semantic Memory 프리페치 (세션 감지 기반 Frozen Snapshot)
        if getattr(ctx, "semantic_memory_enabled", False):
            session_id = self._get_session_id(runtime)
            # 🌟 세션이 전환된 순간에만 디스크에서 최신 마크다운 로드 (세션 내 KV-Cache 보존)
            if self._last_loaded_session_id != session_id:
                self.semantic_store.load_from_disk()
                self._last_loaded_session_id = session_id
                logger.info("[MemoryMiddleware] New session detected (%s): reloaded semantic memory snapshot.", session_id)

            memory_block = self.semantic_store.format_for_prompt("memory")
            user_block = self.semantic_store.format_for_prompt("user")
            if memory_block:
                recalled_parts.append(memory_block)
            if user_block:
                recalled_parts.append(user_block)

        # L2: Episodic Memory 자동 인출 (유저 쿼리 기반)
        if getattr(ctx, "episodic_memory_enabled", False):
            user_query = self._extract_last_user_query(state)
            if user_query:
                try:
                    sessions = self._run_async_safe(
                        self.episodic_store.search_sessions(
                            user_query,
                            top_k=2,
                            exclude_session_id=self._get_session_id(runtime),
                        )
                    )
                    if sessions:
                        recalled_parts.append(self._format_episodic(sessions))
                except Exception as e:
                    logger.warning("Episodic memory retrieval failed: %s", e)

        if recalled_parts:
            ctx.recalled_memory = "\n\n".join(recalled_parts)

        return None

    def after_agent(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        """비동기 데몬 스레드로 Background Review 실행.

        Hermes 패턴: 메인 응답에 영향 없이 백그라운드에서 메모리 갱신.
        """
        ctx = getattr(runtime, "context", None)
        if not ctx:
            return None

        messages = state.get("messages", [])
        if not messages or len(messages) < 2:
            return None

        learning_enabled = getattr(ctx, "memory_learning_enabled", False)
        if not learning_enabled:
            return None

        # 대화 스냅샷 복제
        messages_snapshot = [
            _serialize_message_for_review(m) for m in messages
        ]
        session_id = self._get_session_id(runtime)
        memory_dir = getattr(ctx, "memory_dir", "./artifacts/memory")

        # 비동기 데몬 스레드 생성 (Hermes: _spawn_background_review)
        thread = threading.Thread(
            target=self._background_review,
            args=(messages_snapshot, session_id, memory_dir),
            daemon=True,
            name="hermes-bg-review",
        )
        thread.start()
        logger.info("Background review spawned for session %s", session_id)

        return None

    # ── Background Review (daemon thread) ──

    def _background_review(
        self,
        messages_snapshot: List[Dict[str, str]],
        session_id: str,
        memory_dir: str,
    ) -> None:
        """Background Review 스레드.

        1. LLM에게 대화 리뷰 요청
        2. LLM 응답에서 memory() 호출 파싱
        3. SemanticMemoryStore에 반영
        4. EpisodicStore에 세션 요약 저장
        """
        try:
            logger.info("[BG-Review] Starting background review for session %s", session_id)

            # 1. Semantic Memory 갱신 (LLM 리뷰)
            if self.review_llm:
                self._review_semantic_memory(messages_snapshot)

            # 2. Episodic Memory 세션 요약 저장
            self._review_episodic_memory(messages_snapshot, session_id)

            logger.info("[BG-Review] Completed for session %s", session_id)

        except Exception as e:
            logger.error("[BG-Review] Failed for session %s: %s", session_id, e)

    def _review_semantic_memory(self, messages: List[Dict[str, str]]) -> None:
        """LLM으로 대화를 리뷰하여 MEMORY.md/USER.md 갱신."""
        if not self.review_llm:
            return

        # 현재 엔트리를 프롬프트에 포함
        memory_entries = "\n".join(
            f"- {e}" for e in self.semantic_store.memory_entries
        ) or "(empty)"
        user_entries = "\n".join(
            f"- {e}" for e in self.semantic_store.user_entries
        ) or "(empty)"

        # 대화 텍스트 구성
        conversation_text = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in messages
            if m.get("content") and m.get("role") != "tool"
        )

        review_prompt = (
            f"Conversation:\n{conversation_text[:3000]}\n\n"
            + COMBINED_REVIEW_PROMPT.format(
                memory_entries=memory_entries,
                user_entries=user_entries,
            )
        )

        try:
            response = self.review_llm.invoke(review_prompt)
            content = normalize_content(getattr(response, "content", response))

            # LLM 응답에서 memory() 호출 파싱 및 실행
            self._parse_and_execute_memory_ops(content)

        except Exception as e:
            logger.warning("[BG-Review] Semantic review LLM call failed: %s", e)

    def _parse_and_execute_memory_ops(self, llm_response: str) -> None:
        """LLM 응답에서 memory() 또는 <call:memory ... /> 파싱하여 실행."""
        import re

        patterns = [
            r'''memory\(((?:[^)"']|"[^"]*"|'[^']*')*)\)''',
            r'<call:memory\s+([^/>]+)/?>',
            r'<memory\s+([^/>]+)/?>',
        ]

        matches = []
        for p in patterns:
            matches.extend(re.findall(p, llm_response))

        for match in matches:
            try:
                # 인자 파싱
                action = self._extract_kwarg(match, "action")
                target = self._extract_kwarg(match, "target") or "memory"
                content = self._extract_kwarg(match, "content") or ""
                old_text = self._extract_kwarg(match, "old_text") or ""

                if not action:
                    continue

                if action == "add" and content:
                    result = self.semantic_store.add(target, content)
                elif action == "replace" and old_text and content:
                    result = self.semantic_store.replace(target, old_text, content)
                elif action == "remove" and old_text:
                    result = self.semantic_store.remove(target, old_text)
                else:
                    continue

                if result.get("success"):
                    logger.info("[BG-Review] Memory %s: %s → %s",
                                action, target, content[:50] if content else old_text[:50])
                else:
                    logger.warning("[BG-Review] Memory %s failed: %s",
                                   action, result.get("error", "unknown"))

            except Exception as e:
                logger.warning("[BG-Review] Failed to parse memory op '%s': %s", match, e)

    @staticmethod
    def _extract_kwarg(args_str: str, key: str) -> Optional[str]:
        """키워드 인자 문자열에서 특정 키의 값을 추출."""
        import re
        # key="value" 또는 key='value' 패턴
        pattern = rf'{key}\s*=\s*["\']([^"\']*)["\']'
        m = re.search(pattern, args_str)
        return m.group(1) if m else None

    def _review_episodic_memory(
        self, messages: List[Dict[str, str]], session_id: str
    ) -> None:
        """세션 메시지를 EpisodicStore에 저장 + 요약 생성."""
        try:
            asyncio.run(
                self.episodic_store.finalize_session(
                    session_id=session_id,
                    messages=messages,
                    llm=self.review_llm,
                )
            )
        except Exception as e:
            logger.warning("[BG-Review] Episodic finalize failed: %s", e)

    # ── Helpers ──

    @staticmethod
    def _extract_last_user_query(state: Dict[str, Any]) -> str:
        """state에서 마지막 유저 메시지 추출."""
        messages = state.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, BaseMessage) and msg.type in ("human", "user"):
                return msg.content or ""
            if isinstance(msg, dict) and msg.get("role") in ("human", "user"):
                return msg.get("content", "")
        return ""

    @staticmethod
    def _get_session_id(runtime: Any) -> str:
        """runtime에서 session_id(thread_id) 추출."""
        # 1. context에서 탐색
        ctx = getattr(runtime, "context", None)
        if ctx:
            sid = getattr(ctx, "session_id", None)
            if sid:
                return sid

        # 2. config에서 탐색
        config = getattr(runtime, "config", None)
        if config:
            tid = config.get("configurable", {}).get("thread_id")
            if tid:
                return tid

        return f"session-{int(time.time())}"

    @staticmethod
    def _format_episodic(sessions: List[Dict[str, Any]]) -> str:
        """에피소드 검색 결과를 프롬프트 주입용 텍스트로 포맷."""
        if not sessions:
            return ""

        separator = "═" * 46
        header = "EPISODIC MEMORY (relevant past sessions)"
        lines = [separator, header, separator]

        for i, s in enumerate(sessions, 1):
            lines.append(f"[Session {i}] {s.get('session_id', 'unknown')}")
            lines.append(f"  Summary: {s.get('summary', 'N/A')}")
            keywords = s.get("keywords", [])
            if keywords:
                lines.append(f"  Keywords: {', '.join(keywords)}")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _run_async_safe(coro):
        """비동기 코루틴을 안전하게 동기 실행."""
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            return asyncio.run(coro)
