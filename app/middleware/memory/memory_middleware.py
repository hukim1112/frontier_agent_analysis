"""
===============================================================================
Memory Middleware — AgentMiddleware (before_agent + after_agent + get_tools)
===============================================================================
Source: frontier-agent-lab/modules/hermes/memory_middleware.py
        + memory_tool.py + session_search_tool.py (도구 통합)

고응집 메모리 미들웨어 — 스토어, 도구, 훅을 모두 소유합니다.

Hooks:
- before_agent: L3(시맨틱) 프리페치 → recalled_memory 주입 (Frozen Snapshot, Clean Slate 원칙)
- after_agent: 비동기 데몬 스레드로 Background Review 실행

Tool Factory:
- get_tools(): memory() + session_search() + session_recall() 도구 3종 세트 반환
  - memory(): 에이전트가 직접 MEMORY.md / USER.md를 관리 (add/replace/remove)
  - session_search(): 과거 세션 메타데이터 검색 및 최근 세션 브라우즈 (1단계 탐색)
  - session_recall(): Anchor 기반 과거 세션 상세 메시지 인출 (2단계 정밀 인출)

Background Review (Hermes _spawn_background_review 패턴):
- 메인 스레드를 블로킹하지 않음 (daemon=True)
- 별도 LLM 호출로 대화 리뷰 → memory(add/replace) 파싱 실행
- EpisodicStore에 세션 요약 저장
===============================================================================
"""

import json
import time
import logging
import asyncio
import threading
from typing import Any, Dict, List, Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import BaseMessage
from langchain.tools import tool as langchain_tool
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
    """고응집 메모리 미들웨어 — 스토어, 도구, 훅을 모두 소유.

    Usage:
        memory_mw = MemoryMiddleware(
            semantic_store=semantic_store,
            episodic_store=episodic_store,
            review_llm=llm,
        )
        agent = create_agent(
            model=llm,
            tools=base_tools + memory_mw.get_tools(),
            middleware=[memory_mw],
        )
    """

    def __init__(
        self,
        semantic_store,
        episodic_store,
        review_llm=None,
        auto_recall_episodic: bool = False,
    ):
        self.semantic_store = semantic_store
        self.episodic_store = episodic_store
        self.review_llm = review_llm
        self.auto_recall_episodic = auto_recall_episodic
        self._last_loaded_session_id: Optional[str] = None

        # 내부에서 도구 생성 (스토어 바인딩을 미들웨어가 소유)
        self._memory_tool = self._create_memory_tool()
        self._session_search_tool = self._create_session_search_tool()
        self._session_recall_tool = self._create_session_recall_tool()

    # ══════════════════════════════════════════════════════════════════
    # Tool Factory
    # ══════════════════════════════════════════════════════════════════

    def get_tools(self) -> list:
        """에이전트 생성 시 바인딩할 메모리 도구 목록 반환.

        Returns:
            [memory_tool, session_search_tool, session_recall_tool]
        """
        return [self._memory_tool, self._session_search_tool, self._session_recall_tool]

    def _create_memory_tool(self):
        """@tool memory(action, target, content, old_text) 생성."""
        store = self.semantic_store

        @langchain_tool
        def memory(
            action: str,
            target: str = "memory",
            content: str = "",
            old_text: str = "",
        ) -> str:
            """Manage the agent's long-term semantic memory (MEMORY.md / USER.md).

            Two stores:
            - target="memory": Agent's personal notes (environment facts, project conventions, tool quirks).
            - target="user": What the agent knows about the user (preferences, communication style).

            Actions:
            - action="add": Append a new entry. Content must be concise English.
            - action="replace": Find entry containing old_text, replace with content.
            - action="remove": Find entry containing old_text, delete it.

            Guidelines:
            - Write entries in English for search efficiency.
            - Keep entries short (1-2 sentences). Merge overlapping facts.
            - Do NOT store task-specific data, logs, or temporary state.

            Args:
                action: "add" | "replace" | "remove"
                target: "memory" (MEMORY.md) | "user" (USER.md)
                content: New entry text (required for add/replace)
                old_text: Unique substring to match existing entry (required for replace/remove)

            Returns:
                JSON string with success status, usage info, and current entry count.
            """
            if action == "add":
                result = store.add(target, content)
            elif action == "replace":
                result = store.replace(target, old_text, content)
            elif action == "remove":
                result = store.remove(target, old_text)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown action '{action}'. Use add, replace, or remove.",
                }
            return json.dumps(result, ensure_ascii=False, indent=2)

        return memory

    def _create_session_search_tool(self):
        """@tool session_search(query, limit) 생성 — 1단계 세션 탐색 및 브라우즈."""
        episodic = self.episodic_store

        @langchain_tool
        def session_search(
            query: str = "",
            limit: int = 3,
        ) -> str:
            """Search past conversation sessions to recall relevant context.

            Two modes (inferred from arguments):
            1. DISCOVERY: Pass a query string to search for related past sessions via FTS5.
               Returns session summaries, keywords, and match snippets.
            2. BROWSE: Call with no arguments (or empty query) to list recent sessions chronologically.

            Use this tool when:
            - The user references a past conversation ("we discussed X before")
            - You need context from a previous session to answer the current question
            - The user asks about their past activities or history

            After finding a relevant session_id, use session_recall to drill down
            into the actual messages.

            Args:
                query: Search query in English (empty = browse recent sessions)
                limit: Max sessions to return (default 3)

            Returns:
                JSON with session list: [{session_id, summary, keywords, snippet}, ...]
            """
            if query.strip():
                results = _run_async_safe(
                    episodic.search_sessions(query, top_k=limit)
                )
            else:
                results = _run_async_safe(
                    episodic.browse_recent(limit=limit)
                )

            if not results:
                return json.dumps({
                    "results": [],
                    "message": "No matching sessions found." if query else "No past sessions recorded.",
                })

            return json.dumps({
                "query": query or "(browse recent)",
                "results": results,
                "count": len(results),
            }, ensure_ascii=False, indent=2)

        return session_search

    def _create_session_recall_tool(self):
        """@tool session_recall(session_id, anchor_message, window) 생성 — 2단계 상세 대화 인출."""
        episodic = self.episodic_store

        @langchain_tool
        def session_recall(
            session_id: str,
            anchor_message: str = "",
            window: int = 5,
        ) -> str:
            """Recall messages from a specific past session using anchor-based retrieval.

            Anchor-based retrieval (Hermes Lineage pattern):
            1. Finds the message best matching anchor_message keywords
            2. Returns ±window messages around the anchor
            3. Always includes bookends (first 3 + last 3 messages of the session)

            If anchor_message is empty, returns the last `window` messages of the session.

            Use this tool after session_search finds a relevant session_id to inspect the
            actual raw messages and context.

            Args:
                session_id: The session ID to recall messages from
                anchor_message: Keywords to locate the most relevant message (optional)
                window: Number of messages before/after the anchor (default 5)

            Returns:
                JSON with messages: [{role, content}, ...]
            """
            messages = _run_async_safe(
                episodic.get_anchored_view(
                    session_id=session_id,
                    anchor_keyword=anchor_message,
                    window=window,
                )
            )

            if not messages:
                return json.dumps({
                    "session_id": session_id,
                    "messages": [],
                    "message": f"No messages found for session '{session_id}'.",
                })

            return json.dumps({
                "session_id": session_id,
                "anchor": anchor_message or "(tail view)",
                "message_count": len(messages),
                "messages": [
                    {"role": m["role"], "content": m["content"][:500]}
                    for m in messages
                ],
            }, ensure_ascii=False, indent=2)

        return session_recall

    # ══════════════════════════════════════════════════════════════════
    # AgentMiddleware Hooks
    # ══════════════════════════════════════════════════════════════════

    def before_agent(self, state: Dict[str, Any], runtime: Any) -> Optional[Dict[str, Any]]:
        """L3 시맨틱 메모리 프리페치 (Frozen Snapshot) → AgentContext.recalled_memory에 주입.

        순정 Hermes(A안) 원칙:
        - L3 (Semantic Memory): 세션 시작 시 Frozen Snapshot으로 시스템 프롬프트에 상시 주입.
        - L2 (Episodic Memory): Clean Slate 유지. 프롬프트에 자동 주입하지 않고 에이전트가
          필요 시 session_search / session_recall 도구로 직접 당겨옴(Pull).
        """
        ctx = getattr(runtime, "context", None)
        if not ctx:
            return None

        session_id = self._get_session_id(runtime)
        if session_id != "unknown":
            ctx.session_id = session_id

        recalled_parts = []

        # L3: Semantic Memory 프리페치 (세션 감지 기반 Frozen Snapshot)
        if getattr(ctx, "semantic_memory_enabled", False):
            # 세션이 전환된 순간에만 디스크에서 최신 마크다운 로드 (세션 내 KV-Cache 보존)
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

        # L2: Episodic Memory 자동 인출 (기본값 False: 순정 Hermes의 Pull-only A안 준수)
        # auto_recall_episodic 옵션이 명시적으로 켜진 경우에만 보조적으로 수행
        should_auto_recall = self.auto_recall_episodic or getattr(ctx, "auto_recall_episodic", False)
        if should_auto_recall and getattr(ctx, "episodic_memory_enabled", False):
            user_query = self._extract_last_user_query(state)
            if user_query:
                try:
                    sessions = _run_async_safe(
                        self.episodic_store.search_sessions(
                            user_query,
                            top_k=2,
                            exclude_session_id=session_id,
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

        # 비동기 데몬 스레드 생성 (Hermes: _spawn_background_review)
        thread = threading.Thread(
            target=self._background_review,
            args=(messages_snapshot, session_id),
            daemon=True,
            name="memory-bg-review",
        )
        thread.start()
        logger.info("Background review spawned for session %s", session_id)

        return None

    # ══════════════════════════════════════════════════════════════════
    # Background Review (daemon thread)
    # ══════════════════════════════════════════════════════════════════

    def _background_review(
        self,
        messages_snapshot: List[Dict[str, str]],
        session_id: str,
    ) -> None:
        """Background Review 스레드.

        1. LLM에게 대화 리뷰 요청 → memory() 호출 파싱 실행
        2. EpisodicStore에 세션 요약 저장
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

    # ══════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════

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
    def _get_session_id(runtime: Any, request: Any = None) -> str:
        """runtime 및 request에서 session_id(thread_id)를 지능적으로 추출."""
        # 1. langchain_core의 get_config_from_context() 시도
        try:
            from langchain_core.runnables.config import get_config_from_context
            cfg = get_config_from_context()
            if cfg and isinstance(cfg, dict):
                tid = cfg.get("configurable", {}).get("thread_id")
                if tid:
                    return str(tid)
        except Exception:
            pass

        ctx = getattr(runtime, "context", None)
        if not ctx and request:
            ctx = getattr(getattr(request, "runtime", None), "context", None)

        if ctx:
            sid = getattr(ctx, "session_id", None)
            if sid and sid != "unknown":
                return str(sid)

        if request:
            configurable = getattr(request, "configurable", None)
            if isinstance(configurable, dict) and configurable.get("thread_id"):
                return str(configurable["thread_id"])

        config = getattr(runtime, "config", None)
        if not config and request:
            config = getattr(getattr(request, "runtime", None), "config", None)

        if config:
            if isinstance(config, dict):
                tid = config.get("configurable", {}).get("thread_id")
                if tid:
                    return str(tid)
            elif hasattr(config, "get"):
                configurable = config.get("configurable", {})
                if isinstance(configurable, dict) and configurable.get("thread_id"):
                    return str(configurable["thread_id"])
            elif hasattr(config, "configurable"):
                configurable = getattr(config, "configurable", {})
                if isinstance(configurable, dict) and configurable.get("thread_id"):
                    return str(configurable["thread_id"])

        return "unknown"

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


# ── Module-level helper ──

def _run_async_safe(coro):
    """비동기 코루틴을 안전하게 동기 실행."""
    try:
        loop = asyncio.get_running_loop()
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)
