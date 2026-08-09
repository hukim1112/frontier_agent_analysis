"""
Session Search Tools — @tool: session_search, session_recall

에이전트가 직접 호출하여 과거 세션을 검색하고, Anchor 기반으로 대화를 인출하는 도구.
EpisodicStore 인스턴스를 클로저로 바인딩합니다.

Hermes의 session_search_tool.py에서 3가지 모드를 단순화:
1. DISCOVERY: query 기반 FTS5 검색
2. BROWSE: 최근 세션 목록
3. RECALL: Anchor 기반 과거 대화 인출
"""

import json
import asyncio
from typing import Optional
from langchain.tools import tool


def _run_async(coro):
    """비동기 코루틴을 동기 컨텍스트에서 실행."""
    try:
        loop = asyncio.get_running_loop()
        # 이미 이벤트 루프 내부면 새 스레드에서 실행
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        return asyncio.run(coro)


def create_session_search_tool(episodic_store):
    """EpisodicStore를 바인딩한 session_search 도구를 생성합니다."""

    @tool
    def session_search(
        query: str = "",
        limit: int = 3,
    ) -> str:
        """Search past conversation sessions to recall relevant context.

        Two modes (inferred from arguments):
        1. DISCOVERY: Pass a query string to search for related past sessions via FTS5.
           Returns session summaries, keywords, and match snippets.
        2. BROWSE: Call with no arguments to list recent sessions chronologically.

        Use this tool when:
        - The user references a past conversation ("we discussed X before")
        - You need context from a previous session to answer the current question
        - The user asks about their history

        After finding a relevant session_id, use session_recall to drill down
        into the actual messages.

        Args:
            query: Search query in English (empty = browse recent)
            limit: Max sessions to return (default 3)

        Returns:
            JSON with session list: [{session_id, summary, keywords, snippet}, ...]
        """
        if query.strip():
            results = _run_async(
                episodic_store.search_sessions(query, top_k=limit)
            )
        else:
            results = _run_async(episodic_store.browse_recent(limit=limit))

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


def create_session_recall_tool(episodic_store):
    """EpisodicStore를 바인딩한 session_recall 도구를 생성합니다."""

    @tool
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

        Use this tool after session_search finds a relevant session_id.

        Args:
            session_id: The session ID to recall messages from
            anchor_message: Keywords to locate the most relevant message (optional)
            window: Number of messages before/after the anchor (default 5)

        Returns:
            JSON with messages: [{role, content, id}, ...]
        """
        messages = _run_async(
            episodic_store.get_anchored_view(
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
