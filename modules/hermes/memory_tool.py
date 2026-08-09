"""
Semantic Memory Tool — @tool: memory(add/replace/remove)

에이전트가 직접 호출하여 장기 기억(MEMORY.md / USER.md)을 관리하는 도구.
SemanticMemoryStore 인스턴스를 클로저로 바인딩합니다.
"""

import json
from typing import Optional
from langchain.tools import tool


def create_memory_tool(semantic_store):
    """SemanticMemoryStore를 바인딩한 memory 도구를 생성합니다.

    Args:
        semantic_store: SemanticMemoryStore 인스턴스

    Returns:
        LangChain @tool 데코레이터가 적용된 memory 함수
    """

    @tool
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
            result = semantic_store.add(target, content)
        elif action == "replace":
            result = semantic_store.replace(target, old_text, content)
        elif action == "remove":
            result = semantic_store.remove(target, old_text)
        else:
            result = {
                "success": False,
                "error": f"Unknown action '{action}'. Use add, replace, or remove.",
            }

        return json.dumps(result, ensure_ascii=False, indent=2)

    return memory
