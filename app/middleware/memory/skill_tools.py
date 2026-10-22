"""
===============================================================================
Skill Tools — skills_list / skill_view / skill_manage
===============================================================================
Source: hermes-agent/tools/skills_tool.py + skill_manager_tool.py (교육용 축약 재구성)

같은 도구 팩토리를 두 곳에서 씁니다:
- 메인 에이전트 (포그라운드): review=None → 사용자 지시로 만든 스킬은 사용자 소유
- 백그라운드 스킬 리뷰 에이전트: review=ReviewSession → 자율 쓰기 가드 적용
  (에이전트 소유 스킬만 수정, read-before-write, 삭제는 보관)

Hermes는 skill_manage를 operations 배열(원자적 배치 + 실패 시 롤백)로 노출합니다.
교육용으로 한 번에 작업 하나를 받는 평면(flat) 형태만 남겼습니다.
===============================================================================
"""

import json
import threading
from typing import Callable, List, Optional

from langchain.tools import tool as langchain_tool

from app.middleware.memory.procedural_store import ProceduralMemoryStore, ReviewSession

SKILL_VIEW_TOOL = "skill_view"
SKILL_MANAGE_TOOL = "skill_manage"
SKILLS_LIST_TOOL = "skills_list"


def _dump(result: dict) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


def build_skill_tools(
    get_store: Callable[[], ProceduralMemoryStore],
    review: Optional[ReviewSession] = None,
    cancel_event: Optional[threading.Event] = None,
    include_list: bool = False,
) -> List:
    """스토어 해석 함수에 바인딩된 스킬 도구 목록을 생성합니다.

    Args:
        get_store: 호출 시점의 사용자에 맞는 ProceduralMemoryStore를 돌려주는 함수
        review: 백그라운드 리뷰 컨텍스트 (None이면 포그라운드)
        cancel_event: 설정되면 쓰기를 거부 (새 턴이 시작되어 리뷰가 취소된 경우)
        include_list: skills_list 도구 포함 여부 (메인 에이전트는 L4 카탈로그가 대신함)
    """

    @langchain_tool
    def skills_list() -> str:
        """List learned skills (name + description only). Use skill_view(name) to load full content."""
        skills = get_store().list_skills()
        return _dump({
            "success": True,
            "skills": [{"name": s["name"], "description": s["description"]} for s in skills],
            "count": len(skills),
        })

    @langchain_tool
    def skill_view(name: str, file_path: str = "") -> str:
        """Load a learned skill's full SKILL.md, or one of its linked files.

        Skills hold procedures for recurring task types: the steps, the commands that work,
        the user's preferences for the result, and the pitfalls. The first call returns
        SKILL.md plus a 'linked_files' map of references/templates/scripts; to read one of
        those, call again with file_path (e.g. 'references/api.md').

        Args:
            name: Skill name as listed in PROCEDURAL MEMORY (learned skills)
            file_path: Optional path inside the skill directory; omit for SKILL.md
        """
        return _dump(get_store().view(name, file_path=file_path or None, review=review))

    @langchain_tool
    def skill_manage(
        action: str,
        name: str,
        content: str = "",
        old_string: str = "",
        new_string: Optional[str] = None,
        replace_all: bool = False,
        file_path: str = "",
        file_content: Optional[str] = None,
    ) -> str:
        """Create, update, or delete skills — your procedural memory for recurring task types.

        Actions:
        - create: full SKILL.md in `content` (YAML frontmatter with name + description, then body).
        - patch: targeted fix — replace the EXACT `old_string` with `new_string` (preferred for updates;
          read the file with skill_view first). Use `file_path` to patch a supporting file.
        - write_file / remove_file: supporting files; `file_path` must start with references/,
          templates/, scripts/, or assets/.
        - delete: remove the skill.

        Keep the description a self-contained trigger: 'Use when <trigger>. <one-line behavior>.'
        Write lessons, not logs: imperative rule + why, no dates/ticket IDs/incident narration,
        one rule per lesson, references/ named by topic (extend before adding).

        Args:
            action: "create" | "patch" | "write_file" | "remove_file" | "delete"
            name: Skill name (lowercase letters, digits, '-', '_', '.'; max 64 chars)
            content: Full SKILL.md text (create)
            old_string: Exact text to find (patch)
            new_string: Replacement text; empty string deletes the match (patch)
            replace_all: Replace every occurrence instead of requiring a unique match (patch)
            file_path: Supporting file path relative to the skill directory
            file_content: Content for write_file
        """
        if cancel_event is not None and cancel_event.is_set():
            return _dump({"success": False, "error": "Review cancelled: a new user turn started. Stop now."})
        store = get_store()
        if action == "create":
            result = store.create(name, content, review=review)
        elif action == "patch":
            result = store.patch(
                name, old_string, new_string, file_path=file_path or None,
                replace_all=replace_all, review=review,
            )
        elif action == "write_file":
            result = store.write_file(name, file_path, file_content, review=review)
        elif action == "remove_file":
            result = store.remove_file(name, file_path, review=review)
        elif action == "delete":
            result = store.delete(name, review=review)
        else:
            result = {
                "success": False,
                "error": f"Unknown action '{action}'. Use create, patch, write_file, remove_file, or delete.",
            }
        return _dump(result)

    tools = [skill_view, skill_manage]
    if include_list:
        tools.insert(0, skills_list)
    return tools
