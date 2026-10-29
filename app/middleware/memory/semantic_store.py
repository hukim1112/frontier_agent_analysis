"""
===============================================================================
Semantic Memory Store — MEMORY.md / USER.md
===============================================================================
Source: frontier-agent-lab/modules/hermes/memory_store.py

§ (section sign) 구분자로 엔트리를 구분하는 마크다운 파일 기반 장기 기억 저장소.

두 개의 파일:
- MEMORY.md: 에이전트의 관찰/학습 기록 (환경, 프로젝트 컨벤션, 도구 특성)
- USER.md: 유저에 대한 정보 (선호도, 커뮤니케이션 스타일, 워크플로우)

설계 원칙 (Hermes 원본 유지):
- Frozen Snapshot: 세션 시작 시 캡처, 세션 중 시스템 프롬프트 불변 (프리픽스 캐시 보존)
- Disk-immediate Write: 도구 호출로 변경 시 즉시 파일에 반영 (다음 세션에 반영)
- Bounded Capacity: 글자 수 제한으로 무한 확장 방지
- 영어 저장: FTS5 검색 효율 + 토큰 효율을 위해 팩트는 영어로 저장
- Background Delete Gate: 사용자가 없는 백그라운드 리뷰의 replace/remove는 바로 반영하지 않고
  pending.json에 제안으로만 남김 (Hermes tools/memory_tool.py _background_delete_gate).
  교육용으로 승인 UI는 두지 않고 apply_pending / discard_pending 메서드만 제공
===============================================================================
"""

import os
import json
import uuid
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

ENTRY_DELIMITER = "\n§\n"
PENDING_FILE = "pending.json"

MEMORY_BLOCK_HEADERS = {
    "memory": "MEMORY (agent's personal notes)",
    "user": "USER PROFILE (who the user is)",
}


class SemanticMemoryStore:
    """§ 구분자 기반 마크다운 메모리 스토어.

    Core Operations:
    - load_from_disk(): 파일 → frozen snapshot + live entries
    - add/replace/remove: live entries 변경 → 즉시 디스크 반영
    - format_for_prompt(): frozen snapshot 반환 (시스템 프롬프트 주입용)
    """

    def __init__(
        self,
        memory_dir: str = "./artifacts/memory",
        memory_char_limit: int = 2200,
        user_char_limit: int = 1375,
    ):
        self.memory_dir = Path(memory_dir)
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit

        # Live state — 도구 호출로 변경됨
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []

        # Frozen snapshot — 세션 시작 시 캡처, 시스템 프롬프트에 주입
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}

    # ── 디스크 I/O ──

    def load_from_disk(self) -> None:
        """MEMORY.md, USER.md 로드 → frozen snapshot 캡처.

        파일이 존재하지 않으면 빈 마크다운 파일(MEMORY.md, USER.md)을 즉시 초기화합니다.
        세션 시작 시 한 번 호출. 이후 시스템 프롬프트에 주입되는 내용은
        이 시점의 스냅샷으로 고정됨 (프리픽스 캐시 보존).
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        for filename in ("MEMORY.md", "USER.md"):
            fpath = self.memory_dir / filename
            if not fpath.exists():
                self._write_file(fpath, [])

        self.memory_entries = self._read_file(self.memory_dir / "MEMORY.md")
        self.user_entries = self._read_file(self.memory_dir / "USER.md")

        # 중복 제거 (순서 유지, 첫 등장 보존)
        self.memory_entries = list(dict.fromkeys(self.memory_entries))
        self.user_entries = list(dict.fromkeys(self.user_entries))

        # Frozen snapshot 캡처
        self._system_prompt_snapshot = {
            "memory": self._render_block("memory", self.memory_entries),
            "user": self._render_block("user", self.user_entries),
        }
        logger.info(
            "SemanticMemoryStore loaded: memory=%d entries, user=%d entries",
            len(self.memory_entries), len(self.user_entries),
        )

    def save_to_disk(self, target: str) -> None:
        """엔트리를 디스크에 즉시 저장. 매 변경 후 호출 (다음 세션 반영용)."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    # ── CRUD 액션 ──

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """새 엔트리 추가. 글자 수 초과 시 에러 반환."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}

        entries = self._entries_for(target)

        # 중복 체크
        if content in entries:
            return self._success_response(target, "Entry already exists (no duplicate added).")

        # 용량 체크
        new_entries = entries + [content]
        new_total = len(ENTRY_DELIMITER.join(new_entries))
        limit = self._char_limit(target)

        if new_total > limit:
            current = self._char_count(target)
            return {
                "success": False,
                "error": (
                    f"Memory at {current:,}/{limit:,} chars. "
                    f"Adding this entry ({len(content)} chars) would exceed the limit. "
                    f"Use 'replace' to merge entries or 'remove' stale entries, then retry."
                ),
                "current_entries": entries,
                "usage": f"{current:,}/{limit:,}",
            }

        entries.append(content)
        self._set_entries(target, entries)
        self.save_to_disk(target)
        return self._success_response(target, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """old_text 부분 문자열 매칭으로 엔트리를 찾아 교체."""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete."}

        entries = self._entries_for(target)
        matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

        if not matches:
            return {
                "success": False,
                "error": f"No entry matched '{old_text}'. Check current_entries and retry.",
                "current_entries": entries,
            }
        if len(matches) > 1:
            unique_texts = {e for _, e in matches}
            if len(unique_texts) > 1:
                return {
                    "success": False,
                    "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                    "matches": [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches],
                }

        idx = matches[0][0]
        limit = self._char_limit(target)

        # 교체 후 용량 체크
        test_entries = entries.copy()
        test_entries[idx] = new_content
        new_total = len(ENTRY_DELIMITER.join(test_entries))

        if new_total > limit:
            current = self._char_count(target)
            return {
                "success": False,
                "error": (
                    f"Replacement would put memory at {new_total:,}/{limit:,} chars. "
                    f"Shorten the new content or remove other entries first."
                ),
                "current_entries": entries,
                "usage": f"{current:,}/{limit:,}",
            }

        entries[idx] = new_content
        self._set_entries(target, entries)
        self.save_to_disk(target)
        return self._success_response(target, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """old_text 부분 문자열 매칭으로 엔트리 삭제."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}

        entries = self._entries_for(target)
        matches = [(i, e) for i, e in enumerate(entries) if old_text in e]

        if not matches:
            return {
                "success": False,
                "error": f"No entry matched '{old_text}'. Check current_entries and retry.",
                "current_entries": entries,
            }
        if len(matches) > 1:
            unique_texts = {e for _, e in matches}
            if len(unique_texts) > 1:
                return {
                    "success": False,
                    "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                    "matches": [e[:80] + ("..." if len(e) > 80 else "") for _, e in matches],
                }

        idx = matches[0][0]
        entries.pop(idx)
        self._set_entries(target, entries)
        self.save_to_disk(target)
        return self._success_response(target, "Entry removed.")

    # ── 승인 대기 제안 (Background Delete Gate) ──

    def stage_pending(
        self, action: str, target: str, content: str = "", old_text: str = "",
        origin: str = "background_review",
    ) -> Dict[str, Any]:
        """replace/remove를 실행하지 않고 pending.json에 제안으로 남깁니다."""
        record = {
            "id": uuid.uuid4().hex[:8],
            "action": action,
            "target": target,
            "content": content,
            "old_text": old_text,
            "origin": origin,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._write_pending(self.list_pending() + [record])
        return {
            "success": True,
            "staged": True,
            "pending_id": record["id"],
            "message": (
                f"Background review may not {action} memory entries unattended. "
                "The proposal was staged in pending.json (apply_pending to apply, discard_pending to drop)."
            ),
        }

    def list_pending(self) -> List[Dict[str, Any]]:
        path = self.memory_dir / PENDING_FILE
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read %s: %s", path, e)
            return []

    def apply_pending(self, pending_id: str) -> Dict[str, Any]:
        """승인: 제안을 실제로 반영하고 대기 목록에서 지웁니다 (반영 실패 시 목록에 남김)."""
        pending = self.list_pending()
        record = next((p for p in pending if p.get("id") == pending_id), None)
        if record is None:
            return {"success": False, "error": f"No pending proposal '{pending_id}'."}
        if record["action"] == "replace":
            result = self.replace(record["target"], record["old_text"], record["content"])
        elif record["action"] == "remove":
            result = self.remove(record["target"], record["old_text"])
        else:
            result = {"success": False, "error": f"Unsupported pending action '{record['action']}'."}
        if result.get("success"):
            self._write_pending([p for p in pending if p.get("id") != pending_id])
        return result

    def discard_pending(self, pending_id: str) -> Dict[str, Any]:
        """거절: 반영하지 않고 대기 목록에서 지웁니다."""
        pending = self.list_pending()
        remaining = [p for p in pending if p.get("id") != pending_id]
        if len(remaining) == len(pending):
            return {"success": False, "error": f"No pending proposal '{pending_id}'."}
        self._write_pending(remaining)
        return {"success": True, "message": f"Pending proposal '{pending_id}' discarded."}

    def _write_pending(self, pending: List[Dict[str, Any]]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        path = self.memory_dir / PENDING_FILE
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    # ── 시스템 프롬프트 인터페이스 ──

    def format_for_prompt(self, target: str) -> Optional[str]:
        """Frozen snapshot 반환. 세션 중 불변.

        Returns None if the snapshot is empty (no entries at load time).
        """
        block = self._system_prompt_snapshot.get(target, "")
        return block if block else None

    # ── Internal helpers ──

    def _entries_for(self, target: str) -> List[str]:
        if target == "user":
            return self.user_entries
        return self.memory_entries

    def _set_entries(self, target: str, entries: List[str]) -> None:
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        entries = self._entries_for(target)
        if not entries:
            return 0
        return len(ENTRY_DELIMITER.join(entries))

    def _char_limit(self, target: str) -> int:
        if target == "user":
            return self.user_char_limit
        return self.memory_char_limit

    def _path_for(self, target: str) -> Path:
        if target == "user":
            return self.memory_dir / "USER.md"
        return self.memory_dir / "MEMORY.md"

    def _success_response(self, target: str, message: str) -> Dict[str, Any]:
        entries = self._entries_for(target)
        current = self._char_count(target)
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        return {
            "success": True,
            "done": True,
            "target": target,
            "usage": f"{pct}% — {current:,}/{limit:,} chars",
            "entry_count": len(entries),
            "message": message,
            "note": "Write saved. Do not repeat this operation.",
        }

    def _render_block(self, target: str, entries: List[str]) -> str:
        """시스템 프롬프트 블록 렌더링."""
        if not entries:
            return ""
        limit = self._char_limit(target)
        content = ENTRY_DELIMITER.join(entries)
        current = len(content)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0

        header_key = "user" if target == "user" else "memory"
        header = f"{MEMORY_BLOCK_HEADERS[header_key]} [{pct}% — {current:,}/{limit:,} chars]"
        separator = "═" * 46
        return f"{separator}\n{header}\n{separator}\n{content}"

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """§ 구분자로 엔트리 파싱."""
        if not path.exists():
            return []
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            logger.warning("Failed to read %s: %s", path, e)
            return []
        if not raw.strip():
            return []
        entries = [e.strip() for e in raw.split(ENTRY_DELIMITER)]
        return [e for e in entries if e]

    @staticmethod
    def _write_file(path: Path, entries: List[str]) -> None:
        """엔트리를 § 구분자로 직렬화하여 파일에 저장."""
        content = ENTRY_DELIMITER.join(entries) if entries else ""
        try:
            path.write_text(content, encoding="utf-8")
        except OSError as e:
            logger.error("Failed to write %s: %s", path, e)
