"""
===============================================================================
Procedural Memory Store — 학습된 스킬 (skills/<name>/SKILL.md)
===============================================================================
Source: hermes-agent/tools/skill_manager_tool.py + skills_tool.py + skill_usage.py
        + agent/curator.py (교육용 축약 재구성)

Semantic Memory(MEMORY.md / USER.md)가 "무엇을 / 누구를" 기억한다면,
Procedural Memory는 "어떻게" — 반복되는 작업 유형을 이 사용자에게 맞게 처리하는 절차를 기억합니다.

디렉터리 구조 (사용자별 격리):
    artifacts/users/{user_id}/skills/
    ├── .usage.json                 ← 사용 기록 + 출처(created_by). SKILL.md 본문과 분리
    ├── .archive/<name>/            ← Curator가 보관한 스킬 (삭제하지 않고 복구 가능)
    └── <name>/
        ├── SKILL.md                ← YAML frontmatter(name, description) + 절차 본문
        ├── references/             ← 가끔만 필요한 깊이 있는 자료 (주제별 파일)
        ├── templates/              ← 복사해서 고쳐 쓰는 시작 파일
        └── scripts/                ← 그대로 다시 실행하는 스크립트

설계 원칙 (Hermes 원본 유지):
- Progressive Disclosure: 프롬프트에는 이름 + 설명만, 본문은 skill_view로 필요할 때 연다
- Frozen Snapshot: 카탈로그는 세션 시작 시 캡처, 세션 중 불변 (프리픽스 캐시 보존)
- Provenance: 백그라운드 리뷰가 만든 스킬만 created_by="agent" (자율 수정 대상).
  사용자 요청으로 포그라운드에서 만든 스킬은 사용자 소유 → 백그라운드 리뷰가 수정할 수 없음
- Read-before-write: 백그라운드 리뷰는 이번 리뷰에서 skill_view로 읽은 파일만 수정할 수 있음
- 보관(archive)만 하고 삭제하지 않음: 백그라운드 삭제와 Curator 전이는 .archive/로 이동

교육용으로 생략한 Hermes 운영 기능 (notebooks/04 참고):
승인 대기(staging), 감사 원장(ledger), 원자적 배치(operations 배열 + 롤백), 퍼지 매칭 patch,
보안 스캔/린터, 원격 동기화, Curator 백업, LLM 통합(consolidation), pin/hub/external 구분
===============================================================================
"""

import json
import re
import shutil
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

VALID_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
ALLOWED_SUBDIRS = {"references", "templates", "scripts", "assets"}
USAGE_FILE = ".usage.json"
ARCHIVE_DIR = ".archive"

STATE_ACTIVE, STATE_STALE, STATE_ARCHIVED = "active", "stale", "archived"
CREATED_BY_AGENT = "agent"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def parse_frontmatter(content: str) -> Optional[Dict[str, str]]:
    """SKILL.md 상단 `---` 블록의 key: value 쌍을 파싱합니다. 블록이 없으면 None."""
    match = re.match(r"^---\s*\n(.*?)\n---\s*(\n|$)", content or "", re.DOTALL)
    if not match:
        return None
    meta: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta


class ReviewSession:
    """백그라운드 리뷰 1회의 쓰기 권한 컨텍스트.

    - 이 객체를 넘긴 쓰기는 "사용자가 없는 자율 쓰기"로 취급되어 가드가 적용됩니다.
    - view()로 읽은 파일 경로를 기록해 read-before-write를 강제합니다.
    """

    def __init__(self) -> None:
        self.read_paths: Set[Path] = set()

    def mark_read(self, path: Path) -> None:
        self.read_paths.add(path.resolve())

    def has_read(self, path: Path) -> bool:
        return path.resolve() in self.read_paths


class ProceduralMemoryStore:
    """사용자별 학습 스킬 저장소.

    Core Operations:
    - load_from_disk() / format_for_prompt(): 카탈로그 Frozen Snapshot (L4 주입용)
    - list_skills() / view(): Progressive Disclosure 읽기 (view가 사용 기록 갱신)
    - create / patch / write_file / remove_file / delete: 검증된 쓰기
    - apply_lifecycle() / restore(): Curator — 사용 기록 기반 active → stale → archived
    """

    def __init__(
        self,
        skills_dir: str = "./artifacts/users/default/skills",
        max_skill_chars: int = 20_000,
        max_file_bytes: int = 200_000,
    ):
        self.skills_dir = Path(skills_dir)
        self.max_skill_chars = max_skill_chars
        self.max_file_bytes = max_file_bytes
        self._usage_lock = threading.RLock()
        self._prompt_snapshot: str = ""

    # ══════════════════════════════════════════════════════════════════
    # 카탈로그 (Frozen Snapshot)
    # ══════════════════════════════════════════════════════════════════

    def load_from_disk(self) -> None:
        """스킬 카탈로그를 스캔해 프롬프트 스냅샷을 캡처합니다. 세션 시작 시 한 번 호출."""
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self._prompt_snapshot = self._render_catalog(self.list_skills())
        logger.info("ProceduralMemoryStore loaded: %s", self.skills_dir)

    def format_for_prompt(self) -> Optional[str]:
        """Frozen snapshot 반환. 스킬이 없으면 None."""
        return self._prompt_snapshot or None

    def list_skills(self) -> List[Dict[str, Any]]:
        """활성 스킬 목록(이름 + 설명 + 출처). 본문은 포함하지 않습니다 (Tier 1)."""
        if not self.skills_dir.exists():
            return []
        usage = self.load_usage()
        skills = []
        for skill_dir in sorted(self.skills_dir.iterdir()):
            skill_md = skill_dir / "SKILL.md"
            if skill_dir.name.startswith(".") or not skill_md.is_file():
                continue
            meta = parse_frontmatter(skill_md.read_text(encoding="utf-8")) or {}
            record = usage.get(skill_dir.name, {})
            skills.append({
                "name": skill_dir.name,
                "description": " ".join(meta.get("description", "").split()),
                "created_by": record.get("created_by"),
                "state": record.get("state", STATE_ACTIVE),
            })
        return skills

    @staticmethod
    def _render_catalog(skills: List[Dict[str, Any]]) -> str:
        if not skills:
            return ""
        separator = "═" * 46
        lines = [f"    - {s['name']}: {s['description']}" for s in skills]
        return (
            f"{separator}\nPROCEDURAL MEMORY (learned skills) [{len(skills)} skills]\n{separator}\n"
            "Before replying, scan the skills below. If a skill matches or is even partially relevant "
            "to your task, you MUST load it with skill_view(name) and follow its instructions. Err on the "
            "side of loading — skills encode this user's preferred procedure and output format, so load "
            "them even for tasks you already know how to do.\n"
            "If a loaded skill is wrong or missing a step, fix it with skill_manage(action=\"patch\"). "
            "After a difficult, multi-step task, offer to save the approach as a skill.\n"
            "<available_skills>\n" + "\n".join(lines) + "\n</available_skills>"
        )

    # ══════════════════════════════════════════════════════════════════
    # 읽기 (Progressive Disclosure)
    # ══════════════════════════════════════════════════════════════════

    def view(
        self, name: str, file_path: Optional[str] = None, review: Optional[ReviewSession] = None
    ) -> Dict[str, Any]:
        """SKILL.md 본문(또는 보조 파일)을 반환하고 사용 기록을 갱신합니다."""
        skill_dir = self._existing_skill_dir(name)
        if skill_dir is None:
            return self._not_found(name)

        if file_path:
            target, err = self._resolve_support_file(skill_dir, file_path)
            if err:
                return err
            if not target.is_file():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "linked_files": self._linked_files(skill_dir),
                }
            content = target.read_text(encoding="utf-8")
        else:
            target = skill_dir / "SKILL.md"
            content = target.read_text(encoding="utf-8")

        if review is not None:
            review.mark_read(target)
        else:
            # 포그라운드 열람 = 스킬을 실제로 쓰려는 것 → 사용으로 집계 (Curator stale 타이머 기준)
            self._bump(name, "view_count", "last_viewed_at", also_use=True)

        result: Dict[str, Any] = {
            "success": True,
            "name": name,
            "file": file_path or "SKILL.md",
            "content": content,
        }
        if not file_path:
            result["description"] = (parse_frontmatter(content) or {}).get("description", "")
            linked = self._linked_files(skill_dir)
            if linked:
                result["linked_files"] = linked
                result["usage_hint"] = (
                    "To view linked files, call skill_view(name, file_path) "
                    "where file_path is e.g. 'references/api.md'."
                )
        return result

    # ══════════════════════════════════════════════════════════════════
    # 쓰기 (검증 + 가드)
    # ══════════════════════════════════════════════════════════════════

    def create(self, name: str, content: str, review: Optional[ReviewSession] = None) -> Dict[str, Any]:
        """새 스킬 생성. 백그라운드 리뷰가 만든 스킬만 created_by="agent"로 기록됩니다."""
        err = self._validate_name(name) or self._validate_skill_md(content, expected_name=name)
        if err:
            return self._err(err)
        if (self.skills_dir / name).exists():
            return self._err(
                f"A skill named '{name}' already exists. Use skill_view then action='patch' to update it."
            )
        skill_dir = self.skills_dir / name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(content, encoding="utf-8")

        now = _iso(_now())
        with self._usage_lock:
            usage = self.load_usage()
            usage[name] = {
                "created_by": CREATED_BY_AGENT if review is not None else None,
                "created_at": now,
                "state": STATE_ACTIVE,
                "view_count": 0,
                "use_count": 0,
                "patch_count": 0,
                "last_used_at": None,
                "last_patched_at": None,
            }
            self._save_usage(usage)
        return {
            "success": True,
            "message": f"Skill '{name}' created.",
            "action": "create",
            "name": name,
            "hint": (
                "To add reference files, templates, or scripts, use skill_manage(action='write_file', "
                f"name='{name}', file_path='references/<topic>.md', file_content='...')"
            ),
        }

    def patch(
        self,
        name: str,
        old_string: str,
        new_string: str,
        file_path: Optional[str] = None,
        replace_all: bool = False,
        review: Optional[ReviewSession] = None,
    ) -> Dict[str, Any]:
        """SKILL.md(기본) 또는 보조 파일의 정확한 문자열 교체. replace_all이 아니면 유일 매칭만 허용."""
        if not old_string:
            return self._err(
                "old_string is required for 'patch' and must be the EXACT text currently in the file. "
                "Call skill_view first and copy the snippet verbatim."
            )
        if new_string is None:
            return self._err("new_string is required for 'patch'. Use an empty string to delete matched text.")
        skill_dir, guard = self._locate_for_write(name, "patch", review)
        if guard:
            return guard
        if file_path:
            target, err = self._resolve_support_file(skill_dir, file_path)
            if err:
                return err
        else:
            target = skill_dir / "SKILL.md"
        label = file_path or "SKILL.md"
        if not target.is_file():
            return self._err(f"File not found: {label}")
        if guard := self._read_before_write_guard(name, target, "patch", label, review):
            return guard

        content = target.read_text(encoding="utf-8")
        count = content.count(old_string)
        if count == 0:
            return {**self._err(f"old_string not found in {label}."), "file_preview": content[:500]}
        if count > 1 and not replace_all:
            return self._err(
                f"old_string matched {count} times in {label}. Add surrounding context to make it unique, "
                "or pass replace_all=true."
            )
        new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)
        if not file_path:
            if err := self._validate_skill_md(new_content, expected_name=name):
                return self._err(f"Patch would break SKILL.md structure: {err}")
        elif len(new_content.encode("utf-8")) > self.max_file_bytes:
            return self._err(f"{label} would exceed {self.max_file_bytes:,} bytes.")
        target.write_text(new_content, encoding="utf-8")
        self._bump(name, "patch_count", "last_patched_at")
        replaced = count if replace_all else 1
        return {
            "success": True,
            "message": f"Patched {label} in skill '{name}' ({replaced} replacement{'s' if replaced > 1 else ''}).",
            "action": "patch",
            "name": name,
            "file_path": file_path,
            "old": old_string[:200],
            "new": new_string[:200],
        }

    def write_file(
        self, name: str, file_path: str, file_content: str, review: Optional[ReviewSession] = None
    ) -> Dict[str, Any]:
        """보조 파일(references/ templates/ scripts/ assets/) 생성 또는 덮어쓰기."""
        if file_content is None:
            return self._err("file_content is required for 'write_file'.")
        skill_dir, guard = self._locate_for_write(name, "write_file", review)
        if guard:
            return guard
        target, err = self._resolve_support_file(skill_dir, file_path)
        if err:
            return err
        if len(file_content.encode("utf-8")) > self.max_file_bytes:
            return self._err(f"{file_path} exceeds {self.max_file_bytes:,} bytes.")
        # 기존 파일을 덮어쓸 때만 read-before-write 적용 (새 파일은 읽을 대상이 없음)
        if target.exists():
            if guard := self._read_before_write_guard(name, target, "write_file", file_path, review):
                return guard
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(file_content, encoding="utf-8")
        self._bump(name, "patch_count", "last_patched_at")
        return {
            "success": True,
            "message": f"File '{file_path}' written to skill '{name}'.",
            "action": "write_file",
            "name": name,
            "file_path": file_path,
        }

    def remove_file(self, name: str, file_path: str, review: Optional[ReviewSession] = None) -> Dict[str, Any]:
        """보조 파일 삭제."""
        skill_dir, guard = self._locate_for_write(name, "remove_file", review)
        if guard:
            return guard
        target, err = self._resolve_support_file(skill_dir, file_path)
        if err:
            return err
        if not target.is_file():
            return self._err(f"File not found: {file_path}")
        if guard := self._read_before_write_guard(name, target, "remove_file", file_path, review):
            return guard
        target.unlink()
        self._bump(name, "patch_count", "last_patched_at")
        return {
            "success": True,
            "message": f"File '{file_path}' removed from skill '{name}'.",
            "action": "remove_file",
            "name": name,
            "file_path": file_path,
        }

    def delete(self, name: str, review: Optional[ReviewSession] = None) -> Dict[str, Any]:
        """스킬 삭제. 포그라운드(사용자 지시)는 실제 삭제, 백그라운드는 복구 가능한 보관."""
        skill_dir, guard = self._locate_for_write(name, "delete", review)
        if guard:
            return guard
        if review is not None:
            return {**self.archive(name), "action": "delete"}
        shutil.rmtree(skill_dir)
        with self._usage_lock:
            usage = self.load_usage()
            usage.pop(name, None)
            self._save_usage(usage)
        return {"success": True, "message": f"Skill '{name}' deleted.", "action": "delete", "name": name}

    # ══════════════════════════════════════════════════════════════════
    # Curator — 사용 기록 기반 생애주기 (LLM 없음)
    # ══════════════════════════════════════════════════════════════════

    def apply_lifecycle(
        self,
        stale_after_days: int = 14,
        archive_after_days: int = 30,
        now: Optional[datetime] = None,
    ) -> Dict[str, List[str]]:
        """에이전트가 만든 스킬만 마지막 사용 시각 기준으로 전이합니다.

        - active: 마지막 사용 후 stale_after_days 경과 → stale
        - stale: 마지막 사용 후 archive_after_days 경과 → archived (.archive/로 이동)
        - 사용자 소유 스킬은 건드리지 않습니다. stale 스킬을 다시 열면 active로 돌아갑니다.
        """
        now = now or _now()
        report: Dict[str, List[str]] = {"marked_stale": [], "archived": []}
        for name, record in list(self.load_usage().items()):
            if record.get("created_by") != CREATED_BY_AGENT or record.get("state") == STATE_ARCHIVED:
                continue
            anchor_raw = record.get("last_used_at") or record.get("created_at")
            if not anchor_raw:
                continue
            idle = now - datetime.fromisoformat(anchor_raw)
            if idle >= timedelta(days=archive_after_days):
                if self.archive(name).get("success"):
                    report["archived"].append(name)
            elif idle >= timedelta(days=stale_after_days) and record.get("state") != STATE_STALE:
                self._set_state(name, STATE_STALE)
                report["marked_stale"].append(name)
        return report

    def archive(self, name: str) -> Dict[str, Any]:
        """스킬을 .archive/로 옮깁니다 (사용 기록은 archived 상태로 유지 → restore 가능)."""
        skill_dir = self._existing_skill_dir(name)
        if skill_dir is None:
            return self._not_found(name)
        dest = self.skills_dir / ARCHIVE_DIR / name
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(skill_dir), str(dest))
        self._set_state(name, STATE_ARCHIVED, archived_at=_iso(_now()))
        return {"success": True, "message": f"Skill '{name}' archived.", "name": name}

    def restore(self, name: str) -> Dict[str, Any]:
        """보관된 스킬을 복구합니다."""
        src = self.skills_dir / ARCHIVE_DIR / name
        if not src.is_dir():
            return self._err(f"No archived skill named '{name}'.")
        if (self.skills_dir / name).exists():
            return self._err(f"An active skill named '{name}' already exists.")
        shutil.move(str(src), str(self.skills_dir / name))
        self._set_state(name, STATE_ACTIVE, last_used_at=_iso(_now()))
        return {"success": True, "message": f"Skill '{name}' restored.", "name": name}

    # ══════════════════════════════════════════════════════════════════
    # 사용 기록 (.usage.json 사이드카)
    # ══════════════════════════════════════════════════════════════════

    def load_usage(self) -> Dict[str, Dict[str, Any]]:
        path = self.skills_dir / USAGE_FILE
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("Failed to read %s: %s", path, e)
            return {}

    def is_agent_managed(self, name: str) -> bool:
        return self.load_usage().get(name, {}).get("created_by") == CREATED_BY_AGENT

    def _save_usage(self, usage: Dict[str, Dict[str, Any]]) -> None:
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        path = self.skills_dir / USAGE_FILE
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(usage, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _bump(self, name: str, counter: str, stamp: str, also_use: bool = False) -> None:
        now = _iso(_now())
        with self._usage_lock:
            usage = self.load_usage()
            record = usage.setdefault(name, {"created_by": None, "state": STATE_ACTIVE})
            record[counter] = int(record.get(counter) or 0) + 1
            record[stamp] = now
            if also_use:
                record["use_count"] = int(record.get("use_count") or 0) + 1
                record["last_used_at"] = now
                if record.get("state") == STATE_STALE:
                    record["state"] = STATE_ACTIVE
            self._save_usage(usage)

    def _set_state(self, name: str, state: str, **fields: Any) -> None:
        with self._usage_lock:
            usage = self.load_usage()
            record = usage.setdefault(name, {"created_by": None})
            record["state"] = state
            record.update(fields)
            self._save_usage(usage)

    # ══════════════════════════════════════════════════════════════════
    # 검증 & 가드
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _validate_name(name: str) -> Optional[str]:
        if not name:
            return "Skill name is required."
        if len(name) > MAX_NAME_LENGTH:
            return f"Skill name exceeds {MAX_NAME_LENGTH} characters."
        if not VALID_NAME_RE.match(name):
            return (
                f"Invalid skill name '{name}'. Use lowercase letters, numbers, hyphens, dots, and underscores, "
                "starting with a letter or digit."
            )
        return None

    def _validate_skill_md(self, content: str, expected_name: str) -> Optional[str]:
        if not content or not content.strip():
            return "SKILL.md content cannot be empty."
        if len(content) > self.max_skill_chars:
            return f"SKILL.md is {len(content):,} chars (limit {self.max_skill_chars:,}). Move depth into references/."
        meta = parse_frontmatter(content)
        if meta is None:
            return "SKILL.md must start with YAML frontmatter (---\\nname: ...\\ndescription: ...\\n---)."
        if not meta.get("name") or not meta.get("description"):
            return "Frontmatter must include both 'name' and 'description'."
        if meta["name"] != expected_name:
            return f"Frontmatter name '{meta['name']}' must match the skill name '{expected_name}'."
        if len(meta["description"]) > MAX_DESCRIPTION_LENGTH:
            return f"Description exceeds {MAX_DESCRIPTION_LENGTH} characters."
        return None

    def _resolve_support_file(self, skill_dir: Path, file_path: str):
        """보조 파일 경로 검증: 상대 경로, 허용된 하위 폴더, 스킬 디렉터리 밖으로 탈출 금지."""
        normalized = (file_path or "").replace("\\", "/")
        parts = [p for p in normalized.split("/") if p]
        if not parts or normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
            return None, self._err("file_path must be a relative path like 'references/topic.md'.")
        if any(p in (".", "..") for p in parts):
            return None, self._err("file_path must not contain '.' or '..' segments.")
        if parts[0] not in ALLOWED_SUBDIRS or len(parts) < 2:
            return None, self._err(
                f"file_path must start with one of {sorted(ALLOWED_SUBDIRS)}/ (e.g. 'references/topic.md')."
            )
        target = (skill_dir / Path(*parts)).resolve()
        if not target.is_relative_to(skill_dir.resolve()):
            return None, self._err("file_path escapes the skill directory.")
        return target, None

    def _locate_for_write(self, name: str, action: str, review: Optional[ReviewSession]):
        """기존 스킬 위치 확인 + 백그라운드 쓰기 가드 (에이전트 소유 스킬만 자율 수정)."""
        skill_dir = self._existing_skill_dir(name)
        if skill_dir is None:
            return None, self._not_found(name)
        if review is not None and not self.is_agent_managed(name):
            return None, self._err(
                f"Refusing background {action} for skill '{name}': the skill is user-owned "
                "(not created by background review). User-owned skills are off-limits to autonomous "
                "updates — mention the issue in your reply instead."
            )
        return skill_dir, None

    def _read_before_write_guard(
        self, name: str, target: Path, action: str, label: str, review: Optional[ReviewSession]
    ) -> Optional[Dict[str, Any]]:
        if review is None or review.has_read(target):
            return None
        hint = f"skill_view(name='{name}')" if label == "SKILL.md" else f"skill_view(name='{name}', file_path='{label}')"
        return {
            **self._err(
                f"Refusing {action} for skill '{name}': the current {label} has not been loaded in this "
                f"review. Call {hint}, then retry the write based on the content just returned."
            ),
            "read_before_write_required": True,
        }

    def _existing_skill_dir(self, name: str) -> Optional[Path]:
        if self._validate_name(name):
            return None
        skill_dir = self.skills_dir / name
        return skill_dir if (skill_dir / "SKILL.md").is_file() else None

    @staticmethod
    def _linked_files(skill_dir: Path) -> Dict[str, List[str]]:
        linked: Dict[str, List[str]] = {}
        for sub in sorted(ALLOWED_SUBDIRS):
            sub_dir = skill_dir / sub
            if sub_dir.is_dir():
                files = sorted(
                    p.relative_to(skill_dir).as_posix() for p in sub_dir.rglob("*") if p.is_file()
                )
                if files:
                    linked[sub] = files
        return linked

    def _not_found(self, name: str) -> Dict[str, Any]:
        available = [s["name"] for s in self.list_skills()]
        return {"success": False, "error": f"Skill '{name}' not found.", "available_skills": available}

    @staticmethod
    def _err(message: str) -> Dict[str, Any]:
        return {"success": False, "error": message}
