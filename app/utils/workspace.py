"""
===============================================================================
Unified User / Session / Sub-Agent Workspace Path Resolver (app/utils/workspace.py)
===============================================================================
계층형 아티팩트 디렉터리 구조 (Single Source of Truth):

artifacts/
└── users/
    └── {user_id}/
        ├── memory/                                      # 유저 단위 영구 장기 기억 (마크다운)
        │   ├── USER.md
        │   └── MEMORY.md
        └── sessions/                                    # 메인 대화(Supervisor) 단위 워크스페이스
            └── {session_id}/
                ├── plan/                                # plan_state.json, task_state.json
                ├── logs/                                # audit.jsonl (403 protected internal log)
                ├── scripts/                             # 작성된 파이썬/분석 스크립트
                ├── outputs/                             # 생성된 데이터, 차트, HTML 대시보드, 엑셀
                └── subagents/                           # 호출된 서브에이전트별 격리 워크스페이스
                    └── {subagent_role}_{short_id}/
                        ├── plan/                        # extraction_plan.json, plan_state.json 등
                        ├── logs/                        # 서브에이전트 전용 audit.jsonl
                        ├── scripts/                     # 서브에이전트가 작성한 크롤링/분석 코드 (.py)
                        └── outputs/                     # 서브에이전트가 생성한 산출물 (.json, .png, .html, .xlsx)

설계 원칙:
1. 서브에이전트 여부는 thread_id 문자열 정규식으로 추측하지 않고, 호출자가 명시적으로 전달한
   `parent_session_id` 유무로만 판단합니다 ("정체성은 추측하지 않고 전달한다").
2. 모든 SQLite 데이터베이스(`checkpoints.db`, `episodic.db`, `chainlit.db`)는 `app/database/`에서 중앙 관리하며,
   `artifacts/users/{user_id}/memory/`에는 사람이 읽고 편집할 수 있는 마크다운 장기 기억(`USER.md`, `MEMORY.md`)만 보관합니다.
   (웹 정적 마운트 `/artifacts`에서는 `memory/`, `logs/`, `plan/` 접근을 차단하고 `outputs/` 산출물만 노출합니다.)
===============================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
USERS_ROOT_DIR = PROJECT_ROOT / "artifacts" / "users"
USERS_BASE_DIR = USERS_ROOT_DIR  # 하위 호환 별칭


def sanitize_path_segment(value: Optional[str], fallback: str = "default") -> str:
    """경로 탈출(Path Traversal) 문자를 제거하고 안전한 단일 디렉터리 세그먼트를 반환합니다.
    '.' 이나 '..' 등 상대 경로 세그먼트는 허용하지 않고 즉시 fallback으로 대체합니다.
    """
    raw = (value or fallback).strip()
    safe = re.sub(r"[^a-zA-Z0-9_.\-@]", "_", raw)
    if not safe or safe in {".", ".."} or set(safe) == {"."}:
        return fallback
    return safe


@dataclass(frozen=True)
class SessionWorkspace:
    """특정 (user_id, session_id) 실행 컨텍스트의 표준 디렉터리 경로 모음."""

    user_id: str
    session_id: str
    root_session_id: str
    is_subagent: bool
    subagent_name: Optional[str]
    root_dir: Path

    @property
    def parent_session_id(self) -> Optional[str]:
        return self.root_session_id if self.is_subagent else None

    @property
    def plan_dir(self) -> Path:
        return self.root_dir / "plan"

    @property
    def logs_dir(self) -> Path:
        return self.root_dir / "logs"

    @property
    def scripts_dir(self) -> Path:
        return self.root_dir / "scripts"

    @property
    def outputs_dir(self) -> Path:
        return self.root_dir / "outputs"

    @property
    def subagents_dir(self) -> Path:
        """메인 세션 산하의 subagents 루트 디렉터리."""
        if self.is_subagent:
            return self.root_dir.parent
        return self.root_dir / "subagents"

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    @property
    def rel_root(self) -> str:
        return self._rel(self.root_dir)

    @property
    def rel_plan_dir(self) -> str:
        return self._rel(self.plan_dir)

    @property
    def rel_logs_dir(self) -> str:
        return self._rel(self.logs_dir)

    @property
    def rel_scripts_dir(self) -> str:
        return self._rel(self.scripts_dir)

    @property
    def rel_outputs_dir(self) -> str:
        return self._rel(self.outputs_dir)

    def ensure_dirs(self) -> "SessionWorkspace":
        """워크스페이스의 표준 하위 디렉터리(plan, logs, scripts, outputs)를 생성합니다."""
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        return self


def resolve_user_memory_dir(
    user_id: Optional[str] = "default",
    base_dir: Optional[Path] = None,
    ensure_exists: bool = False,
) -> Path:
    """사용자별 영구 장기 기억 디렉터리(`artifacts/users/{user_id}/memory`)를 반환합니다."""
    safe_uid = sanitize_path_segment(user_id, fallback="default")
    root = base_dir if base_dir is not None else USERS_ROOT_DIR
    mem_dir = root / safe_uid / "memory"
    if ensure_exists:
        mem_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("MEMORY.md", "USER.md"):
            fpath = mem_dir / fname
            if not fpath.exists():
                fpath.write_text("", encoding="utf-8")
    return mem_dir


def resolve_session_workspace(
    user_id: Optional[str] = "default",
    session_id: Optional[str] = "default",
    parent_session_id: Optional[str] = None,
    subagent_role: Optional[str] = None,
    base_dir: Optional[Path] = None,
    ensure_exists: bool = False,
) -> SessionWorkspace:
    """(user_id, session_id, parent_session_id)에 대응하는 메인 세션 또는 서브에이전트 워크스페이스를 반환합니다.

    - 메인 에이전트 (`parent_session_id` 미지정):
      `artifacts/users/{user_id}/sessions/{session_id}/`
    - 서브에이전트 (`parent_session_id` 명시):
      `artifacts/users/{user_id}/sessions/{parent_session_id}/subagents/{sub_folder}/`
    """
    safe_uid = sanitize_path_segment(user_id, fallback="default")
    safe_sid = sanitize_path_segment(session_id, fallback="default")

    is_sub = False
    root_sid = safe_sid
    sub_folder: Optional[str] = None

    if parent_session_id and str(parent_session_id).strip():
        safe_parent = sanitize_path_segment(parent_session_id, fallback="default")
        if safe_parent != safe_sid:
            is_sub = True
            root_sid = safe_parent
            prefix = f"{safe_parent}_"
            if safe_sid.startswith(prefix) and len(safe_sid) > len(prefix):
                sub_folder = safe_sid[len(prefix):]
            elif subagent_role and str(subagent_role).strip():
                safe_role = sanitize_path_segment(subagent_role, fallback="subagent")
                sub_folder = (
                    safe_sid
                    if safe_sid.startswith(f"{safe_role}_")
                    else f"{safe_role}_{safe_sid}"
                )
            else:
                sub_folder = safe_sid

    users_root = base_dir if base_dir is not None else USERS_ROOT_DIR
    main_session_dir = users_root / safe_uid / "sessions" / root_sid

    if is_sub and sub_folder:
        target_root = main_session_dir / "subagents" / sub_folder
    else:
        target_root = main_session_dir

    ws = SessionWorkspace(
        user_id=safe_uid,
        session_id=safe_sid,
        root_session_id=root_sid,
        is_subagent=is_sub,
        subagent_name=sub_folder,
        root_dir=target_root,
    )
    if ensure_exists:
        ws.ensure_dirs()
    return ws

