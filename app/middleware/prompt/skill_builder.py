"""
===============================================================================
Skill Prompt Builder — Frontmatter Scanner & Dynamic Catalog Builder
===============================================================================
Source: frontier-agent-lab/app/prompts/skill_builder.py

스킬 행동 수칙(가이드라인) + 설치된 스킬 디렉토리(SKILL.md)의 YAML Frontmatter를
스캔하여 <skills> 카탈로그를 동적으로 생성합니다.
===============================================================================
"""

import os
import re
from typing import List, Optional, Dict, Any


class SkillPromptBuilder:
    """스킬 행동 수칙(가이드라인) + 설치된 스킬 카탈로그(프론트매터) 통합 조립기"""

    DEFAULT_GUIDELINES = """## 🛠️ Progressive Skill Execution Policy
1. 사용자의 요청을 해결할 수 있는 스킬이 아래 카탈로그에 있다면, raw 셸 명령보다 해당 스킬을 우선 활용하세요.
2. 스킬을 사용하기 전, 반드시 `file_read`로 해당 스킬의 `SKILL.md`를 열람하여 정확한 파라미터와 스크립트 실행법을 파악하세요.
3. 지침에 따라 `bash_command`로 스크립트를 실행하여 작업을 완료하세요."""

    def __init__(
        self,
        skills_dirs: Optional[List[str]] = None,
        guidelines_path: Optional[str] = "app/prompts/SKILL.md",
    ):
        self.skills_dirs = skills_dirs or ["./skills", "./.agents/skills", "skills", "../skills"]
        self.guidelines_path = guidelines_path

    def _extract_frontmatter(self, skill_md_path: str) -> Dict[str, str]:
        """SKILL.md 상단의 YAML Frontmatter(name, description) 추출 (없으면 제목/설명 폴백)"""
        try:
            with open(skill_md_path, "r", encoding="utf-8") as f:
                content = f.read(2048)  # 상단 2KB 읽기 (I/O 최적화)

            # 1. --- ... --- 사이의 YAML Frontmatter 정규식 추출
            match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
            if match:
                meta = {}
                for line in match.group(1).strip().splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        meta[k.strip().lower()] = v.strip().strip("\"'")
                if meta.get("name") or meta.get("description"):
                    return meta

            # 2. Frontmatter가 없을 때의 마크다운 폴백 (# Title 및 첫 문단)
            meta = {}
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            for idx, line in enumerate(lines):
                if line.startswith("# ") and "name" not in meta:
                    meta["name"] = line.replace("# ", "").replace("Skill", "").strip()
                elif not line.startswith("#") and not line.startswith("---") and "description" not in meta:
                    meta["description"] = line

            return meta
        except Exception:
            return {}

    def build_catalog(self) -> str:
        """설치된 모든 스킬 디렉토리를 스캔하여 <skills> 카탈로그 텍스트 생성"""
        catalog_entries = []
        seen_dirs = set()

        for base_dir in self.skills_dirs:
            if not os.path.exists(base_dir):
                continue

            try:
                entries = sorted(os.listdir(base_dir))
            except Exception:
                continue

            for folder_name in entries:
                dir_path = os.path.join(base_dir, folder_name)
                if not os.path.isdir(dir_path):
                    continue

                dir_key = os.path.normcase(os.path.abspath(dir_path))
                if dir_key in seen_dirs:
                    continue

                found_skill_file = None
                for fname in ["SKILL.md", "Skill.md", "skill.md"]:
                    candidate = os.path.join(dir_path, fname)
                    if os.path.exists(candidate):
                        found_skill_file = candidate
                        break

                if not found_skill_file:
                    continue

                seen_dirs.add(dir_key)
                meta = self._extract_frontmatter(found_skill_file)

                name = meta.get("name", folder_name)
                desc = meta.get("description", "No description provided.")
                rel_path = os.path.relpath(found_skill_file).replace("\\", "/")

                catalog_entries.append(f"- **{name}** (`{rel_path}`):\n    {desc}")

        if not catalog_entries:
            return "No custom skills currently registered in local `skills/` directory."

        return "<skills>\n" + "\n".join(catalog_entries) + "\n</skills>"

    def assemble(self) -> str:
        """가이드라인 + 카탈로그를 하나로 조립하여 Layer 2 주입용 문자열 반환"""
        guidelines = self.DEFAULT_GUIDELINES
        if self.guidelines_path and os.path.exists(self.guidelines_path):
            try:
                with open(self.guidelines_path, "r", encoding="utf-8") as f:
                    guidelines = f.read().strip()
            except Exception:
                pass

        catalog = self.build_catalog()
        return f"{guidelines}\n\n### 📦 Available Skills Catalog (Indexed from Frontmatter)\n{catalog}"
