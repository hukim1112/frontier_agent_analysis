---
name: main-agent-skills
description: Skill discovery and execution policy for the main orchestrator agent. Skills live under app/agents/main_agent/skills/.
---

# Main Agent Skills Directory Guidelines

You are equipped with a progressive skill-disclosure architecture. Your skills live in `app/agents/main_agent/skills/` (one folder per skill, each with a `SKILL.md`).

## Guidelines for Discovering and Running Skills:
1. ALWAYS prioritize the skills listed in the catalog below over running raw shell commands directly in `bash_command`.
2. Before using a skill, read its `SKILL.md` (path shown in the catalog) with `file_read` to learn the exact scripts, parameters, and usage.
3. Execute the documented script or code via `bash_command`, using paths relative to the workspace root.
4. If the requested skill does not exist locally:
   - Run `npx -y skills add [owner/repo] --skill [name]` (or with `@` shorthand) via `bash_command` to download it.
   - If the skill is installed inside `.agents/skills/[name]`, copy it to `app/agents/main_agent/skills/[name]` using `cp -r` so it becomes part of your catalog.

