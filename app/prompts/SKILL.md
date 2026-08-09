# Public & Custom Skills Directory Guidelines

You are equipped with a progressive skill-disclosure architecture that supports using external script packages from the awesome-agent-skills ecosystem.

## Guidelines for Discovering and Running Skills:
1. ALWAYS prioritize searching and utilizing custom scripts under the local `skills/` folder over running raw shell commands directly in `bash_command`.
2. If the requested skill does not exist locally:
   - Run `npx -y skills add [owner/repo] --skill [name]` (or with `@` shorthand) via `bash_command` to download it.
   - If the skill is installed inside `.agents/skills/[name]`, copy it to `skills/[name]` using `cp -r` to keep it in the project path.
3. Once downloaded, read the corresponding `Skill.md` file under the downloaded folder using `file_read` to learn how to run the script and its arguments.
4. Execute the script via `bash_command` using the exact python path and parameters to fulfill the request.
