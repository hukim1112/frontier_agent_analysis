"""
===============================================================================
[H-02] Amnesia Guard Middleware
===============================================================================
Source: frontier-agent-lab/modules/claude_code/amnesia_guard.py

컨텍스트 컴팩션 후 최근 작업 컨텍스트(파일, 계획) 소실을 방지하는 복원 미들웨어.
- 최근 접근한 파일 경로 추적 (LRU, max_restore_files 제한)
- 활성 계획(Active Plan) 텍스트 보존 (str 및 list[str] 타입 모두 지원)
- 컴팩션 시 create_recovery_attachments()로 SystemMessage 복원 블록 생성
- 파일당 max_file_chars 제한 + Head/Tail 트리밍으로 복원 블록 크기 폭발 방지

H-01 (Compactor)의 AutoCompactor/ReactiveCompactor와 쌍으로 동작합니다.
도구 호출 인터셉터(create_amnesia_guard_middleware)를 통해 자동 추적합니다.
===============================================================================
"""

import os
from langchain_core.messages import SystemMessage
from langchain.agents.middleware import wrap_tool_call, AgentMiddleware


class AmnesiaGuardMiddleware(AgentMiddleware):
    """Tracks recently accessed files and active plans to restore them after context compaction.
    Limits per-file content to max_file_chars with Head/Tail trimming to prevent token explosion."""

    def __init__(self, max_restore_files: int = 5, max_file_chars: int = 3000):
        self.max_restore_files = max_restore_files
        self.max_file_chars = max_file_chars
        self.recent_files = []  # List of file paths
        self.active_plan = None  # Active plan text

    def track_file_access(self, file_path: str):
        if not file_path:
            return
        
        # Normalize path
        normalized = os.path.normpath(file_path)
        
        # Remove if already exists to move it to the end (most recent)
        if normalized in self.recent_files:
            self.recent_files.remove(normalized)
            
        self.recent_files.append(normalized)
        
        # Keep only the last max_restore_files
        if len(self.recent_files) > self.max_restore_files:
            self.recent_files.pop(0)

    def set_active_plan(self, plan):
        """Set active plan. Accepts both str and list[str] types."""
        if isinstance(plan, list):
            self.active_plan = "\n".join(str(item) for item in plan)
        elif isinstance(plan, str):
            self.active_plan = plan

    def _trim_file_content(self, content: str, file_path: str = "") -> str:
        """Trim large file content to max_file_chars using Head(40 lines)/Tail(10 lines) strategy.
        Includes Actionable Hint for full content retrieval."""
        if len(content) <= self.max_file_chars:
            return content

        lines = content.split("\n")
        head_lines = 40
        tail_lines = 10

        if len(lines) <= head_lines + tail_lines:
            # File has few lines but large content per line — truncate by chars
            truncated = content[:self.max_file_chars]
            hint = f"\n... [{len(content) - self.max_file_chars} chars omitted. Use `read_file(path='{file_path}')` for full content]"
            return truncated + hint

        head = "\n".join(lines[:head_lines])
        tail = "\n".join(lines[-tail_lines:])
        omitted_count = len(lines) - head_lines - tail_lines

        trimmed = (
            f"{head}\n"
            f"\n... [{omitted_count} lines omitted. Use `read_file(path='{file_path}')` for full content] ...\n\n"
            f"{tail}"
        )

        # Final safety check against max_file_chars
        if len(trimmed) > self.max_file_chars:
            trimmed = trimmed[:self.max_file_chars] + f"\n... [Trimmed to {self.max_file_chars} chars]"

        return trimmed

    def create_recovery_attachments(self) -> list:
        recovery_sections = []

        if self.active_plan:
            recovery_sections.append(f"=== Active Plan ===\n{self.active_plan}")

        if self.recent_files:
            file_snapshots = []
            for f in self.recent_files:
                if os.path.exists(f) and os.path.isfile(f):
                    try:
                        with open(f, "r", encoding="utf-8") as file:
                            content = file.read()
                        trimmed_content = self._trim_file_content(content, file_path=f)
                        file_snapshots.append(f"File: {f}\nContent:\n{trimmed_content}")
                    except Exception as e:
                        file_snapshots.append(f"File: {f}\nContent: (Error reading: {e})")
                else:
                    file_snapshots.append(f"File: {f}\nContent: (File does not exist on disk)")
            
            recovery_sections.append("=== Recently Modified/Accessed Files ===\n" + "\n\n".join(file_snapshots))

        if not recovery_sections:
            return []

        attachment_text = (
            "[Compaction Amnesia Guard: Restoring Recent Work Context]\n"
            "The following context has been restored from your recent actions:\n\n"
            + "\n\n".join(recovery_sections)
        )
        return [SystemMessage(content=attachment_text)]


def create_amnesia_guard_middleware(amnesia_guard: AmnesiaGuardMiddleware):
    """Creates a LangChain @wrap_tool_call middleware that automatically intercepts tool calls to track file and plan access."""
    @wrap_tool_call
    def amnesia_tool_interceptor(request, handler):
        tool_name = request.tool_call.get("name", "")
        args = request.tool_call.get("args", {})
        
        for key in ["file_path", "path", "filename", "file", "notebook_path"]:
            if key in args and isinstance(args[key], str):
                amnesia_guard.track_file_access(args[key])

        if tool_name in ["update_plan", "create_plan", "set_plan", "enter_plan"]:
            for key in ["plan", "content", "text", "steps"]:
                if key in args:
                    val = args[key]
                    if isinstance(val, (str, list)):
                        amnesia_guard.set_active_plan(val)

        return handler(request)

    return amnesia_tool_interceptor
