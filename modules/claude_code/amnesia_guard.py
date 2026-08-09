import os
from langchain_core.messages import SystemMessage
from langchain.agents.middleware import wrap_tool_call, AgentMiddleware

class AmnesiaGuardMiddleware(AgentMiddleware):
    """Tracks recently accessed files and active plans to restore them after context compaction."""
    def __init__(self, max_restore_files: int = 5):
        self.max_restore_files = max_restore_files
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

    def set_active_plan(self, plan: str):
        self.active_plan = plan

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
                        file_snapshots.append(f"File: {f}\nContent:\n{content}")
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

        if tool_name in ["update_plan", "create_plan", "set_plan"]:
            for key in ["plan", "content", "text"]:
                if key in args and isinstance(args[key], str):
                    amnesia_guard.set_active_plan(args[key])

        return handler(request)

    return amnesia_tool_interceptor

