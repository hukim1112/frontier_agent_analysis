import os
import json
from typing import List, Dict, Any

class PromptManager:
    """Manager for loading and assembling prompt layers from prompt_dir templates."""
    def __init__(self, prompt_dir=None):
        if prompt_dir is None:
            prompt_dir = os.path.dirname(os.path.abspath(__file__))
        self.prompt_dir = prompt_dir
        self.boundary_marker = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"
        
        # L1: Static System Role (PROMPT.md)
        self.l1_role = self._load_file("PROMPT.md", "You are a professional software engineering agent.")
        
        # L2: Static Agent Policy & Guidelines
        self.l2_guidelines = "Verify environment before executing commands.\nAvoid introducing bugs, write clean and tested code."
        
        # Tools Specifications (L2 Alphabetical)
        self.l2_tools = "No registered tools."
        
        # Skills context (Loaded from SKILL.md)
        self.skills_context = self._load_file("SKILL.md", "No public skills registered.")
        
        # Static Reference Context (Stored above boundary to enable caching)
        self.static_reference = ""
        
        # L5: User Context & Project Rules (AGENT.md)
        self.l5_agent_rules = self._load_file("AGENT.md", "## Core Policy:\n1. Prioritize security boundaries.\n2. Do not mutate state without pre-approval.")
        self.l5_dynamic_context = "No dynamic context provided."

    def _load_file(self, filename: str, fallback: str) -> str:
        filepath = os.path.join(self.prompt_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read().strip()
        return fallback

    def set_reference_context(self, context_text: str):
        """Sets the large static reference manual (caching target) above the boundary."""
        self.static_reference = context_text

    def set_dynamic_context(self, context_text: str):
        """Sets the dynamic context (L5) below the boundary."""
        self.l5_dynamic_context = context_text

    def build_tool_specifications(self, tools: List[Any]):
        """Dynamically build L2 tool specifications sorted alphabetically by name."""
        if not tools:
            self.l2_tools = "No registered tools."
            return

        def get_tool_name(t):
            if isinstance(t, dict):
                return t.get("name", "")
            return getattr(t, "name", str(t))

        sorted_tools = sorted(tools, key=get_tool_name)
        spec_lines = []
        for idx, t in enumerate(sorted_tools):
            name = get_tool_name(t)
            desc = t.get("description", "") if isinstance(t, dict) else getattr(t, "description", "")
            spec_lines.append(f"### [Tool {idx+1}] name: {name}")
            spec_lines.append(f"  - description: {desc}")
            args_schema = t.get("args") if isinstance(t, dict) else getattr(t, "args", None)
            if args_schema:
                spec_lines.append(f"  - arguments_schema: {json.dumps(args_schema, ensure_ascii=False)}")
            spec_lines.append("")
        self.l2_tools = "\n".join(spec_lines).strip()

    def build_system_prompt(self, dynamic_state: dict) -> str:
        """
        Assembles 5-layer prompt stack with boundary marker.
        L1, L2 (Tools & Skills) & Static Reference -> ABOVE boundary (Cached).
        L4 (Dynamic env/session) & L5 (AGENT.md & User Context) -> BELOW boundary (Uncached).
        """
        permission_string = f"Current User Permissions: {dynamic_state.get('user_permission', 'NONE')}"
        project_string = f"Active Target Project: {dynamic_state.get('active_project', 'NONE')}"
        cwd_string = f"CWD: {dynamic_state.get('cwd', '/workspace')}"
        l4_env = f"- {permission_string}\n- {project_string}\n- {cwd_string}"
        
        static_part = (
            f"=== ROLE (L1) ===\n{self.l1_role}\n\n"
            f"=== OPERATING GUIDELINES ===\n{self.l2_guidelines}\n\n"
            f"=== PUBLIC SKILLS CATALOG ===\n{self.skills_context}\n\n"
            f"=== TOOLS SPECS (L2) ===\n{self.l2_tools}\n\n"
            f"=== STATIC REFERENCE CONTEXT ===\n{self.static_reference}\n\n"
            f"{self.boundary_marker}"
        )
        
        dynamic_part = (
            f"=== DYNAMIC RULES/ENV (L4) ===\n{l4_env}\n\n"
            f"=== USER & PROJECT CONTEXT / AGENT.md (L5) ===\n{self.l5_agent_rules}\n\n"
            f"=== DYNAMIC CONTEXT ===\n{self.l5_dynamic_context}"
        )
        
        return f"{static_part}\n\n{dynamic_part}"


