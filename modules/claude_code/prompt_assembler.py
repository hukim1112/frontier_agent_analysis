import os
import json
from typing import List, Dict, Any, Optional, Union, Callable
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents.middleware import dynamic_prompt, ModelRequest

class PromptAssembler:
    """Claude Code style 5-layer prompt assembler for Prompt Caching optimization.
    
    Layers:
    - Layer 1: Static Rules & Identity (PROMPT.md / System Rules)
    - Layer 2: Tool Capabilities (Alphabetically sorted for cache preservation)
    - Layer 3: Dynamic Boundary Marker (__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__)
    - Layer 4: Session Guidance & Memory Registry (CWD, OS, MEMORY.md, MCP.md, etc.)
    - Layer 5: User & Project Context (AGENT.md, CLAUDE.md, CONVENTIONS.md, Git Status)
    """
    def __init__(
        self,
        system_rules: str,
        tool_schemas: list,
        memory_path: Optional[str] = None,
        agent_rules_path: Optional[str] = None,
        l4_docs: Optional[Dict[str, Union[str, Callable]]] = None,
        l5_docs: Optional[Dict[str, Union[str, Callable]]] = None
    ):
        self.system_rules = system_rules
        self.tool_schemas = tool_schemas or []
        self.boundary_marker = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"

        # Multi-document registries for Layer 4 & Layer 5
        self.l4_docs: Dict[str, Union[str, Callable]] = dict(l4_docs or {})
        self.l5_docs: Dict[str, Union[str, Callable]] = dict(l5_docs or {})

        # Backward compatibility with single path parameters
        if memory_path:
            self.memory_path = memory_path
            self.l4_docs["MEMORY.md"] = memory_path
        else:
            self.memory_path = self.l4_docs.get("MEMORY.md")

        if agent_rules_path:
            self.agent_rules_path = agent_rules_path
            self.l5_docs["AGENT.md"] = agent_rules_path
        else:
            self.agent_rules_path = self.l5_docs.get("AGENT.md")

    def add_l4_doc(self, name: str, source: Union[str, Callable]):
        """Adds a dynamic document to Layer 4 (e.g. 'MCP.md', 'SCRATCHPAD.md')."""
        self.l4_docs[name] = source

    def add_l5_doc(self, name: str, source: Union[str, Callable]):
        """Adds a project context document to Layer 5 (e.g. 'CLAUDE.md', 'CONVENTIONS.md')."""
        self.l5_docs[name] = source

    def format_tool_capabilities(self) -> str:
        """Formats L2 tool capabilities sorted alphabetically by name for cache consistency."""
        if not self.tool_schemas:
            return "No registered tools."

        def get_tool_name(schema: Any) -> str:
            if isinstance(schema, dict):
                return schema.get("name", "")
            return getattr(schema, "name", str(schema))

        sorted_tools = sorted(self.tool_schemas, key=get_tool_name)
        
        tool_lines = []
        for idx, tool in enumerate(sorted_tools):
            name = get_tool_name(tool)
            desc = ""
            args_schema = None

            if isinstance(tool, dict):
                desc = tool.get("description", "")
                args_schema = tool.get("args") or tool.get("parameters")
            else:
                desc = getattr(tool, "description", "")
                if hasattr(tool, "args"):
                    args_schema = tool.args
                elif hasattr(tool, "args_schema") and tool.args_schema:
                    try:
                        args_schema = tool.args_schema.schema()
                    except Exception:
                        args_schema = str(tool.args_schema)

            tool_str = f"### [{idx+1}] {name}\nDescription: {desc}"
            if args_schema:
                schema_str = json.dumps(args_schema, ensure_ascii=False) if isinstance(args_schema, dict) else str(args_schema)
                tool_str += f"\nSchema: {schema_str}"
            tool_lines.append(tool_str)

        return "\n\n".join(tool_lines)

    def read_and_truncate_doc(
        self,
        source: Union[str, Callable],
        max_lines: int = 200,
        max_bytes: int = 25000
    ) -> str:
        """Reads content from path, callable, or text string with line & byte truncation."""
        if not source:
            return "Content not available."

        # Evaluate callable source
        if callable(source):
            try:
                raw_content = str(source())
            except Exception as e:
                return f"Error evaluating document source: {e}"
        # Check if source is a file path
        elif isinstance(source, str) and os.path.exists(source):
            try:
                with open(source, "r", encoding="utf-8") as f:
                    raw_content = f.read()
            except Exception as e:
                return f"Error reading file '{source}': {e}"
        elif isinstance(source, str):
            raw_content = source
        else:
            raw_content = str(source)

        lines = raw_content.splitlines(keepends=True)
        truncated = False

        if len(lines) > max_lines:
            lines = lines[:max_lines]
            truncated = True

        content = "".join(lines)
        encoded_bytes = content.encode("utf-8")
        if len(encoded_bytes) > max_bytes:
            content = encoded_bytes[:max_bytes].decode("utf-8", errors="ignore")
            truncated = True

        if truncated:
            content += "\n\n... [Content truncated due to size limits] ..."

        return content.strip()

    def build_static_content(self) -> str:
        """Assembles Layers 1~3 (Static identity, Tools, Policy, Boundary Marker)."""
        tool_str = self.format_tool_capabilities()
        return (
            f"=== Layer 1: Static System Rules ===\n{self.system_rules}\n\n"
            f"=== Layer 2: Tool Capabilities (Alphabetical) ===\n{tool_str}\n\n"
            f"=== Layer 3: Core Policy ===\n"
            f"- Verify environment before executing commands.\n"
            f"- Avoid introducing bugs, write clean and tested code.\n\n"
            f"{self.boundary_marker}"
        )

    def build_dynamic_content(self, session_context: dict) -> str:
        """Assembles Layers 4~5 (Dynamic Session/Memory Registry & User/Project Context)."""
        # --- Layer 4: Session Guidance & Memory Registry ---
        session_items = [
            f"- Working Directory (CWD): {session_context.get('cwd', '/workspace')}",
            f"- Session ID: {session_context.get('session_id', 'unknown')}",
            f"- Host OS: {session_context.get('os', os.name)}"
        ]
        if "user_permission" in session_context:
            session_items.append(f"- User Permission: {session_context['user_permission']}")
        if "active_project" in session_context:
            session_items.append(f"- Active Project: {session_context['active_project']}")

        l4_sections = [
            "=== Layer 4: Session Guidance & Memory (Registry) ===",
            "Session Information:\n" + "\n".join(session_items)
        ]

        if not self.l4_docs:
            l4_sections.append("MEMORY.md Content:\nNo persistent memory available.")
        else:
            for doc_name, source in self.l4_docs.items():
                doc_content = self.read_and_truncate_doc(source, max_lines=200, max_bytes=25000)
                l4_sections.append(f"{doc_name} Content:\n{doc_content}")

        # --- Layer 5: User & Project Context ---
        l5_sections = [
            "=== Layer 5: User & Project Context ==="
        ]

        if not self.l5_docs:
            l5_sections.append("Project Rules:\nNo project-specific AGENT.md context provided.")
        else:
            for doc_name, source in self.l5_docs.items():
                doc_content = self.read_and_truncate_doc(source, max_lines=500, max_bytes=50000)
                l5_sections.append(f"{doc_name} Rules ({doc_name}):\n{doc_content}")

        git_status = session_context.get("git_status", "Clean / Not tracked")
        l5_sections.append(f"Git Status / Workspace Diff:\n{git_status}")

        full_l4 = "\n\n".join(l4_sections)
        full_l5 = "\n\n".join(l5_sections)

        return f"{full_l4}\n\n{full_l5}"

    def build_system_prompt(self, session_context: dict) -> str:
        """Assembles all 5 layers into a single prompt string."""
        static_part = self.build_static_content()
        dynamic_part = self.build_dynamic_content(session_context)
        return f"{static_part}\n\n{dynamic_part}"

    def assemble(self, user_input: str, session_context: dict, chat_history: list = None) -> list:
        """Assembles full list of messages as structured SystemMessages + Chat History + HumanMessage."""
        static_content = self.build_static_content()
        dynamic_content = self.build_dynamic_content(session_context)

        assembled = [
            SystemMessage(content=static_content),
            SystemMessage(content=dynamic_content)
        ]

        if chat_history:
            assembled.extend(chat_history)

        assembled.append(HumanMessage(content=user_input))
        return assembled

from langchain.agents.middleware import wrap_model_call

def create_prompt_assembler_middleware(assembler: PromptAssembler):
    """Creates a production-grade custom middleware from PromptAssembler.
    
    Injects 2 SystemMessages into request.messages:
    - Message 0: Layer 1~3 Static Prompt with cache_control metadata (Ephemeral Caching Target)
    - Message 1: Layer 4~5 Dynamic Context (Session, Memory, Project Rules)
    """
    @wrap_model_call
    def claude_code_prompt_middleware(request: ModelRequest, handler):
        ctx = getattr(request.runtime, "context", None)
        session_context = {
            "cwd": getattr(ctx, "cwd", "/workspace") if ctx else "/workspace",
            "session_id": getattr(ctx, "session_id", "unknown") if ctx else "unknown"
        }

        static_msg = SystemMessage(
            content=assembler.build_static_content(),
            additional_kwargs={"cache_control": {"type": "ephemeral"}}
        )
        dynamic_msg = SystemMessage(
            content=assembler.build_dynamic_content(session_context)
        )

        # Filter out existing SystemMessages to avoid duplication
        filtered_msgs = [m for m in request.messages if not isinstance(m, SystemMessage)]
        new_messages = [static_msg, dynamic_msg] + filtered_msgs

        new_request = request.override(messages=new_messages)
        return handler(new_request)

    return claude_code_prompt_middleware



