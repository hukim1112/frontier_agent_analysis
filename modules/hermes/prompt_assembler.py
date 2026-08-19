"""
범용 5-Layer PromptAssembler — Claude Code 규격을 준수하는 범용 프롬프트 조립기.

Hermes 메모리 구조와 통합:
- Layer 1: Static Identity & Rules (PROMPT.md)                ← 캐싱 대상 (Static Prefix)
- Layer 2: Tool Specs + Skill Index (알파벳 정렬)              ← 캐싱 대상 (Static Prefix)
  ────── __SYSTEM_PROMPT_DYNAMIC_BOUNDARY__ ──────
- Layer 3: Dynamic Session Environment (CWD, Session ID, OS)  ← 매 턴 변경 (Uncached Suffix)
- Layer 4: Dynamic Session Documents (MCP.md) + Recalled Memory
- Layer 5: User & Project Rules (AGENT.md)

recalled_memory 필드를 통해 MemoryMiddleware가 인출한 L2(에피소드)/L3(시멘틱)
메모리를 시스템 프롬프트에 자연스럽게 주입합니다.
"""

import os
import json
from typing import List, Dict, Any, Optional, Union, Callable

from langchain_core.messages import SystemMessage
from langchain.agents.middleware import wrap_model_call, ModelRequest


class PromptAssembler:
    """범용 5-Layer 프롬프트 조립기.

    Claude Code의 PromptAssembler에서 공급자 특화 로직을 제거하고,
    Hermes 메모리 주입을 위한 recalled_memory 레이어를 추가.
    """

    def __init__(
        self,
        system_rules: str,
        tool_schemas: Optional[list] = None,
        skill_catalog: Optional[Union[str, Callable]] = None,
        l4_docs: Optional[Dict[str, Union[str, Callable]]] = None,
    ):
        self.system_rules = system_rules
        self.tool_schemas = tool_schemas or []
        self.skill_catalog = skill_catalog
        self.boundary_marker = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"

        # Layer 4: Dynamic Session/MCP documents (e.g. 'MCP.md')
        self.l4_docs: Dict[str, Union[str, Callable]] = dict(l4_docs or {})
        # Layer 5: User/Project context rules (e.g. 'AGENT.md')
        self.l5_docs: Dict[str, Union[str, Callable]] = {}

    def set_skill_catalog(self, source: Union[str, Callable]) -> None:
        """Layer 2에 스킬 카탈로그/가이드라인을 설정 (e.g., SkillPromptBuilder.assemble)."""
        self.skill_catalog = source

    def add_l4_doc(self, name: str, source: Union[str, Callable]) -> None:
        """Layer 4에 동적 문서를 추가 (e.g., 'MCP.md', 'SCRATCHPAD.md')."""
        self.l4_docs[name] = source

    def add_l5_doc(self, name: str, source: Union[str, Callable]) -> None:
        """Layer 5에 프로젝트 컨텍스트 규칙 문서를 추가 (e.g., 'AGENT.md')."""
        self.l5_docs[name] = source

    # ── Layer Builders ──

    def build_static_content(self) -> str:
        """Layer 1~2 + Boundary Marker 조립 (캐싱 대상)."""
        tool_str = self._format_tool_capabilities()
        skill_str = ""
        if self.skill_catalog:
            raw_skill = self._read_and_truncate_doc(
                self.skill_catalog, max_lines=300, max_bytes=35000
            )
            skill_str = f"\n\n=== Layer 2.1: Available Skills Catalog ===\n{raw_skill}"

        return (
            f"=== Layer 1: System Identity & Rules ===\n{self.system_rules}\n\n"
            f"=== Layer 2: Tool Capabilities (Alphabetical) ===\n{tool_str}"
            f"{skill_str}\n\n"
            f"{self.boundary_marker}"
        )

    def build_dynamic_content(self, session_context: dict) -> str:
        """Layer 3~5 조립 (매 턴 변경).

        session_context keys:
        - cwd: 작업 디렉토리
        - session_id: 세션 식별자
        - os: 운영체제
        - recalled_memory: L2+L3 메모리 인출 결과 (MemoryMiddleware가 주입)
        - user_permission, active_project: 가드레일 정보
        """
        # ── Layer 3: Dynamic Session Context ──
        session_items = [
            f"- Working Directory: {session_context.get('cwd', '/workspace')}",
            f"- Session ID: {session_context.get('session_id', 'unknown')}",
            f"- Host OS: {session_context.get('os', os.name)}",
        ]
        if "user_permission" in session_context:
            session_items.append(f"- User Permission: {session_context['user_permission']}")
        if "active_project" in session_context:
            session_items.append(f"- Active Project: {session_context['active_project']}")

        l3_sections = [
            "=== Layer 3: Dynamic Session Context ===",
            "Session Information:\n" + "\n".join(session_items),
        ]

        # Recalled Memory 주입 (L2 에피소드 + L3 시멘틱)
        recalled = session_context.get("recalled_memory", "")
        if recalled and recalled.strip():
            l3_sections.append(f"Recalled Memory:\n{recalled}")

        # ── Layer 4: Dynamic Session Documents (MCP.md 등) ──
        l4_sections = ["=== Layer 4: Dynamic Session Documents ==="]
        if not self.l4_docs:
            l4_sections.append("No dynamic session documents provided.")
        else:
            for doc_name, source in self.l4_docs.items():
                doc_content = self._read_and_truncate_doc(
                    source, max_lines=500, max_bytes=50000
                )
                l4_sections.append(f"{doc_name}:\n{doc_content}")

        # ── Layer 5: User & Project Rules (AGENT.md, SKILL.md 등) ──
        l5_sections = ["=== Layer 5: User & Project Rules ==="]
        if not self.l5_docs:
            l5_sections.append("No project-specific rules provided.")
        else:
            for doc_name, source in self.l5_docs.items():
                doc_content = self._read_and_truncate_doc(
                    source, max_lines=500, max_bytes=50000
                )
                l5_sections.append(f"{doc_name}:\n{doc_content}")

        full_l3 = "\n\n".join(l3_sections)
        full_l4 = "\n\n".join(l4_sections)
        full_l5 = "\n\n".join(l5_sections)
        return f"{full_l3}\n\n{full_l4}\n\n{full_l5}"

    def build_system_prompt(self, session_context: dict) -> str:
        """전체 4-Layer 프롬프트 문자열 조립."""
        static_part = self.build_static_content()
        dynamic_part = self.build_dynamic_content(session_context)
        return f"{static_part}\n\n{dynamic_part}"

    # ── Internal Helpers ──

    def _format_tool_capabilities(self) -> str:
        """Layer 2: 도구 스펙을 알파벳 순으로 정렬하여 포맷."""
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

            tool_str = f"### [{idx + 1}] {name}\nDescription: {desc}"
            if args_schema:
                schema_str = (
                    json.dumps(args_schema, ensure_ascii=False)
                    if isinstance(args_schema, dict)
                    else str(args_schema)
                )
                tool_str += f"\nSchema: {schema_str}"
            tool_lines.append(tool_str)

        return "\n\n".join(tool_lines)

    @staticmethod
    def _read_and_truncate_doc(
        source: Union[str, Callable],
        max_lines: int = 200,
        max_bytes: int = 25000,
    ) -> str:
        """파일 경로, callable, 문자열 소스에서 내용을 읽고 truncate."""
        if not source:
            return "Content not available."

        # Callable 평가
        if callable(source):
            try:
                raw_content = str(source())
            except Exception as e:
                return f"Error evaluating source: {e}"
        # 파일 경로
        elif isinstance(source, str) and os.path.exists(source):
            try:
                with open(source, "r", encoding="utf-8") as f:
                    raw_content = f.read()
            except Exception as e:
                return f"Error reading file '{source}': {e}"
        # 문자열 그대로
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
        encoded = content.encode("utf-8")
        if len(encoded) > max_bytes:
            content = encoded[:max_bytes].decode("utf-8", errors="ignore")
            truncated = True

        if truncated:
            content += "\n\n... [Content truncated due to size limits] ..."

        return content.strip()


def create_prompt_middleware(assembler: PromptAssembler):
    """PromptAssembler 기반 프롬프트 주입 미들웨어.

    AgentContext.recalled_memory를 읽어 session_context에 포함시키고,
    SystemMessage 2개(static + dynamic)를 메시지 앞에 삽입합니다.
    """

    @wrap_model_call
    async def prompt_middleware(request: ModelRequest, handler):
        ctx = getattr(request.runtime, "context", None)
        session_context = {
            "cwd": getattr(ctx, "cwd", "/workspace") if ctx else "/workspace",
            "session_id": "unknown",
            "recalled_memory": getattr(ctx, "recalled_memory", "") if ctx else "",
            "user_permission": getattr(ctx, "user_permission", "GUEST") if ctx else "GUEST",
            "active_project": getattr(ctx, "active_project", "UNKNOWN") if ctx else "UNKNOWN",
        }

        # thread_id from config
        config = getattr(request.runtime, "config", None)
        if config:
            session_context["session_id"] = (
                config.get("configurable", {}).get("thread_id", "unknown")
            )

        static_msg = SystemMessage(content=assembler.build_static_content())
        dynamic_msg = SystemMessage(content=assembler.build_dynamic_content(session_context))

        # 기존 SystemMessage 제거 및 빈 메시지 정제
        filtered_msgs = []
        for m in request.messages:
            if isinstance(m, SystemMessage):
                continue
            # Gemini Vertex AI 400 Error (empty parts) 방지: content나 tool_calls가 없는 비어있는 메시지 제거
            content = getattr(m, "content", "")
            tool_calls = getattr(m, "tool_calls", None)
            if not content and not tool_calls:
                continue
            filtered_msgs.append(m)

        new_messages = [static_msg, dynamic_msg] + filtered_msgs
        new_request = request.override(messages=new_messages)
        return await handler(new_request)

    return prompt_middleware
