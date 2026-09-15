"""
===============================================================================
Advanced Tests for Claude Code 16-Module Prompt Assembler
===============================================================================
Validates: 16-module factory, alphabetical tool sort, User Context Bypass,
           Git snapshot freezing, MCP Delta, Scratchpad, Middleware extensions.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Import from APP-level (production) prompt_assembler
from app.middleware.prompt.prompt_assembler import (
    PromptAssembler,
    PromptAssemblerMiddleware,
    create_prompt_assembler_middleware,
    # Module factories
    get_simple_intro_section,
    get_simple_system_section,
    get_simple_doing_tasks_section,
    get_actions_section,
    get_using_your_tools_section,
    get_simple_tone_and_style_section,
    get_output_efficiency_section,
    get_session_guidance_section,
    get_memory_index_section,
    get_env_info_section,
    get_language_section,
    get_output_style_section,
    get_mcp_instructions_section,
    get_scratchpad_section,
    get_frc_section,
    get_summarize_tool_results_section,
    # Bypass & Delta
    get_frozen_git_snapshot,
    prepend_user_context,
    attach_mcp_delta_if_needed,
    create_session_scratchpad,
    # Constants
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    MAX_STATUS_CHARS,
)


# =============================================================================
# Test: 16-Module Factory Functions
# =============================================================================

class TestModuleFactories:
    """각 모듈 팩토리가 비어있지 않은 문자열을 반환하는지 검증."""

    def test_module_0_intro(self):
        result = get_simple_intro_section()
        assert "interactive" in result.lower() or "agent" in result.lower()
        assert "URL" in result

    def test_module_1_system(self):
        result = get_simple_system_section()
        assert "Markdown" in result
        assert "deny" in result.lower() or "denial" in result.lower()

    def test_module_2_doing_tasks(self):
        result = get_simple_doing_tasks_section()
        assert "refactor" in result.lower() or "IMPORTANT" in result
        assert "false" in result.lower() or "done" in result.lower()

    def test_module_3_actions(self):
        result = get_actions_section()
        assert "destructive" in result.lower() or "rm -rf" in result
        assert "confirmation" in result.lower()

    def test_module_4_tools(self):
        result = get_using_your_tools_section()
        assert "cat" in result or "FileRead" in result
        assert "PARALLEL" in result or "parallel" in result

    def test_module_5_tone_style(self):
        result = get_simple_tone_and_style_section()
        assert "emoji" in result.lower()

    def test_module_6_output_efficiency(self):
        result = get_output_efficiency_section()
        assert "greetings" in result.lower() or "recap" in result.lower()
        assert "concise" in result.lower()

    def test_module_8_session_guidance(self):
        result = get_session_guidance_section()
        assert "!" in result or "slash" in result.lower()

    def test_module_9_memory_index_with_content(self):
        result = get_memory_index_section("Some memory content here")
        assert "MEMORY.md" in result
        assert "Some memory content here" in result

    def test_module_9_memory_index_empty(self):
        result = get_memory_index_section("")
        assert result == ""

    def test_module_10_env_info(self):
        result = get_env_info_section(
            cwd="/workspace", host_os="Linux", session_id="test-001",
            model_id="claude-4-sonnet", current_date="2026-09-15"
        )
        assert "/workspace" in result
        assert "Linux" in result
        assert "test-001" in result
        assert "claude-4-sonnet" in result

    def test_module_11_language(self):
        result = get_language_section("Korean")
        assert "Korean" in result
        assert "technical terms" in result.lower() or "Preserve" in result

    def test_module_11_language_empty(self):
        assert get_language_section("") == ""

    def test_module_12_output_style(self):
        result = get_output_style_section("Concise")
        assert "Concise" in result

    def test_module_12_output_style_empty(self):
        assert get_output_style_section("") == ""

    def test_module_13_mcp_instructions(self):
        result = get_mcp_instructions_section("Use PostgreSQL tools for DB ops.")
        assert "PostgreSQL" in result

    def test_module_13_mcp_delta_enabled_returns_empty(self):
        result = get_mcp_instructions_section("Use tools.", mcp_delta_enabled=True)
        assert result == ""

    def test_module_14_scratchpad(self):
        result = get_scratchpad_section("/tmp/agent_scratch_abc")
        assert "/tmp/agent_scratch_abc" in result
        assert "sandbox" in result.lower() or "Sandbox" in result

    def test_module_14_scratchpad_empty(self):
        assert get_scratchpad_section("") == ""

    def test_module_15_frc(self):
        result = get_frc_section()
        assert "cleared" in result.lower() or "context" in result.lower()

    def test_module_16_summarize(self):
        result = get_summarize_tool_results_section()
        assert "key findings" in result.lower() or "persist" in result.lower()


# =============================================================================
# Test: PromptAssembler with claude_code_modules=True
# =============================================================================

class TestPromptAssemblerModules:
    """claude_code_modules=True 시 16대 모듈 조립 결과 검증."""

    def test_static_content_includes_all_l1_modules(self):
        assembler = PromptAssembler(
            system_rules="Custom project rules.",
            tool_schemas=[{"name": "read_file", "description": "Reads a file"}],
            claude_code_modules=True,
        )
        static = assembler.build_static_content()

        # L1 모듈들이 포함되었는지 확인
        assert "Global Constitution" in static
        assert "interactive" in static.lower() or "agent" in static.lower()  # Module 0
        assert "IMPORTANT" in static  # Module 2 or 3
        assert "Custom project rules." in static  # 사용자 규칙 합성

        # L2 도구
        assert "read_file" in static
        assert "Anti-Raw Bash" in static or "FileRead" in static  # Module 4

        # 경계선
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in static

    def test_dynamic_content_includes_l3_l4_l5_modules(self):
        assembler = PromptAssembler(
            system_rules="Rules",
            tool_schemas=[],
            claude_code_modules=True,
            language="Korean",
            output_style="Concise",
        )
        session_ctx = {
            "cwd": "/project",
            "session_id": "sess-001",
            "os": "linux",
            "recalled_memory": "Previous task: implement auth module",
        }
        dynamic = assembler.build_dynamic_content(session_ctx)

        # L3: Session Guidance
        assert "Session Interaction" in dynamic

        # L4: Environment + Memory
        assert "/project" in dynamic
        assert "sess-001" in dynamic
        assert "Korean" in dynamic
        assert "Concise" in dynamic
        assert "Previous task: implement auth module" in dynamic

        # L5: FRC + Summarize
        assert "cleared" in dynamic.lower() or "context" in dynamic.lower()
        assert "key findings" in dynamic.lower() or "persist" in dynamic.lower()

    def test_full_assembly_structure(self):
        """assemble()가 정확히 3개 메시지(SystemMessage x2 + HumanMessage)를 반환하는지 검증."""
        assembler = PromptAssembler(
            system_rules="Test rules",
            tool_schemas=[{"name": "z_tool", "description": "Z"}, {"name": "a_tool", "description": "A"}],
            claude_code_modules=True,
        )
        messages = assembler.assemble(
            user_input="Hello",
            session_context={"cwd": "/workspace", "session_id": "test"}
        )
        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)  # Static (L1+L2+Boundary)
        assert isinstance(messages[1], SystemMessage)  # Dynamic (L3+L4+L5)
        assert isinstance(messages[2], HumanMessage)
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in messages[0].content


# =============================================================================
# Test: Alphabetical Tool Sort
# =============================================================================

class TestAlphabeticalToolSort:
    """도구 스키마가 알파벳 사전순으로 정렬되는지 검증."""

    def test_tool_sort_order(self):
        tools = [
            {"name": "z_write", "description": "Writes"},
            {"name": "a_read", "description": "Reads"},
            {"name": "m_search", "description": "Searches"},
        ]
        assembler = PromptAssembler(system_rules="", tool_schemas=tools)
        formatted = assembler.format_tool_capabilities()

        # a_read가 z_write보다 먼저 나와야 함
        idx_a = formatted.index("a_read")
        idx_m = formatted.index("m_search")
        idx_z = formatted.index("z_write")
        assert idx_a < idx_m < idx_z

    def test_tool_sort_determinism(self):
        """동일한 도구 집합은 항상 동일한 출력을 생성해야 함 (캐시 일관성)."""
        tools = [
            {"name": "c_tool", "description": "C"},
            {"name": "a_tool", "description": "A"},
            {"name": "b_tool", "description": "B"},
        ]
        assembler1 = PromptAssembler(system_rules="", tool_schemas=tools)
        assembler2 = PromptAssembler(system_rules="", tool_schemas=list(reversed(tools)))

        assert assembler1.format_tool_capabilities() == assembler2.format_tool_capabilities()


# =============================================================================
# Test: User Context Bypass Injection
# =============================================================================

class TestUserContextBypass:
    """prepend_user_context가 messages[0]에 <system-reminder>를 정확히 1회 주입하는지 검증."""

    def test_injection_adds_system_reminder(self):
        messages = [
            HumanMessage(content="What is this project?"),
            AIMessage(content="Let me check."),
        ]
        result = prepend_user_context(
            messages,
            claude_md_content="Always use pytest for testing.",
            git_snapshot="main: (clean)\nRecent commits: abc1234",
        )
        assert len(result) == 2
        assert "<system-reminder>" in result[0].content
        assert "Always use pytest for testing." in result[0].content
        assert "abc1234" in result[0].content
        assert "What is this project?" in result[0].content

    def test_injection_is_idempotent(self):
        """이미 주입된 경우 재주입하지 않아야 함."""
        messages = [
            HumanMessage(content="<system-reminder>\nAlready injected.\n</system-reminder>\n\nHello"),
        ]
        result = prepend_user_context(
            messages,
            claude_md_content="New content",
            git_snapshot="new snapshot",
        )
        # 이미 <system-reminder>가 있으므로 변경 없어야 함
        assert result[0].content == messages[0].content

    def test_injection_skipped_when_no_content(self):
        """주입할 내용이 없으면 원본 메시지 그대로 반환."""
        messages = [HumanMessage(content="Hello")]
        result = prepend_user_context(messages, claude_md_content="", git_snapshot="")
        assert result[0].content == "Hello"

    def test_injection_preserves_non_human_messages(self):
        """SystemMessage 뒤에 HumanMessage가 있는 경우 올바르게 처리."""
        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="User query"),
        ]
        result = prepend_user_context(
            messages,
            claude_md_content="Rules",
        )
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "System"
        assert "<system-reminder>" in result[1].content

    def test_injection_empty_messages(self):
        """빈 메시지 리스트에 대해 안전하게 처리."""
        result = prepend_user_context([], claude_md_content="Rules")
        assert result == []


# =============================================================================
# Test: Git Snapshot Freezing
# =============================================================================

class TestGitSnapshotFreezing:
    """Git 스냅샷이 2,000자 상한선을 적용하고 동결되는지 검증."""

    @patch("app.middleware.prompt.prompt_assembler.subprocess.check_output")
    def test_snapshot_truncation(self, mock_check_output):
        # lru_cache 초기화
        get_frozen_git_snapshot.cache_clear()

        long_status = "M  " + "a" * 3000 + ".py"
        mock_check_output.side_effect = [long_status, "abc1234 Initial commit"]

        result = get_frozen_git_snapshot(cwd="/test")
        assert "truncated" in result
        assert len(result.split("snapshot in time")[0]) < MAX_STATUS_CHARS + 500
        get_frozen_git_snapshot.cache_clear()

    @patch("app.middleware.prompt.prompt_assembler.subprocess.check_output")
    def test_snapshot_frozen(self, mock_check_output):
        """lru_cache에 의해 동결되므로 두 번 호출해도 subprocess가 한 번만 실행."""
        get_frozen_git_snapshot.cache_clear()

        mock_check_output.side_effect = [
            "M file.py",
            "abc1234 commit msg",
        ]
        result1 = get_frozen_git_snapshot(cwd="/freeze_test")
        result2 = get_frozen_git_snapshot(cwd="/freeze_test")

        assert result1 == result2
        assert mock_check_output.call_count == 2  # status + log = 2 calls (1 invocation)
        get_frozen_git_snapshot.cache_clear()

    @patch("app.middleware.prompt.prompt_assembler.subprocess.check_output")
    def test_snapshot_exception_returns_empty(self, mock_check_output):
        get_frozen_git_snapshot.cache_clear()
        mock_check_output.side_effect = Exception("Not a git repo")
        result = get_frozen_git_snapshot(cwd="/no_git")
        assert result == ""
        get_frozen_git_snapshot.cache_clear()


# =============================================================================
# Test: MCP Attachments Delta
# =============================================================================

class TestMCPDelta:
    """MCP Delta가 마지막 HumanMessage 꼬리에 첨부되는지 검증."""

    def test_mcp_delta_attachment(self):
        messages = [
            SystemMessage(content="System"),
            HumanMessage(content="What databases are available?"),
        ]
        result = attach_mcp_delta_if_needed(
            messages,
            new_mcp_instructions="PostgreSQL MCP: Use pg_query tool for SQL queries.",
        )
        assert "<attachment type='mcp_instructions_delta'>" in result[1].content
        assert "PostgreSQL" in result[1].content
        assert result[0].content == "System"  # SystemMessage 변경 없음

    def test_mcp_delta_empty_instructions(self):
        messages = [HumanMessage(content="Hello")]
        result = attach_mcp_delta_if_needed(messages, new_mcp_instructions="")
        assert result[0].content == "Hello"

    def test_mcp_delta_no_human_message(self):
        messages = [SystemMessage(content="System only")]
        result = attach_mcp_delta_if_needed(messages, new_mcp_instructions="New MCP tool")
        assert len(result) == 1
        assert result[0].content == "System only"


# =============================================================================
# Test: Scratchpad Directory
# =============================================================================

class TestScratchpad:
    """세션별 스크래치패드 디렉터리 생성 검증."""

    def test_create_scratchpad_with_base_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = create_session_scratchpad(session_id="test-001", base_dir=tmpdir)
            assert os.path.isdir(path)
            assert "test-001" in path
            assert "scratchpad" in path

    def test_create_scratchpad_without_base_dir(self):
        path = create_session_scratchpad()
        assert os.path.isdir(path)
        assert "agent_scratch_" in path
        # Cleanup
        os.rmdir(path)

    def test_assembler_scratchpad_lazy_init(self):
        assembler = PromptAssembler(
            system_rules="",
            tool_schemas=[],
            claude_code_modules=True,
        )
        dir1 = assembler.get_scratchpad_dir(session_id="lazy-001")
        dir2 = assembler.get_scratchpad_dir(session_id="lazy-001")
        assert dir1 == dir2  # Lazy init: 같은 디렉터리 반환


# =============================================================================
# Test: Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    """기존 호출 패턴이 깨지지 않는지 검증 (main_agent.py, notebook 호환)."""

    def test_legacy_constructor(self):
        """기존 시그니처 그대로 동작해야 함."""
        assembler = PromptAssembler(
            system_rules="Core Rules: No emojis.",
            tool_schemas=[{"name": "test_tool", "description": "some tool"}],
        )
        messages = assembler.assemble(
            user_input="Hello",
            session_context={"cwd": "/workspace", "session_id": "test"}
        )
        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], SystemMessage)
        assert isinstance(messages[2], HumanMessage)
        assert "Core Rules: No emojis." in messages[0].content
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in messages[0].content

    def test_legacy_with_memory_path(self):
        """memory_path 파라미터 지원 (하위 호환)."""
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as tmp:
            tmp.write("- [Task](file.md) - description\n" * 10)
            tmp_name = tmp.name

        try:
            assembler = PromptAssembler(
                system_rules="Rules",
                tool_schemas=[],
                memory_path=tmp_name,
            )
            messages = assembler.assemble(
                user_input="Hello",
                session_context={"cwd": "/workspace", "session_id": "test"}
            )
            assert "- [Task](file.md)" in messages[1].content
        finally:
            os.unlink(tmp_name)

    def test_legacy_with_l4_l5_docs(self):
        """l4_docs, l5_docs dict 파라미터 지원."""
        assembler = PromptAssembler(
            system_rules="Rules",
            tool_schemas=[],
            l4_docs={"MCP.md": "MCP server instructions here."},
            l5_docs={"AGENT.md": "Agent rules: be concise."},
        )
        dynamic = assembler.build_dynamic_content(session_context={})
        assert "MCP server instructions here." in dynamic
        assert "Agent rules: be concise." in dynamic

    def test_legacy_skill_catalog(self):
        """skill_catalog callable 지원."""
        assembler = PromptAssembler(
            system_rules="Rules",
            tool_schemas=[],
            skill_catalog=lambda: "Available skills: search, code_review",
        )
        static = assembler.build_static_content()
        assert "search" in static
        assert "code_review" in static

    def test_create_middleware_factory_signature(self):
        """create_prompt_assembler_middleware 기존 시그니처 호환."""
        assembler = PromptAssembler(system_rules="Rules", tool_schemas=[])
        middleware = create_prompt_assembler_middleware(assembler, merge_system=True)
        assert isinstance(middleware, PromptAssemblerMiddleware)
        assert middleware.merge_system is True

    def test_main_agent_pattern(self):
        """main_agent.py의 실제 호출 패턴 재현."""
        assembler = PromptAssembler(
            system_rules="SUPERVISOR_SYSTEM_PROMPT content here",
            tool_schemas=[
                {"name": "supervisor_delegate", "description": "Delegates tasks"},
                {"name": "human_escalation", "description": "Escalates to human"},
            ],
            skill_catalog=lambda: "Skill 1: Search\nSkill 2: Code Review",
            l4_docs={},
            agent_rules_path=None,
        )
        prompt_mw = create_prompt_assembler_middleware(assembler, merge_system=True)

        assert isinstance(prompt_mw, PromptAssemblerMiddleware)
        assert prompt_mw.merge_system is True


# =============================================================================
# Test: Middleware with bypass_user_context
# =============================================================================

class TestMiddlewareBypass:
    """PromptAssemblerMiddleware의 bypass_user_context 기능 검증."""

    def test_middleware_bypass_user_context(self):
        from langchain.agents.middleware import ModelRequest

        assembler = PromptAssembler(
            system_rules="Core Rules.",
            tool_schemas=[],
            claude_code_modules=True,
        )

        class DummyContext:
            def __init__(self):
                self.cwd = "/test_cwd"
                self.session_id = "test_sess"

        class DummyRuntime:
            def __init__(self):
                self.context = DummyContext()

        request = ModelRequest(
            messages=[HumanMessage(content="Hello World")],
            model="mock-model",
            tools=[],
            tool_choice="auto",
            model_settings={},
            state={"messages": [HumanMessage(content="Hello World")]},
            runtime=DummyRuntime()
        )

        middleware = create_prompt_assembler_middleware(
            assembler,
            merge_system=False,
            bypass_user_context=True,
            claude_md_content="Always use type hints.",
        )

        def dummy_handler(req):
            return req

        # Mock git snapshot to avoid real git calls
        with patch(
            "app.middleware.prompt.prompt_assembler.get_frozen_git_snapshot",
            return_value="M file.py\nRecent commits: abc1234"
        ):
            modified_req = middleware.wrap_model_call(request, dummy_handler)

        # 3 messages: Static System + Dynamic System + HumanMessage (with bypass)
        assert len(modified_req.messages) == 3

        # HumanMessage에 <system-reminder>가 주입되었는지 확인
        human_msg = modified_req.messages[2]
        assert isinstance(human_msg, HumanMessage)
        assert "<system-reminder>" in human_msg.content
        assert "Always use type hints." in human_msg.content
        assert "Hello World" in human_msg.content

    def test_middleware_without_bypass(self):
        """bypass_user_context=False 시 기존과 동일한 동작."""
        from langchain.agents.middleware import ModelRequest

        assembler = PromptAssembler(
            system_rules="Core Rules.",
            tool_schemas=[],
        )

        class DummyContext:
            def __init__(self):
                self.cwd = "/test_cwd"
                self.session_id = "test_sess"

        class DummyRuntime:
            def __init__(self):
                self.context = DummyContext()

        request = ModelRequest(
            messages=[HumanMessage(content="Hello")],
            model="mock-model",
            tools=[],
            tool_choice="auto",
            model_settings={},
            state={"messages": [HumanMessage(content="Hello")]},
            runtime=DummyRuntime()
        )

        middleware = create_prompt_assembler_middleware(assembler, merge_system=False)

        def dummy_handler(req):
            return req

        modified_req = middleware.wrap_model_call(request, dummy_handler)

        # SystemMessage에 cache_control 존재 확인
        assert modified_req.messages[0].additional_kwargs.get("cache_control") == {"type": "ephemeral"}
        # HumanMessage에 <system-reminder> 없음
        assert "<system-reminder>" not in modified_req.messages[2].content

