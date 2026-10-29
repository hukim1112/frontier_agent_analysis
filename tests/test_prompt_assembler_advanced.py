"""
===============================================================================
Advanced Tests for Claude Code 16-Module Prompt Assembler
===============================================================================
Validates: 16-module factory, alphabetical tool sort, User Context (separate message),
           Git snapshot (system context), session-cached sections, date_change,
           MCP Delta, Scratchpad, Middleware extensions.
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
    get_memory_section,
    get_memory_index_section,
    get_env_info_section,
    get_language_section,
    get_output_style_section,
    get_mcp_instructions_section,
    get_scratchpad_section,
    get_frc_section,
    get_summarize_tool_results_section,
    # Bypass & Attachments
    get_frozen_git_snapshot,
    prepend_user_context,
    append_episodic_memory,
    split_recalled_memory,
    get_skill_listing_delta,
    get_mcp_instructions_delta,
    announced_skill_names,
    announced_mcp_servers,
    create_session_scratchpad,
    sort_tools_for_cache,
    # Constants
    SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
    MAX_STATUS_CHARS,
)
from app.utils.message_markers import (
    ORIGIN_DATE_CHANGE,
    ORIGIN_EPISODIC_MEMORY,
    ORIGIN_MCP_INSTRUCTIONS_DELTA,
    ORIGIN_SKILL_LISTING,
    ORIGIN_USER_CONTEXT,
    TRANSCRIPT_SKIP_KEY,
    is_user_turn_start,
    last_user_turn_start,
    message_origin,
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

    def test_module_0_identity_replaces_default(self):
        """에이전트 역할 지침이 있으면 기본 코딩 에이전트 정체성을 대신하고, URL 금지 문장은 남습니다."""
        result = get_simple_intro_section("You are The Analyst.")
        assert result.startswith("You are The Analyst.")
        assert "CLI-based coding agent" not in result
        assert "NEVER make up URLs" in result

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

    def test_module_9_memory_rules(self):
        result = get_memory_section()
        assert "# auto memory" in result
        assert "USER.md" in result and "MEMORY.md" in result
        assert "skill_view" in result and "skill_manage" in result
        assert "session_recall" in result

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
            claude_code_modules=True,
        )
        static = assembler.build_static_content()

        # L1 모듈들이 포함되었는지 확인
        assert "Global Constitution" in static
        assert "NEVER make up URLs" in static  # Module 0
        assert "IMPORTANT" in static  # Module 2 or 3
        assert "Custom project rules." in static  # 에이전트 역할 지침이 Module 0의 정체성 자리에 들어감
        assert "CLI-based coding agent" not in static, "정체성 문장은 하나만"
        assert "Project-Specific System Rules" not in static

        # L2: 도구 사용 원칙(Module 4)만 들어가고, 도구 정의는 원본처럼 API tools 필드로 보냅니다.
        assert "Anti-Raw Bash" in static or "FileRead" in static  # Module 4

        # 경계선
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in static

    def test_dynamic_content_includes_l3_l4_l5_modules(self):
        assembler = PromptAssembler(
            system_rules="Rules",
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

        # L4: Environment + Memory Rules (메모리 데이터는 Message 3으로 가고, L4에는 규칙만 고정)
        assert "/project" in dynamic
        assert "sess-001" in dynamic
        assert "Korean" in dynamic
        assert "Concise" in dynamic
        assert "# auto memory" in dynamic
        assert "Previous task: implement auth module" not in dynamic

        # L5: FRC + Summarize
        assert "cleared" in dynamic.lower() or "context" in dynamic.lower()
        assert "key findings" in dynamic.lower() or "persist" in dynamic.lower()

    def test_full_assembly_structure(self):
        """assemble(): SystemMessage x2 + 사용자 컨텍스트 메시지 + 사용자 질문."""
        assembler = PromptAssembler(
            system_rules="Test rules",
            claude_code_modules=True,
        )
        messages = assembler.assemble(
            user_input="Hello",
            session_context={
                "cwd": "/workspace",
                "session_id": "test",
                "current_date": "2026-10-23",
                "memory_snapshot": "MEMORY: Use ruff.",
                "episodic_context": "EPISODIC: Past session summary.",
            },
            claude_md_content="Use pytest.",
        )
        assert len(messages) == 5
        assert isinstance(messages[0], SystemMessage)  # Static (L1+L2+Boundary)
        assert isinstance(messages[1], SystemMessage)  # Dynamic (L3+L4+L5)
        assert message_origin(messages[2]) == ORIGIN_USER_CONTEXT
        assert "# claudeMd\nUse pytest." in messages[2].content
        assert "# memorySnapshot\nMEMORY: Use ruff." in messages[2].content
        assert "Today's date is 2026-10-23." in messages[2].content
        assert messages[3].content == "Hello"
        assert message_origin(messages[4]) == ORIGIN_EPISODIC_MEMORY
        assert "# episodicMemory\nEPISODIC: Past session summary." in messages[4].content
        assert SYSTEM_PROMPT_DYNAMIC_BOUNDARY in messages[0].content

    def test_session_workspace_is_in_env_info(self):
        """에이전트 역할 지침이 가리키는 세션 작업 폴더가 Claude Code 모드의 환경 정보에도 들어갑니다."""
        assembler = PromptAssembler(claude_code_modules=True)
        dynamic = assembler.build_dynamic_content(
            {"session_id": "s1", "workspace_dir": "artifacts/users/u/s1", "outputs_dir": "artifacts/users/u/s1/outputs"}
        )
        assert "Session Workspace Root: artifacts/users/u/s1" in dynamic
        assert "Outputs Dir (data/charts/reports): artifacts/users/u/s1/outputs" in dynamic
        assert "Plan Dir (plan_state/extraction_plan): artifacts/users/u/s1/plan" in dynamic

    def test_default_identity_without_system_rules(self):
        static = PromptAssembler(claude_code_modules=True).build_static_content()
        assert "CLI-based coding agent" in static

    def test_date_and_git_are_not_in_system_env(self):
        """원본 env_info에는 날짜·git 상태가 없습니다. git은 시스템 프롬프트 맨 끝(gitStatus)으로 갑니다."""
        assembler = PromptAssembler(system_rules="", claude_code_modules=True)
        dynamic = assembler.build_dynamic_content(
            {"session_id": "s1", "current_date": "2026-10-23", "git_status": "Current branch: main"}
        )
        assert "Current Date" not in dynamic and "2026-10-23" not in dynamic
        assert dynamic.rstrip().endswith("gitStatus: Current branch: main")

    def test_dynamic_sections_are_computed_once_per_session(self):
        """systemPromptSection: 같은 세션에서는 처음 계산한 값을 재사용하고, 새 세션은 새로 계산합니다."""
        assembler = PromptAssembler(system_rules="", claude_code_modules=True)
        first = assembler.build_dynamic_content({"session_id": "s1", "cwd": "/a"})
        again = assembler.build_dynamic_content({"session_id": "s1", "cwd": "/b"})
        other = assembler.build_dynamic_content({"session_id": "s2", "cwd": "/b"})
        assert first == again and "/a" in again
        assert "/b" in other
        assembler.clear_session_sections("s1")
        assert "/b" in assembler.build_dynamic_content({"session_id": "s1", "cwd": "/b"})

    def test_memory_data_goes_to_user_messages_not_l4(self):
        """recalled_memory가 바뀌어도 L4(Message 2)는 불변이고, 스냅샷은 Message 3에, 에피소드는 끝에 붙습니다."""
        assembler = PromptAssembler(system_rules="", claude_code_modules=True)
        dyn1 = assembler.build_dynamic_content({"session_id": "s1", "recalled_memory": "first recall"})
        dyn2 = assembler.build_dynamic_content({"session_id": "s1", "recalled_memory": "second recall"})
        assert dyn1 == dyn2
        assert "first recall" not in dyn1 and "# auto memory" in dyn1

    def test_no_legacy_placeholders_in_claude_code_mode(self):
        """claude_code_modules=True일 때 구형 플레이스홀더('No dynamic memory...', 'No local project rules...')를 출력하지 않습니다."""
        assembler = PromptAssembler(claude_code_modules=True)
        dynamic = assembler.build_dynamic_content({"session_id": "s1", "cwd": "/workspace"})
        assert "No dynamic memory or session documents active." not in dynamic
        assert "No local project rules (AGENT.md/CLAUDE.md) provided." not in dynamic
        assert "=== Layer 5: Resource Control ===" in dynamic

    def test_session_date_is_fixed_within_session(self):
        assembler = PromptAssembler(system_rules="", claude_code_modules=True)
        with patch("app.middleware.prompt.prompt_assembler.get_local_iso_date", return_value="2026-10-23"):
            assert assembler.session_date("s1") == "2026-10-23"
        with patch("app.middleware.prompt.prompt_assembler.get_local_iso_date", return_value="2026-10-24"):
            assert assembler.session_date("s1") == "2026-10-23"
            assert assembler.session_date("s2") == "2026-10-24"


# =============================================================================
# Test: Alphabetical Tool Sort
# =============================================================================

class TestAlphabeticalToolSort:
    """API tools 필드로 보낼 도구 정의가 이름순으로 정렬되는지 검증."""

    def test_tool_sort_order(self):
        tools = [{"name": "z_write"}, {"name": "a_read"}, {"name": "m_search"}]
        assert [t["name"] for t in sort_tools_for_cache(tools)] == ["a_read", "m_search", "z_write"]

    def test_sort_tools_for_cache_keeps_mcp_after_builtins(self):
        tools = [{"name": "mcp__db__query"}, {"name": "z_write"}, {"name": "mcp__a__x"}, {"name": "a_read"}]
        assert [t["name"] for t in sort_tools_for_cache(tools)] == [
            "a_read", "z_write", "mcp__a__x", "mcp__db__query"]

    def test_tool_sort_determinism(self):
        """동일한 도구 집합은 등록 순서와 무관하게 항상 같은 순서가 되어야 함 (캐시 일관성)."""
        tools = [{"name": "c_tool"}, {"name": "a_tool"}, {"name": "b_tool"}]
        assert sort_tools_for_cache(tools) == sort_tools_for_cache(list(reversed(tools)))


class TestToolDefinitionsStayOutOfSystemPrompt:
    """원본처럼 도구 정의는 API tools 필드로만 보내고, 시스템 프롬프트에는 넣지 않습니다."""

    @pytest.mark.parametrize("claude_code_modules", [True, False])
    @pytest.mark.parametrize("merge_system", [True, False])
    def test_request_tools_are_not_rendered_into_system_prompt(self, claude_code_modules, merge_system):
        from langchain.agents.middleware import ModelRequest

        assembler = PromptAssembler(system_rules="Rules", claude_code_modules=claude_code_modules)
        middleware = create_prompt_assembler_middleware(assembler, merge_system=merge_system)
        tools = [{"name": "zz_unique_tool", "description": "UNIQUE_TOOL_DESCRIPTION",
                  "parameters": {"properties": {"unique_param": {"type": "string"}}}}]
        request = ModelRequest(
            messages=[HumanMessage(content="hi")], model="mock-model", tools=tools, tool_choice="auto",
            model_settings={}, state={"messages": []},
            runtime=MagicMock(context=None, config=None, execution_info=None),
        )
        out = middleware.wrap_model_call(request, lambda req: req)

        system_text = "\n".join(m.content for m in out.messages if isinstance(m, SystemMessage))
        for needle in ("zz_unique_tool", "UNIQUE_TOOL_DESCRIPTION", "unique_param"):
            assert needle not in system_text
        assert [t["name"] for t in out.tools] == ["zz_unique_tool"], "도구 정의는 tools 필드로 그대로 전달"


# =============================================================================
# Test: User Context Bypass Injection
# =============================================================================

class TestUserContextBypass:
    """prepend_user_context: 원본 prependUserContext처럼 대화 맨 앞에 별도 메시지를 1회 끼워 넣는지 검증."""

    def test_injection_adds_separate_message(self):
        messages = [
            HumanMessage(content="What is this project?"),
            AIMessage(content="Let me check."),
        ]
        result = prepend_user_context(
            messages,
            claude_md_content="Always use pytest for testing.",
            current_date="2026-10-23",
        )
        assert len(result) == 3
        ctx_msg = result[0]
        assert message_origin(ctx_msg) == ORIGIN_USER_CONTEXT
        assert ctx_msg.content.startswith("<system-reminder>\nAs you answer the user's questions")
        assert "# claudeMd\nAlways use pytest for testing." in ctx_msg.content
        assert "# currentDate\nToday's date is 2026-10-23." in ctx_msg.content
        assert "may or may not be relevant" in ctx_msg.content
        assert result[1].content == "What is this project?", "첫 사용자 메시지는 고치지 않습니다"

    def test_injection_is_idempotent(self):
        """이미 주입된 경우 재주입하지 않아야 함."""
        once = prepend_user_context([HumanMessage(content="Hello")], claude_md_content="Rules",
                                    current_date="2026-10-23")
        twice = prepend_user_context(once, claude_md_content="New content", current_date="2026-10-24")
        assert twice == once

    def test_user_context_is_not_a_user_turn(self):
        result = prepend_user_context([HumanMessage(content="Hello")], current_date="2026-10-23")
        assert last_user_turn_start(result) == 1

    def test_injection_skipped_when_no_content(self):
        """주입할 내용이 없으면 원본 메시지 그대로 반환."""
        messages = [HumanMessage(content="Hello")]
        result = prepend_user_context(messages, claude_md_content="", current_date="")
        assert result == messages

    def test_injection_goes_after_system_messages(self):
        """SystemMessage 뒤, 첫 대화 메시지 앞에 들어갑니다."""
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
        assert message_origin(result[1]) == ORIGIN_USER_CONTEXT
        assert result[2].content == "User query"

    def test_injection_empty_messages(self):
        """빈 메시지 리스트에 대해 안전하게 처리."""
        result = prepend_user_context([], claude_md_content="Rules")
        assert result == []


# =============================================================================
# Test: Git Snapshot Freezing
# =============================================================================

class TestGitSnapshotFreezing:
    """Git 스냅샷이 2,000자 상한선을 적용하고 동결되는지 검증."""

    @staticmethod
    def _git_outputs(status="M file.py", log="abc1234 commit msg"):
        # rev-parse(is-inside) → branch → origin HEAD → user.name → status → log
        return ["true", "feature-x", "origin/main", "Alice", status, log]

    @patch("app.middleware.prompt.prompt_assembler.subprocess.check_output")
    def test_snapshot_matches_original_format(self, mock_check_output):
        get_frozen_git_snapshot.cache_clear()
        mock_check_output.side_effect = self._git_outputs()
        result = get_frozen_git_snapshot(cwd="/fmt")
        assert result.startswith("This is the git status at the start of the conversation.")
        assert "Current branch: feature-x" in result
        assert "Main branch (you will usually use this for PRs): main" in result
        assert "Git user: Alice" in result
        assert "Status:\nM file.py" in result
        assert result.endswith("Recent commits:\nabc1234 commit msg")
        get_frozen_git_snapshot.cache_clear()

    @patch("app.middleware.prompt.prompt_assembler.subprocess.check_output")
    def test_snapshot_truncation(self, mock_check_output):
        # lru_cache 초기화
        get_frozen_git_snapshot.cache_clear()

        long_status = "M  " + "a" * 3000 + ".py"
        mock_check_output.side_effect = self._git_outputs(status=long_status)

        result = get_frozen_git_snapshot(cwd="/test")
        assert "truncated because it exceeds 2k characters" in result
        status_part = result.split("Status:\n")[1].split("\n\nRecent commits")[0]
        assert len(status_part) < MAX_STATUS_CHARS + 200
        get_frozen_git_snapshot.cache_clear()

    @patch("app.middleware.prompt.prompt_assembler.subprocess.check_output")
    def test_snapshot_frozen(self, mock_check_output):
        """lru_cache에 의해 동결되므로 두 번 호출해도 git은 한 번만 실행."""
        get_frozen_git_snapshot.cache_clear()

        mock_check_output.side_effect = self._git_outputs()
        result1 = get_frozen_git_snapshot(cwd="/freeze_test")
        result2 = get_frozen_git_snapshot(cwd="/freeze_test")

        assert result1 == result2
        assert mock_check_output.call_count == 6  # 1회 수집분만
        get_frozen_git_snapshot.cache_clear()

    @patch("app.middleware.prompt.prompt_assembler.subprocess.check_output")
    def test_snapshot_not_a_repo(self, mock_check_output):
        get_frozen_git_snapshot.cache_clear()
        mock_check_output.side_effect = ["false"]
        assert get_frozen_git_snapshot(cwd="/plain_dir") == ""
        get_frozen_git_snapshot.cache_clear()

    @patch("app.middleware.prompt.prompt_assembler.subprocess.check_output")
    def test_snapshot_exception_returns_empty(self, mock_check_output):
        get_frozen_git_snapshot.cache_clear()
        mock_check_output.side_effect = Exception("Not a git repo")
        result = get_frozen_git_snapshot(cwd="/no_git")
        assert result == ""
        get_frozen_git_snapshot.cache_clear()


# =============================================================================
# Test: ⑦ Attachments — 스킬 목록 · MCP 지침 (대화에 저장되는 알림)
# =============================================================================

SKILLS = [
    {"name": "pdf", "description": "Extract text from PDFs", "path": "skills/pdf/SKILL.md"},
    {"name": "yfinance", "description": "Market data", "path": "skills/yfinance/SKILL.md"},
]


class TestSkillListingAttachment:
    """원본 getSkillListingAttachments: 첫 알림은 전체, 이후 새 스킬만."""

    def test_first_listing_contains_all_skills(self):
        msg = get_skill_listing_delta([HumanMessage(content="hi")], SKILLS)
        assert message_origin(msg) == ORIGIN_SKILL_LISTING
        assert msg.content.startswith("<system-reminder>\n")
        assert "- pdf: Extract text from PDFs (skills/pdf/SKILL.md)" in msg.content
        assert "- yfinance: Market data" in msg.content
        assert not is_user_turn_start(msg), "알림은 사용자 턴이 아닙니다"
        assert msg.additional_kwargs.get(TRANSCRIPT_SKIP_KEY), "원문 저장소에 사용자 발화로 기록하지 않습니다"

    def test_nothing_new_returns_none(self):
        first = get_skill_listing_delta([], SKILLS)
        assert get_skill_listing_delta([HumanMessage(content="hi"), first], SKILLS) is None

    def test_only_new_skill_is_announced(self):
        first = get_skill_listing_delta([], SKILLS)
        more = SKILLS + [{"name": "new-skill", "description": "Installed mid-session", "path": "p"}]
        delta = get_skill_listing_delta([first], more)
        assert "new-skill" in delta.content
        assert "pdf" not in delta.content
        assert announced_skill_names([first, delta]) == {"pdf", "yfinance", "new-skill"}

    def test_long_description_is_truncated(self):
        msg = get_skill_listing_delta([], [{"name": "x", "description": "a" * 1000, "path": ""}])
        assert len(msg.content) < 500


class TestMCPInstructionsDelta:
    """원본 getMcpInstructionsDelta: 이름 기준 diff, 알린 상태는 대화에서 다시 계산."""

    def test_first_connect_announces_server_blocks(self):
        msg = get_mcp_instructions_delta([], {"wikipedia": "Use search first.", "db": "Use pg_query."})
        assert message_origin(msg) == ORIGIN_MCP_INSTRUCTIONS_DELTA
        assert "# MCP Server Instructions" in msg.content
        assert "## db\nUse pg_query." in msg.content
        assert msg.content.index("## db") < msg.content.index("## wikipedia"), "서버 이름순"

    def test_no_change_returns_none(self):
        first = get_mcp_instructions_delta([], {"wikipedia": "Use search first."})
        assert get_mcp_instructions_delta([first], {"wikipedia": "Use search first."}) is None

    def test_connect_and_disconnect(self):
        first = get_mcp_instructions_delta([], {"wikipedia": "W", "github": "G"})
        delta = get_mcp_instructions_delta([first], {"wikipedia": "W", "slack": "S"})
        assert "## slack\nS" in delta.content
        assert "## wikipedia" not in delta.content, "이미 알린 서버는 다시 알리지 않습니다"
        assert "have disconnected" in delta.content and "github" in delta.content
        assert announced_mcp_servers([first, delta]) == {"wikipedia", "slack"}

    def test_server_without_instructions_is_ignored(self):
        assert get_mcp_instructions_delta([], {"silent": ""}) is None

    def test_announced_message_is_never_modified(self):
        """한 번 넣은 알림은 그대로 두고 다음 알림만 추가 → 앞부분 캐시가 깨지지 않습니다."""
        user1 = HumanMessage(content="질문1")
        first = get_mcp_instructions_delta([user1], {"wikipedia": "W"})
        history = [user1, first, AIMessage(content="답1"), HumanMessage(content="질문2")]
        snapshot = [m.content for m in history]
        assert get_mcp_instructions_delta(history, {"wikipedia": "W"}) is None
        assert [m.content for m in history] == snapshot


class TestAssemblerAttachments:
    def _assembler(self, **kw):
        return PromptAssembler(
            system_rules="Rules", claude_code_modules=True,
            skill_guidelines="Read SKILL.md before using a skill.",
            skill_provider=lambda: list(SKILLS),
            mcp_instructions=lambda: {"wikipedia": "Use search first."},
            mcp_delta_enabled=True, **kw,
        )

    def test_static_prompt_has_guidelines_but_no_skill_list(self):
        static = self._assembler().build_static_content()
        assert "Read SKILL.md before using a skill." in static
        assert "yfinance" not in static, "스킬 목록은 정적 영역에 넣지 않습니다"

    def test_static_prompt_is_independent_of_installed_skills(self):
        skills = list(SKILLS)
        assembler = PromptAssembler(system_rules="", claude_code_modules=True,
                                    skill_guidelines="Policy", skill_provider=lambda: skills)
        before = assembler.build_static_content()
        skills.append({"name": "late", "description": "d", "path": "p"})
        assert assembler.build_static_content() == before

    def test_dynamic_prompt_has_no_mcp_instructions_in_delta_mode(self):
        dynamic = self._assembler().build_dynamic_content({"session_id": "s1"})
        assert "Use search first." not in dynamic

    def test_mcp_dict_goes_to_system_prompt_when_delta_disabled(self):
        assembler = PromptAssembler(system_rules="", claude_code_modules=True,
                                    mcp_instructions=lambda: {"wikipedia": "Use search first."})
        dynamic = assembler.build_dynamic_content({"session_id": "s1"})
        assert "# MCP Server Instructions" in dynamic and "## wikipedia\nUse search first." in dynamic

    def test_collect_attachments_then_nothing_more(self):
        assembler = self._assembler()
        history = [HumanMessage(content="hi")]
        first = assembler.collect_attachments(history)
        assert [message_origin(m) for m in first] == [ORIGIN_SKILL_LISTING, ORIGIN_MCP_INSTRUCTIONS_DELTA]
        assert assembler.collect_attachments(history + first) == []

    def test_assemble_appends_attachments_after_user_input(self):
        messages = self._assembler().assemble("질문", {"session_id": "s1", "current_date": "2026-10-23"})
        origins = [message_origin(m) for m in messages]
        assert origins == [None, None, ORIGIN_USER_CONTEXT, None, ORIGIN_SKILL_LISTING, ORIGIN_MCP_INSTRUCTIONS_DELTA]
        assert messages[3].content == "질문"

    def test_legacy_skill_catalog_still_static(self):
        assembler = PromptAssembler(system_rules="", skill_catalog=lambda: "Catalog: pdf")
        assert "Catalog: pdf" in assembler.build_static_content()
        assert assembler.collect_attachments([HumanMessage(content="hi")]) == []

    def test_static_content_is_built_once(self):
        calls = []
        assembler = PromptAssembler(system_rules="", claude_code_modules=True,
                                    skill_guidelines=lambda: calls.append(1) or "Policy")
        assembler.build_static_content()
        assembler.build_static_content()
        assert len(calls) == 1
        assembler.clear_static_cache()
        assembler.build_static_content()
        assert len(calls) == 2


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
            claude_code_modules=True,
        )
        dir1 = assembler.get_scratchpad_dir(session_id="lazy-001")
        dir2 = assembler.get_scratchpad_dir(session_id="lazy-001")
        assert dir1 == dir2  # Lazy init: 같은 디렉터리 반환

    def test_scratchpad_is_separate_per_session(self):
        assembler = PromptAssembler(system_rules="", claude_code_modules=True)
        assert assembler.get_scratchpad_dir("sess-a") != assembler.get_scratchpad_dir("sess-b")
        dyn_a = assembler.build_dynamic_content({"session_id": "sess-a"})
        dyn_b = assembler.build_dynamic_content({"session_id": "sess-b"})
        assert assembler.get_scratchpad_dir("sess-a") in dyn_a
        assert assembler.get_scratchpad_dir("sess-b") in dyn_b


# =============================================================================
# Test: Backward Compatibility
# =============================================================================

class TestBackwardCompatibility:
    """기존 호출 패턴이 깨지지 않는지 검증 (main_agent.py, notebook 호환)."""

    def test_legacy_constructor(self):
        """기존 시그니처 그대로 동작해야 함."""
        assembler = PromptAssembler(
            system_rules="Core Rules: No emojis.",
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
            skill_catalog=lambda: "Available skills: search, code_review",
        )
        static = assembler.build_static_content()
        assert "search" in static
        assert "code_review" in static

    def test_create_middleware_factory_signature(self):
        """create_prompt_assembler_middleware 기존 시그니처 호환."""
        assembler = PromptAssembler(system_rules="Rules")
        middleware = create_prompt_assembler_middleware(assembler, merge_system=True)
        assert isinstance(middleware, PromptAssemblerMiddleware)
        assert middleware.merge_system is True

    def test_main_agent_pattern(self):
        """main_agent.py의 실제 호출 패턴 재현."""
        assembler = PromptAssembler(
            system_rules="SUPERVISOR_SYSTEM_PROMPT content here",
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

        request = request.override(state={**request.state, "session_start_date": "2026-10-23"})
        # Mock git snapshot to avoid real git calls
        with patch(
            "app.middleware.prompt.prompt_assembler.get_frozen_git_snapshot",
            return_value="Current branch: main\n\nRecent commits:\nabc1234"
        ):
            modified_req = middleware.wrap_model_call(request, dummy_handler)

        # 4 messages: Static System + Dynamic System(+gitStatus) + 사용자 컨텍스트 + 사용자 질문
        assert len(modified_req.messages) == 4
        assert modified_req.messages[1].content.rstrip().endswith("abc1234")
        assert "gitStatus: Current branch: main" in modified_req.messages[1].content

        ctx_msg = modified_req.messages[2]
        assert message_origin(ctx_msg) == ORIGIN_USER_CONTEXT
        assert "Always use type hints." in ctx_msg.content
        assert "Today's date is 2026-10-23." in ctx_msg.content, "state에 고정된 세션 시작 날짜를 씁니다"
        assert modified_req.messages[3].content == "Hello World"

    def test_middleware_sorts_tools(self):
        from langchain.agents.middleware import ModelRequest

        middleware = create_prompt_assembler_middleware(PromptAssembler(system_rules="", claude_code_modules=True))
        tools = [{"name": "mcp__db__q"}, {"name": "z_tool"}, {"name": "a_tool"}]
        request = ModelRequest(
            messages=[HumanMessage(content="hi")], model="mock-model", tools=tools, tool_choice="auto",
            model_settings={}, state={"messages": []}, runtime=MagicMock(context=None, config=None,
                                                                         execution_info=None),
        )
        out = middleware.wrap_model_call(request, lambda req: req)
        assert [t["name"] for t in out.tools] == ["a_tool", "z_tool", "mcp__db__q"]


# =============================================================================
# Test: Session date & date_change (before_model)
# =============================================================================

class TestDateChange:
    """원본 getDateChangeAttachments: 첫 호출은 기록만, 날짜가 바뀌면 대화 끝에 알림."""

    def _mw(self):
        return create_prompt_assembler_middleware(PromptAssembler(system_rules="", claude_code_modules=True))

    def test_first_call_records_date_only(self):
        with patch("app.middleware.prompt.prompt_assembler.get_local_iso_date", return_value="2026-10-23"):
            update = self._mw().before_model({"messages": []})
        assert update == {"session_start_date": "2026-10-23", "last_emitted_date": "2026-10-23"}

    def test_same_day_does_nothing(self):
        state = {"messages": [], "session_start_date": "2026-10-23", "last_emitted_date": "2026-10-23"}
        with patch("app.middleware.prompt.prompt_assembler.get_local_iso_date", return_value="2026-10-23"):
            assert self._mw().before_model(state) is None

    def test_midnight_appends_notice_and_keeps_start_date(self):
        state = {"messages": [], "session_start_date": "2026-10-23", "last_emitted_date": "2026-10-23"}
        with patch("app.middleware.prompt.prompt_assembler.get_local_iso_date", return_value="2026-10-24"):
            update = self._mw().before_model(state)
        assert "session_start_date" not in update, "앞부분(사용자 컨텍스트)의 날짜는 바꾸지 않습니다"
        assert update["last_emitted_date"] == "2026-10-24"
        notice = update["messages"][0]
        assert message_origin(notice) == ORIGIN_DATE_CHANGE
        assert "Today's date is now 2026-10-24" in notice.content
        assert not is_user_turn_start(notice)

    def test_middleware_without_bypass(self):
        """bypass_user_context=False 시 기존과 동일한 동작."""
        from langchain.agents.middleware import ModelRequest

        assembler = PromptAssembler(
            system_rules="Core Rules.",
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



# =============================================================================
# Test: Middleware — 알림 저장 · 압축 후 재알림 · 세션별 준비값
# =============================================================================

class TestMiddlewareAttachments:
    def _mw(self, skills=None, servers=None, **kw):
        skills = SKILLS if skills is None else skills
        servers = {"wikipedia": "Use search first."} if servers is None else servers
        assembler = PromptAssembler(
            system_rules="", claude_code_modules=True,
            skill_guidelines="Policy", skill_provider=lambda: list(skills),
            mcp_instructions=lambda: dict(servers), mcp_delta_enabled=True,
        )
        return create_prompt_assembler_middleware(assembler, **kw)

    def _state(self, messages):
        return {"messages": messages, "session_start_date": "2026-10-23", "last_emitted_date": "2026-10-23"}

    def test_before_model_saves_attachments_once(self):
        mw = self._mw()
        history = [HumanMessage(content="질문1", id="u1")]
        with patch("app.middleware.prompt.prompt_assembler.get_local_iso_date", return_value="2026-10-23"):
            update = mw.before_model(self._state(history))
            origins = [message_origin(m) for m in update["messages"]]
            assert origins == [ORIGIN_SKILL_LISTING, ORIGIN_MCP_INSTRUCTIONS_DELTA]
            # 다음 모델 호출: state에 알림이 저장되어 있으므로 아무것도 추가하지 않습니다.
            assert mw.before_model(self._state(history + update["messages"])) is None

    def test_newly_installed_skill_is_appended_at_the_end(self):
        skills = list(SKILLS)
        mw = self._mw(skills=skills)
        with patch("app.middleware.prompt.prompt_assembler.get_local_iso_date", return_value="2026-10-23"):
            first = mw.before_model(self._state([HumanMessage(content="q")]))["messages"]
            skills.append({"name": "late-skill", "description": "d", "path": "p"})
            later = mw.before_model(self._state([HumanMessage(content="q"), *first, AIMessage(content="a")]))
        assert len(later["messages"]) == 1 and "late-skill" in later["messages"][0].content

    def test_wrap_model_call_keeps_saved_attachments_in_place(self):
        from langchain.agents.middleware import ModelRequest

        mw = self._mw(bypass_user_context=True, claude_md_content="Use pytest.")
        user = HumanMessage(content="질문1")
        attachments = mw.assembler.collect_attachments([user])
        msgs = [user, *attachments]
        request = ModelRequest(messages=msgs, model="mock-model", tools=[], tool_choice="auto",
                               model_settings={}, state={"messages": msgs, "session_start_date": "2026-10-23"},
                               runtime=MagicMock(context=None, config=None, execution_info=None))
        with patch("app.middleware.prompt.prompt_assembler.get_frozen_git_snapshot", return_value=""):
            out = mw.wrap_model_call(request, lambda req: req)
        origins = [message_origin(m) for m in out.messages]
        assert origins == [None, None, ORIGIN_USER_CONTEXT, None, ORIGIN_SKILL_LISTING, ORIGIN_MCP_INSTRUCTIONS_DELTA]
        assert out.messages[3].content == "질문1", "사용자 메시지는 고치지 않습니다"

    def test_compaction_clears_session_sections_and_reannounces(self):
        from app.utils.message_markers import origin_kwargs, ORIGIN_COMPACTION_SUMMARY

        mw = self._mw()
        mw.assembler.build_dynamic_content({"session_id": "unknown", "cwd": "/old"})
        summary = HumanMessage(content="[Compaction Summary: Previous Conversation]\n...", id="sum1",
                               additional_kwargs=origin_kwargs(ORIGIN_COMPACTION_SUMMARY))
        with patch("app.middleware.prompt.prompt_assembler.get_local_iso_date", return_value="2026-10-23"):
            update = mw.before_model(self._state([summary, HumanMessage(content="q")]))
        assert update["prompt_compaction_marker"] == "sum1"
        assert "/new" in mw.assembler.build_dynamic_content({"session_id": "unknown", "cwd": "/new"})
        # 요약으로 이전 알림이 사라졌으므로 스킬·MCP 알림을 다시 저장합니다.
        assert [message_origin(m) for m in update["messages"]] == [ORIGIN_SKILL_LISTING, ORIGIN_MCP_INSTRUCTIONS_DELTA]

    def test_legacy_mcp_delta_instructions_is_announced_once(self):
        mw = create_prompt_assembler_middleware(
            PromptAssembler(system_rules="", claude_code_modules=True),
            mcp_delta_instructions="PostgreSQL MCP: Use pg_query.",
        )
        user = HumanMessage(content="DB 목록?")
        with patch("app.middleware.prompt.prompt_assembler.get_local_iso_date", return_value="2026-10-23"):
            update = mw.before_model(self._state([user]))
        notice = update["messages"][0]
        assert message_origin(notice) == ORIGIN_MCP_INSTRUCTIONS_DELTA
        assert "PostgreSQL MCP: Use pg_query." in notice.content
        assert user.content == "DB 목록?", "사용자 메시지에 덧붙이지 않습니다"

    def test_git_snapshot_is_collected_once_per_session(self):
        from langchain.agents.middleware import ModelRequest

        mw = create_prompt_assembler_middleware(PromptAssembler(system_rules="", claude_code_modules=True),
                                                bypass_user_context=True)

        def request_for(thread_id):
            msgs = [HumanMessage(content="hi")]
            return ModelRequest(messages=msgs, model="mock-model", tools=[], tool_choice="auto",
                                model_settings={}, state={"messages": msgs},
                                runtime=MagicMock(context=None, execution_info=None,
                                                  config={"configurable": {"thread_id": thread_id}}))

        with patch("app.middleware.prompt.prompt_assembler.get_frozen_git_snapshot",
                   side_effect=lambda cwd, session_id: f"snapshot-{session_id}") as snap:
            a1 = mw.wrap_model_call(request_for("A"), lambda r: r)
            a2 = mw.wrap_model_call(request_for("A"), lambda r: r)
            b1 = mw.wrap_model_call(request_for("B"), lambda r: r)
        assert snap.call_count == 2, "세션당 1회만 수집"
        assert "snapshot-A" in a1.messages[1].content and a1.messages[1].content == a2.messages[1].content
        assert "snapshot-B" in b1.messages[1].content


# =============================================================================
# Test: SkillPromptBuilder.list_skills
# =============================================================================

class TestSkillBuilderListSkills:
    def _write(self, root, name, desc):
        os.makedirs(os.path.join(root, name), exist_ok=True)
        with open(os.path.join(root, name, "SKILL.md"), "w", encoding="utf-8") as f:
            f.write(f"---\nname: {name}\ndescription: {desc}\n---\n# body\n")

    def test_lists_sorted_skills_and_detects_new_ones(self):
        from app.middleware.prompt.skill_builder import SkillPromptBuilder

        with tempfile.TemporaryDirectory() as root:
            self._write(root, "zeta", "Z skill")
            self._write(root, "alpha", "A skill")
            builder = SkillPromptBuilder(skills_dirs=[root], guidelines_path=None)
            assert [s["name"] for s in builder.list_skills()] == ["alpha", "zeta"]
            self._write(root, "beta", "B skill")
            assert [s["name"] for s in builder.list_skills()] == ["alpha", "beta", "zeta"]
            assert "Z skill" in builder.build_catalog()
