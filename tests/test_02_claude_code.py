import os
import tempfile
import pytest
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
from modules.claude_code.prompt_assembler import PromptAssembler
from modules.claude_code.compactor import AutoCompactor
from modules.claude_code.self_correction import StopHooksMiddleware, extract_code_blocks
from modules.claude_code.amnesia_guard import AmnesiaGuardMiddleware

# Mock LLM for unit tests
class MockLLM:
    def __init__(self, response_text="Mocked Summary Response"):
        self.response_text = response_text

    def invoke(self, messages):
        return AIMessage(content=self.response_text)

    def get_num_tokens_from_messages(self, messages):
        # simple heuristic for mocking
        return sum(len(str(m.content)) for m in messages) // 4

def test_prompt_assembler():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as tmp:
        # Write memory index content
        tmp.write("- [Title](file.md) - active work description\n" * 10)
        tmp_name = tmp.name

    try:
        assembler = PromptAssembler(
            system_rules="Core Rules: No emojis.",
            tool_schemas=[{"name": "test_tool", "description": "some tool"}],
            memory_path=tmp_name
        )
        
        messages = assembler.assemble(
            user_input="Hello",
            session_context={"cwd": "/workspace", "session_id": "test_session"}
        )
        
        assert len(messages) == 3
        assert isinstance(messages[0], SystemMessage)
        assert isinstance(messages[1], SystemMessage)
        assert isinstance(messages[2], HumanMessage)
        
        # Verify boundary marker
        assert "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__" in messages[0].content
        # Verify memory content inclusion
        assert "- [Title](file.md)" in messages[1].content
        assert "/workspace" in messages[1].content
    finally:
        os.unlink(tmp_name)

def test_prompt_assembler_with_history():
    assembler = PromptAssembler(
        system_rules="Core Rules.",
        tool_schemas=[],
        memory_path=None
    )
    history = [
        HumanMessage(content="Question 1"),
        AIMessage(content="Answer 1")
    ]
    messages = assembler.assemble(
        user_input="Question 2",
        session_context={},
        chat_history=history
    )
    assert len(messages) == 5
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], SystemMessage)
    assert messages[2].content == "Question 1"
    assert messages[3].content == "Answer 1"
    assert messages[4].content == "Question 2"

def test_prompt_assembler_truncation():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as tmp:
        # Write more than 300 lines to trigger truncation (new version uses max_lines=300)
        tmp.write("line\n" * 350)
        tmp_name = tmp.name

    try:
        assembler = PromptAssembler(
            system_rules="Rules",
            tool_schemas=[],
            memory_path=tmp_name
        )
        messages = assembler.assemble(user_input="Hello", session_context={})
        assert "truncated due to size limits" in messages[1].content
    finally:
        os.unlink(tmp_name)

def test_auto_compactor():
    llm = MockLLM("Summarized conversation history.")
    # Large threshold - no compaction
    compactor = AutoCompactor(llm=llm, threshold_tokens=1000)
    messages = [
        SystemMessage(content="System context"),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi there")
    ]
    compacted, is_compacted = compactor.compact_if_needed(messages)
    assert not is_compacted
    assert len(compacted) == 3

    # Small threshold - triggers compaction
    compactor_small = AutoCompactor(llm=llm, threshold_tokens=2)
    compacted_small, is_compacted_small = compactor_small.compact_if_needed(messages)
    assert is_compacted_small
    assert len(compacted_small) == 3  # SystemMessage + SummaryMessage + Last HumanMessage
    assert "Summarized conversation history." in compacted_small[1].content

def test_extract_code_blocks():
    text = "Here is some code:\n```python\nprint('hello')\n```\nAnd another one:\n```\n1 + 1\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    assert "print('hello')" in blocks[0]
    assert "1 + 1" in blocks[1]

def test_stop_hooks_middleware():
    def dummy_validator(code):
        if "syntax_error" in code:
            return "Syntax Error: invalid syntax"
        return None

    middleware = StopHooksMiddleware(validators=[{"name": "dummy", "fn": dummy_validator}])

    # Valid response
    state = {
        "messages": [
            HumanMessage(content="Write a function"),
            AIMessage(content="Here it is:\n```python\ndef my_func():\n    return 42\n```")
        ]
    }
    res = middleware.after_agent(state, None)
    assert res is None or res.get("transition") == "completed"

    # Invalid response with syntax_error
    state_err = {
        "messages": [
            HumanMessage(content="Write a function"),
            AIMessage(content="Here it is:\n```python\ndef syntax_error():\n    return 42 \n```")
        ]
    }
    res_err = middleware.after_agent(state_err, None)
    assert res_err is not None
    assert len(res_err["messages"]) == 3
    assert "[dummy]: Syntax Error: invalid syntax" in res_err["messages"][-1].content

def test_amnesia_guard():
    with tempfile.NamedTemporaryFile(mode="w+", delete=False, encoding="utf-8") as tmp:
        tmp.write("print('Hello World')")
        tmp_name = tmp.name

    try:
        guard = AmnesiaGuardMiddleware(max_restore_files=2)
        guard.track_file_access(tmp_name)
        guard.set_active_plan("Step 1: Write test code.")

        recovery = guard.create_recovery_attachments()
        assert len(recovery) == 1
        assert "Step 1: Write test code." in recovery[0].content
        assert "print('Hello World')" in recovery[0].content
        assert tmp_name in recovery[0].content
    finally:
        os.unlink(tmp_name)


def test_prompt_assembler_middleware():
    from modules.claude_code.prompt_assembler import create_prompt_assembler_middleware, PromptAssembler
    from langchain.agents.middleware import ModelRequest
    from langchain_core.messages import SystemMessage

    assembler = PromptAssembler(
        system_rules="Core Rules.",
        tool_schemas=[],
        memory_path=None
    )

    class DummyContext:
        def __init__(self):
            self.cwd = "/test_cwd"
            self.session_id = "test_sess"

    class DummyRuntime:
        def __init__(self):
            self.context = DummyContext()

    # Create dummy ModelRequest
    request = ModelRequest(
        messages=[HumanMessage(content="Hello")],
        model="mock-model",
        tools=[],
        tool_choice="auto",
        model_settings={},
        state={"messages": [HumanMessage(content="Hello")]},
        runtime=DummyRuntime()
    )

    middleware = create_prompt_assembler_middleware(assembler)
    
    # The middleware wraps the model call. We call the wrap_model_call handler.
    def dummy_handler(req):
        return req

    modified_req = middleware.wrap_model_call(request, dummy_handler)
    
    # Check if 2 SystemMessages were correctly injected in the request.messages
    assert len(modified_req.messages) == 3
    assert isinstance(modified_req.messages[0], SystemMessage)
    assert isinstance(modified_req.messages[1], SystemMessage)
    assert isinstance(modified_req.messages[2], HumanMessage)
    assert modified_req.messages[0].additional_kwargs.get("cache_control") == {"type": "ephemeral"}
    assert "/test_cwd" in modified_req.messages[1].content
    assert "test_sess" in modified_req.messages[1].content

