import os
import pytest
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from modules.claude_code.compactor import (
    SnipCompactor,
    MicroCompactor,
    ContextCollapse,
    AutoCompactor,
    ReactiveCompactor,
    create_compactor_middleware
)

def test_snip_compactor():
    compactor = SnipCompactor(age_threshold=2)
    messages = [
        SystemMessage(content="System Rules"),
        HumanMessage(content="Read file 1"),
        ToolMessage(content="Line 1: code line\n" * 20, tool_call_id="call_1", name="file_read"), # Old tool msg (age 3)
        HumanMessage(content="Read file 2"),
        ToolMessage(content="Recent line 1\n" * 20, tool_call_id="call_2", name="file_read"), # Recent tool msg (age 1)
        HumanMessage(content="Current query")
    ]
    
    compacted, modified = compactor.compact(messages)
    assert modified is True
    # The first ToolMessage should be snipped
    assert "[Tool result snipped:" in compacted[2].content
    # The second ToolMessage should remain intact
    assert "Recent line 1" in compacted[4].content

def test_micro_compactor(tmp_path):
    swap_dir = str(tmp_path / "swaps")
    compactor = MicroCompactor(max_chars=200, swap_dir=swap_dir)
    
    large_content = "X" * 500
    messages = [
        SystemMessage(content="System Rules"),
        ToolMessage(content=large_content, tool_call_id="call_large", name="bash_command")
    ]
    
    compacted, modified = compactor.compact(messages)
    assert modified is True
    assert "[Output (500 chars) microcompacted to disk:" in compacted[1].content
    assert os.path.exists(swap_dir)
    assert len(os.listdir(swap_dir)) == 1

def test_context_collapse():
    collapse = ContextCollapse(min_consecutive=3)
    messages = [
        SystemMessage(content="System Rules"),
        AIMessage(content="Searching...", tool_calls=[{"name": "grep_search", "args": {}, "id": "1"}]),
        ToolMessage(content="Match 1", tool_call_id="1", name="grep_search"),
        AIMessage(content="Searching 2...", tool_calls=[{"name": "glob_search", "args": {}, "id": "2"}]),
        ToolMessage(content="Match 2", tool_call_id="2", name="glob_search"),
        HumanMessage(content="Tell me what you found")
    ]
    
    compacted, modified = collapse.compact(messages)
    assert modified is True
    # The 4 consecutive tool/AI messages should be collapsed into 1 SystemMessage
    assert len(compacted) == 3
    assert "[Context Collapsed: 4 research steps" in compacted[1].content

def test_reactive_compactor():
    reactive = ReactiveCompactor(slice_ratio=0.20)
    messages = [
        SystemMessage(content="System Rules"),
        HumanMessage(content="Turn 1"),
        AIMessage(content="Ans 1"),
        HumanMessage(content="Turn 2"),
        AIMessage(content="Ans 2"),
        HumanMessage(content="Turn 3"),
        AIMessage(content="Ans 3"),
        HumanMessage(content="Turn 4")
    ]
    
    overflow_result = reactive.handle_overflow(messages)
    assert len(overflow_result) < len(messages)
    assert "[Reactive Compact (Silent Withholding)]" in overflow_result[1].content

def test_autocompactor_circuit_breaker():
    class FailingLLM:
        def invoke(self, messages):
            raise RuntimeError("API Timeout / Error")
        def get_num_tokens_from_messages(self, messages):
            return 10000

    compactor = AutoCompactor(llm=FailingLLM(), threshold_tokens=100, max_consecutive_failures=3)
    messages = [
        SystemMessage(content="System Rules"),
        HumanMessage(content="Hello"),
        AIMessage(content="Hi"),
        HumanMessage(content="Are you there?")
    ]

    # 1st failure
    compactor.compact_if_needed(messages, force=True)
    assert compactor.consecutive_failures == 1
    assert not compactor.is_circuit_open

    # 2nd failure
    compactor.compact_if_needed(messages, force=True)
    assert compactor.consecutive_failures == 2
    assert not compactor.is_circuit_open

    # 3rd failure -> trips circuit breaker
    compactor.compact_if_needed(messages, force=True)
    assert compactor.consecutive_failures == 3
    assert compactor.is_circuit_open

    # 4th call: circuit breaker prevents any LLM call!
    res_messages, was_compacted = compactor.compact_if_needed(messages, force=True)
    assert was_compacted is False
    assert res_messages == messages

    # Test reset
    compactor.reset_circuit_breaker()
    assert compactor.consecutive_failures == 0
    assert not compactor.is_circuit_open

def test_context_collapse_unsafe_tools():
    collapse = ContextCollapse(min_consecutive=3)
    # Includes an unsafe tool (write_to_file / file_writer)
    messages = [
        SystemMessage(content="System Rules"),
        AIMessage(content="Writing file...", tool_calls=[{"name": "file_writer", "args": {}, "id": "1"}]),
        ToolMessage(content="Written", tool_call_id="1", name="file_writer"),
        AIMessage(content="Checking...", tool_calls=[{"name": "file_read", "args": {}, "id": "2"}]),
        ToolMessage(content="Checked", tool_call_id="2", name="file_read"),
        HumanMessage(content="Status?")
    ]
    # Should NOT collapse because file_writer is in UNSAFE_TO_COLLAPSE_TOOLS
    compacted, modified = collapse.compact(messages)
    assert modified is False
    assert len(compacted) == len(messages)

