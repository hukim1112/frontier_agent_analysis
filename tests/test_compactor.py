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
