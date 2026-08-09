import pytest
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from modules.claude_code.self_correction import (
    StopHooksMiddleware,
    ModelErrorHandlerMiddleware,
    AbortStreamingHandler,
    AbortToolsHandler
)

def test_stop_hook_blocking_triggers_on_syntax_error():
    middleware = StopHooksMiddleware()
    broken_code_msg = AIMessage(
        content="이것은 테스트용 잘못된 파이썬 코드입니다:\n```python\ndef bad_func(\n    print('Unclosed parenthesis')\n```"
    )
    state = {"messages": [HumanMessage(content="함수 작성해줘"), broken_code_msg]}
    result = middleware.after_agent(state)
    
    assert result is not None
    assert result.get("transition") == "stop_hook_blocking"
    assert len(result["messages"]) == 3
    assert "SyntaxError" in result["messages"][-1].content

def test_stop_hook_passes_on_valid_code():
    middleware = StopHooksMiddleware()
    valid_code_msg = AIMessage(
        content="이것은 올바른 파이썬 코드입니다:\n```python\ndef good_func():\n    return 42\n```"
    )
    state = {"messages": [HumanMessage(content="함수 작성해줘"), valid_code_msg]}
    result = middleware.after_agent(state)
    
    assert result is not None
    assert result.get("transition") == "completed"

def test_aborted_streaming():
    handler = AbortStreamingHandler()
    msgs = [HumanMessage(content="작업 진행해줘")]
    updated = handler.handle_abort(msgs)
    assert len(updated) == 2
    assert "aborted_streaming" in updated[-1].content

def test_aborted_tools():
    handler = AbortToolsHandler()
    msgs = [HumanMessage(content="터미널 스크립트 실행해줘")]
    updated = handler.handle_tool_abort(msgs, active_tool_name="bash_command")
    assert len(updated) == 2
    assert "aborted_tools" in updated[-1].content
