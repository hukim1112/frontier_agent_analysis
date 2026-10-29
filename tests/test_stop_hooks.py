from types import SimpleNamespace

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from app.middleware.error_control import (
    StopHooksMiddleware,
    AbortStreamingMiddleware,
    AbortToolsMiddleware,
)

def test_stop_hook_blocking_triggers_on_syntax_error():
    middleware = StopHooksMiddleware()
    broken_code_msg = AIMessage(
        content="이것은 테스트용 잘못된 파이썬 코드입니다:\n```python\ndef bad_func(\n    print('Unclosed parenthesis')\n```"
    )
    state = {"messages": [HumanMessage(content="함수 작성해줘"), broken_code_msg]}
    result = middleware.after_agent(state)
    
    assert result is not None
    # Claude Code의 stop_hook_blocking 전이 → LangChain에서는 jump_to="model"로 모델 노드 복귀
    assert result.get("jump_to") == "model"
    # add_messages 리듀서: 새로 주입할 blockingError 1건만 반환
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], HumanMessage)
    assert "[stop_hook_blocking]" in result["messages"][0].content
    assert "SyntaxError" in result["messages"][-1].content

def test_stop_hook_passes_on_valid_code():
    middleware = StopHooksMiddleware()
    valid_code_msg = AIMessage(
        content="이것은 올바른 파이썬 코드입니다:\n```python\ndef good_func():\n    return 42\n```"
    )
    state = {"messages": [HumanMessage(content="함수 작성해줘"), valid_code_msg]}
    result = middleware.after_agent(state)
    
    # 검증 통과(completed) → 상태 변경 없이 정상 종료
    assert result is None

def test_aborted_streaming():
    def interrupted(request):
        raise KeyboardInterrupt("Ctrl+C")

    response = AbortStreamingMiddleware().wrap_model_call(None, interrupted)
    assert isinstance(response.result[-1], AIMessage)
    assert "aborted_streaming" in response.result[-1].content

def test_aborted_tools():
    def interrupted(request):
        raise KeyboardInterrupt("Ctrl+C")

    request = SimpleNamespace(tool_call={"name": "bash_command", "args": {}, "id": "call_1"})
    result = AbortToolsMiddleware().wrap_tool_call(request, interrupted)
    assert isinstance(result, ToolMessage)
    assert result.tool_call_id == "call_1"
    assert "aborted_tools" in result.content
