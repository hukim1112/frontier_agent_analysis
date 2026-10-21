"""
===============================================================================
[Error Control / Self-Recovery] Abort Handlers Middleware
===============================================================================
사용자 인터럽트(Ctrl+C, Abort Signal) 발생 시 에이전트 크래시 대신
안전한 종료 메시지를 반환하는 그레이스풀 셧다운 미들웨어.

1. AbortStreamingMiddleware: LLM 토큰 스트리밍 도중 사용자 중단 처리
2. AbortToolsMiddleware: 장시간 도구 실행 도중 사용자 중단 처리

참조:
  - Claude Code: query.ts → aborted_streaming / aborted_tools transition
===============================================================================
"""

from langchain_core.messages import AIMessage, ToolMessage
from langchain.agents.middleware import AgentMiddleware, ModelResponse


def _is_user_abort(error: Exception) -> bool:
    """사용자 인터럽트 시그널인지 판별합니다."""
    if isinstance(error, KeyboardInterrupt):
        return True
    error_str = str(error).lower()
    return any(kw in error_str for kw in ("abort", "cancelled", "interrupted"))


# =============================================================================
# 1. AbortStreamingMiddleware (LLM 응답 스트리밍 중단)
# =============================================================================
class AbortStreamingMiddleware(AgentMiddleware):
    """LLM 토큰 스트리밍 도중 사용자 Ctrl+C / Abort 시그널을 안전하게 처리합니다."""

    def wrap_model_call(self, request, handler):
        try:
            return handler(request)
        except (KeyboardInterrupt, Exception) as error:
            if _is_user_abort(error):
                print("\n🛑 [aborted_streaming] User pressed Ctrl+C / Abort signal during streaming.")
                abort_notice = AIMessage(
                    content="🛑 [aborted_streaming] 사용자가 스트리밍 응답 생성을 취소(Ctrl+C / Abort)했습니다."
                )
                return ModelResponse(result=[abort_notice])
            raise error

    async def awrap_model_call(self, request, handler):
        try:
            return await handler(request)
        except (KeyboardInterrupt, Exception) as error:
            if _is_user_abort(error):
                print("\n🛑 [aborted_streaming] User pressed Ctrl+C / Abort signal during streaming.")
                abort_notice = AIMessage(
                    content="🛑 [aborted_streaming] 사용자가 스트리밍 응답 생성을 취소(Ctrl+C / Abort)했습니다."
                )
                return ModelResponse(result=[abort_notice])
            raise error


# =============================================================================
# 2. AbortToolsMiddleware (도구 실행 중 사용자 중단)
# =============================================================================
class AbortToolsMiddleware(AgentMiddleware):
    """장시간 도구(bash_command, 웹 스크래핑 등) 실행 도중 사용자 인터럽트를 안전하게 처리합니다."""

    def _extract_tool_info(self, request):
        """ToolRequest에서 도구 이름과 ID를 안전하게 추출합니다."""
        tool_name = "unknown_tool"
        tool_call_id = "abort_call_001"
        if hasattr(request, "tool_call") and isinstance(request.tool_call, dict):
            tool_name = request.tool_call.get("name", tool_name)
            tool_call_id = request.tool_call.get("id", tool_call_id)
        elif hasattr(request, "name"):
            tool_name = getattr(request, "name", tool_name)
        return tool_name, tool_call_id

    def wrap_tool_call(self, request, handler):
        tool_name, tool_call_id = self._extract_tool_info(request)
        try:
            return handler(request)
        except (KeyboardInterrupt, Exception) as error:
            if _is_user_abort(error):
                print(f"\n🛑 [aborted_tools] User interrupted long-running tool '{tool_name}'.")
                return ToolMessage(
                    content=f"[aborted_tools] '{tool_name}' 도구 실행 도중 사용자 중단 시그널을 수신하여 실행을 취소했습니다.",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            raise error

    async def awrap_tool_call(self, request, handler):
        tool_name, tool_call_id = self._extract_tool_info(request)
        try:
            return await handler(request)
        except (KeyboardInterrupt, Exception) as error:
            if _is_user_abort(error):
                print(f"\n🛑 [aborted_tools] User interrupted long-running tool '{tool_name}'.")
                return ToolMessage(
                    content=f"[aborted_tools] '{tool_name}' 도구 실행 도중 사용자 중단 시그널을 수신하여 실행을 취소했습니다.",
                    tool_call_id=tool_call_id,
                    name=tool_name,
                )
            raise error
