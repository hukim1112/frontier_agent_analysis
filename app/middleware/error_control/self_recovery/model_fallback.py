"""
===============================================================================
[Error Control / Self-Recovery] Model Fallback Middleware
===============================================================================
프로덕션 에이전트의 LLM 통신 장애 대응 통합 미들웨어.

동작 흐름 (3-Phase Cascade):
  Phase 1: 주 모델 호출 → 일시적 에러 시 지수 백오프 재시도 (max_retries회)
  Phase 2: 주 모델 재시도 소진 → 백업 모델(fallback)로 투명 전환
  Phase 3: 백업 모델도 실패 → 에이전트 크래시 대신 안전 메시지 반환

참조:
  - Claude Code: query.ts → model_error / reactive_compact_retry transition
  - 기존 agent_lab: ModelErrorHandlerMiddleware + ModelFallbackMiddleware (분리형)
  - 기존 harness_agent: retry_on_transient_error + dynamic_model_fallback (분리형)
===============================================================================
"""

import time
from typing import Any, Optional

from langchain_core.messages import AIMessage
from langchain.agents.middleware import AgentMiddleware, ModelResponse


# =============================================================================
# Retryable Error Detection
# =============================================================================
# 재시도할 가치가 있는 일시적(Transient) 에러만 필터링합니다.
# AuthenticationError, InvalidRequestError 등 영구적 에러는 재시도해도 무의미합니다.

RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    ConnectionResetError,
    TimeoutError,
    OSError,
)

RETRYABLE_STATUS_KEYWORDS = (
    "429", "500", "502", "503", "504",
    "rate_limit", "rate limit",
    "overloaded", "temporarily unavailable",
    "service unavailable", "server error",
    "connection reset", "timeout",
    "resource_exhausted", "resource exhausted",
)

CONTEXT_OVERFLOW_KEYWORDS = (
    "413",
    "prompt_too_long",
    "context_length_exceeded",
    "too many tokens",
    "maximum context length",
)


def _is_context_overflow(error: Exception) -> bool:
    """에러가 컨텍스트 길이 초과(413 / context_length_exceeded) 에러인지 판별합니다.
    이 에러는 백업 모델 전환이나 안전 메시지로 삼키지 않고 상위의 CompactorMiddleware(ReactiveCompactor)로 전파합니다."""
    error_str = str(error).lower()
    return any(kw in error_str for kw in CONTEXT_OVERFLOW_KEYWORDS)


def _is_retryable(error: Exception) -> bool:
    """에러가 재시도할 가치가 있는 일시적(Transient) 에러인지 판별합니다."""
    if _is_context_overflow(error):
        return False
    if isinstance(error, RETRYABLE_EXCEPTIONS):
        return True
    error_str = str(error).lower()
    return any(kw in error_str for kw in RETRYABLE_STATUS_KEYWORDS)


# =============================================================================
# ModelFallbackMiddleware (Retry + Fallback + Safe Failure 통합)
# =============================================================================
class ModelFallbackMiddleware(AgentMiddleware):
    """LLM API 통신 장애 대응 통합 미들웨어.

    3-Phase Cascade 패턴으로 에이전트의 무중단 운영을 보장합니다:
      Phase 1: 주 모델 재시도 (Exponential Backoff)
      Phase 2: 백업 모델 전환 (Dynamic Failover)
      Phase 3: 안전 종료 메시지 (Graceful Degradation)

    Args:
        max_retries: 주 모델 재시도 최대 횟수 (기본 3).
        initial_delay: 첫 번째 재시도 전 대기 시간(초) (기본 0.5).
        fallback_model_name: 백업 모델 이름 (기본 "gpt-6.1-sol").
        fallback_llm: 사전 생성된 백업 LLM 인스턴스. None이면 lazy init.

    Example:
        ```python
        from app.middleware.error_control.self_recovery import ModelFallbackMiddleware

        middleware = ModelFallbackMiddleware(
            max_retries=3,
            fallback_model_name="gpt-6.1-sol"
        )
        agent = create_agent(model=llm, tools=tools, middleware=[middleware])
        ```
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 0.5,
        fallback_model_name: str = "gpt-6.1-sol",
        fallback_llm: Optional[Any] = None,
    ):
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.fallback_model_name = fallback_model_name
        self._fallback_llm = fallback_llm

    def _get_fallback_llm(self):
        """백업 LLM 인스턴스를 반환합니다 (Lazy Initialization)."""
        if self._fallback_llm is not None:
            return self._fallback_llm
        from app.utils.langchain_wrapper import init_chat_model
        self._fallback_llm = init_chat_model(model=self.fallback_model_name, temperature=0.0)
        return self._fallback_llm

    def _make_safe_message(self, primary_error, fallback_error=None) -> ModelResponse:
        """모든 모델이 실패했을 때 에이전트 크래시 대신 안전 메시지를 반환합니다."""
        if fallback_error:
            content = (
                f"⚠️ [Model Fallback 실패] 주 모델과 백업 모델({self.fallback_model_name}) 모두 응답 불가 상태입니다.\n"
                f"주 모델 에러: {primary_error}\n"
                f"백업 모델 에러: {fallback_error}\n"
                f"네트워크 연결 상태 및 API 키를 점검해 주세요."
            )
        else:
            content = (
                f"⚠️ [Model Error] LLM API 통신 장애가 발생했습니다.\n"
                f"에러: {primary_error}\n"
                f"네트워크 연결 상태 및 API 키를 점검해 주세요."
            )
        return ModelResponse(result=[AIMessage(content=content)])

    @staticmethod
    def _with_fallback_model(request: Any, fallback_llm: Any) -> Any:
        """request.override(model=...)로 새 요청을 만듭니다 (원본 요청은 바꾸지 않음).
        override를 지원하지 않는 요청 객체(테스트용 더미 등)에서만 속성을 직접 갱신합니다."""
        if hasattr(request, "override") and callable(request.override):
            try:
                return request.override(model=fallback_llm)
            except Exception:
                pass
        try:
            request.model = fallback_llm
        except Exception:
            pass
        return request

    # ---- Sync ----
    def wrap_model_call(self, request, handler):
        """동기 모델 호출: Retry → Fallback → Safe Message 3단계 복구 (413은 Compactor로 전파)."""
        primary_error = None

        # Phase 1: 주 모델 재시도 (Exponential Backoff)
        for attempt in range(1, self.max_retries + 1):
            try:
                return handler(request)
            except Exception as error:
                if _is_context_overflow(error):
                    raise error
                primary_error = error
                if not _is_retryable(error):
                    print(f"🛑 [ModelFallback] Non-retryable error (attempt {attempt}): {error}")
                    break  # 재시도 불가능한 에러 → 즉시 Phase 2로
                if attempt < self.max_retries:
                    sleep_time = self.initial_delay * (2 ** (attempt - 1))
                    print(f"⚠️ [ModelFallback] Transient error (attempt {attempt}/{self.max_retries}). "
                          f"Retrying in {sleep_time:.1f}s... ({error})")
                    time.sleep(sleep_time)
                else:
                    print(f"🛑 [ModelFallback] All {self.max_retries} retries exhausted for primary model.")

        # Phase 2: 백업 모델 전환 (Dynamic Failover)
        try:
            fallback_llm = self._get_fallback_llm()
            print(f"🔄 [ModelFallback] Switching to backup model '{self.fallback_model_name}'...")
            fallback_req = self._with_fallback_model(request, fallback_llm)
            return handler(fallback_req)
        except Exception as fallback_error:
            if _is_context_overflow(fallback_error):
                raise fallback_error
            print(f"🛑 [ModelFallback] Backup model '{self.fallback_model_name}' also failed: {fallback_error}")
            # Phase 3: 안전 종료 메시지 (Graceful Degradation)
            return self._make_safe_message(primary_error, fallback_error)

    # ---- Async ----
    async def awrap_model_call(self, request, handler):
        """비동기 모델 호출: Retry → Fallback → Safe Message 3단계 복구 (413은 Compactor로 전파)."""
        import asyncio
        primary_error = None

        # Phase 1: 주 모델 재시도 (Exponential Backoff)
        for attempt in range(1, self.max_retries + 1):
            try:
                return await handler(request)
            except Exception as error:
                if _is_context_overflow(error):
                    raise error
                primary_error = error
                if not _is_retryable(error):
                    print(f"🛑 [ModelFallback] Non-retryable error (attempt {attempt}): {error}")
                    break
                if attempt < self.max_retries:
                    sleep_time = self.initial_delay * (2 ** (attempt - 1))
                    print(f"⚠️ [ModelFallback] Transient error (attempt {attempt}/{self.max_retries}). "
                          f"Retrying in {sleep_time:.1f}s... ({error})")
                    await asyncio.sleep(sleep_time)
                else:
                    print(f"🛑 [ModelFallback] All {self.max_retries} retries exhausted for primary model.")

        # Phase 2: 백업 모델 전환 (Dynamic Failover)
        try:
            fallback_llm = self._get_fallback_llm()
            print(f"🔄 [ModelFallback] Switching to backup model '{self.fallback_model_name}'...")
            fallback_req = self._with_fallback_model(request, fallback_llm)
            return await handler(fallback_req)
        except Exception as fallback_error:
            if _is_context_overflow(fallback_error):
                raise fallback_error
            print(f"🛑 [ModelFallback] Backup model '{self.fallback_model_name}' also failed: {fallback_error}")
            return self._make_safe_message(primary_error, fallback_error)
