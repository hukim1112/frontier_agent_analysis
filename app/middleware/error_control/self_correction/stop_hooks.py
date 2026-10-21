"""
===============================================================================
[Error Control / Self-Correction] Stop Hooks Middleware
===============================================================================
LLM의 출력 품질을 검증하고, 결함 발견 시 피드백을 주입하여
에이전트가 스스로 버그를 수정하도록 유도하는 인루프 자가 교정 미들웨어.

Claude Code의 StopHook 아키텍처 구현:
  - 에이전트 응답 내 Python 코드 블록(```python / ```py)을 compile()로 문법 검증
  - 검증 실패 시 blockingError 피드백을 HumanMessage로 주입
  - jump_to="model"로 모델 노드에 되돌려 보내 수정된 코드를 재생성 (Self-Repair Loop)
  - max_retries 회 연속 실패 시 루프를 종료하여 무한 루프 방지

Claude Code ↔ LangChain 대응:
  - Claude Code: query.ts가 `transition: stop_hook_blocking`으로 루프를 계속 진행
  - LangChain  : after_agent 훅에 @hook_config(can_jump_to=["model"])를 선언하고
                 {"jump_to": "model"}을 반환해야 그래프가 END 대신 모델 노드로 복귀
                 (데코레이터가 없으면 after_agent → END 엣지가 무조건 연결됨)

참조:
  - Claude Code: query.ts → stop_hook_blocking transition
===============================================================================
"""

import re
from typing import Any, Optional

from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents.middleware import AgentMiddleware, hook_config

from app.utils.message_markers import (
    ORIGIN_STOP_HOOK,
    STOP_HOOK_TAG,  # 주입되는 blockingError 메시지 식별 태그 (로그 표기 및 구버전 메시지 판별)
    is_stop_hook_feedback,
    is_user_turn_start,
    origin_kwargs,
)

# 언어 태그가 python / python3 / py 인 펜스 코드 블록만 검증 대상으로 삼습니다.
# (mermaid / sql / html / bash 및 태그 없는 블록(ASCII 다이어그램 등)은 오탐 방지를 위해 제외)
_PYTHON_FENCE_RE = re.compile(r"```[ \t]*(?:python3?|py)[ \t]*\r?\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code_blocks(text: str) -> list[str]:
    """텍스트에서 Python 언어 태그(python/python3/py)가 붙은 코드 블록만 추출합니다."""
    return [block.strip("\n") for block in _PYTHON_FENCE_RE.findall(text or "")]


def _strip_interactive_syntax(code_str: str) -> str:
    """REPL 프롬프트(>>> / ...)와 IPython 매직(!pip, %timeit)을 제거하여 순수 Python 코드만 남깁니다."""
    lines = code_str.splitlines()
    if any(line.lstrip().startswith(">>>") for line in lines):
        repl_lines = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith((">>> ", "... ")):
                repl_lines.append(stripped[4:])
            elif stripped in (">>>", "..."):
                repl_lines.append("")
            # 그 외 줄은 REPL 출력 결과이므로 제외
        lines = repl_lines
    return "\n".join("" if line.lstrip().startswith(("!", "%")) else line for line in lines)


def check_python_syntax(code_str: str) -> Optional[str]:
    """기본 검증기: Python 문법 오류 메시지 또는 None을 반환합니다."""
    try:
        compile(_strip_interactive_syntax(code_str), "<string>", "exec")
        return None
    except SyntaxError as e:
        return f"SyntaxError at line {e.lineno}: {e.msg} -> '{e.text.strip() if e.text else ''}'"
    except Exception as e:
        return f"Code validation error: {e}"


def is_stop_hook_message(message: Any) -> bool:
    """StopHooksMiddleware가 주입한 blockingError HumanMessage인지 판별합니다."""
    return is_stop_hook_feedback(message)


# =============================================================================
# StopHooksMiddleware (Self-Correction via Output Validation)
# =============================================================================
class StopHooksMiddleware(AgentMiddleware):
    """에이전트 응답 내 코드 블록을 검증하고, 실패 시 수정 피드백을 주입해 모델로 되돌립니다.

    검증기(Validator)를 플러그인 방식으로 확장할 수 있습니다.
    기본값은 Python 문법(Syntax) 검증기입니다.

    Args:
        validators: 검증기 리스트. 각 항목은 {"name": str, "fn": Callable} 딕셔너리.
                    fn은 코드 문자열을 받아 에러 메시지(str) 또는 None을 반환합니다.
        max_retries: 한 사용자 턴 안에서 허용할 연속 자가 수정 횟수 (기본 2회).
                     한도에 도달하면 더 이상 되돌리지 않고 턴을 종료합니다.

    Returns (after_agent):
        - 검증 실패: {"messages": [blockingError HumanMessage], "jump_to": "model"}
        - 검증 통과 / 코드 없음 / 재시도 한도 도달: None (정상 종료)

    Example:
        ```python
        from app.middleware.error_control.self_correction import StopHooksMiddleware
        from app.middleware.error_control.self_correction.stop_hooks import check_python_syntax

        # 기본 Python 문법 검증
        middleware = StopHooksMiddleware()

        # 커스텀 검증기 추가
        def check_no_eval(code: str) -> str | None:
            if "eval(" in code:
                return "Security: eval() 사용 금지"
            return None

        middleware = StopHooksMiddleware(validators=[
            {"name": "python_syntax", "fn": check_python_syntax},
            {"name": "no_eval", "fn": check_no_eval},
        ])
        ```
    """

    def __init__(self, validators: Optional[list[dict[str, Any]]] = None, max_retries: int = 2):
        if validators is None:
            self.validators = [{"name": "python_syntax", "fn": check_python_syntax}]
        else:
            self.validators = validators
        self.max_retries = max_retries

    @staticmethod
    def _count_consecutive_blocks(messages: list) -> int:
        """현재 사용자 턴에서 이미 주입된 blockingError 개수를 셉니다 (실제 사용자 메시지를 만나면 중단)."""
        count = 0
        for msg in reversed(messages):
            if is_user_turn_start(msg):
                break
            if is_stop_hook_message(msg):
                count += 1
        return count

    def _validate(self, content_str: str) -> list[str]:
        validation_errors = []
        for code in extract_code_blocks(content_str):
            for val in self.validators:
                val_name = val.get("name", "unknown")
                val_fn = val.get("fn")
                if val_fn:
                    try:
                        error = val_fn(code)
                        if error:
                            validation_errors.append(f"[{val_name}]: {error}")
                    except Exception as e:
                        validation_errors.append(f"[{val_name}] execution error: {e}")
        return validation_errors

    @hook_config(can_jump_to=["model"])
    def after_agent(self, state: dict, runtime=None) -> dict | None:
        messages = state.get("messages", [])
        if not messages:
            return None

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage) or not last_msg.content:
            return None

        from app.utils.message_utils import normalize_content
        validation_errors = self._validate(normalize_content(last_msg.content))
        if not validation_errors:
            return None

        retries = self._count_consecutive_blocks(messages)
        if retries >= self.max_retries:
            print(f"\n⚠️ {STOP_HOOK_TAG} Retry limit reached ({retries}/{self.max_retries}) -> Ending turn without further self-repair.")
            return None

        error_details = "\n".join(validation_errors)
        correction_prompt = (
            f"🛑 {STOP_HOOK_TAG} Code Validation Failed (self-repair {retries + 1}/{self.max_retries})\n"
            f"The code you generated failed quality validation tests:\n"
            f"{error_details}\n\n"
            f"Please analyze the error details above, fix the bug in your code, and output the corrected version."
        )
        print(f"\n🔴 {STOP_HOOK_TAG} Validation Failed -> Injecting blockingError & jumping back to model ({retries + 1}/{self.max_retries})")
        # messages 채널은 add_messages 리듀서이므로 새 메시지만 반환하면 기존 이력 뒤에 추가됩니다.
        # harness_origin 표식: 사용자 턴 경계로 세지 않도록 (app.utils.message_markers 규칙)
        feedback = HumanMessage(content=correction_prompt, additional_kwargs=origin_kwargs(ORIGIN_STOP_HOOK))
        return {"messages": [feedback], "jump_to": "model"}
