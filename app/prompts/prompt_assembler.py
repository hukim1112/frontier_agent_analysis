"""
범용 4-Layer PromptAssembler — app/prompts/ 이식 버전.

modules/hermes/prompt_assembler.py에서 이식.
import 경로만 변경, 로직 동일.
"""

from modules.hermes.prompt_assembler import PromptAssembler, create_prompt_middleware

__all__ = ["PromptAssembler", "create_prompt_middleware"]
