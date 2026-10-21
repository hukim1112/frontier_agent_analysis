from typing import Any, Optional
from app.utils.langchain_wrapper import init_chat_model


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    **kwargs: Any,
):
    """Backward-compatible helper delegating to init_chat_model."""
    model = model_name or "gemini-3.8-flash"
    return init_chat_model(model=model, temperature=temperature, max_tokens=max_tokens, **kwargs)
