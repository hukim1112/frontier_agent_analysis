"""
app/utils/langchain_llm.py
==========================
LangChain Universal LLM & Chat Model Factory

LangChain 표준 `init_chat_model`의 인터페이스를 유지하면서,
OpenAI, Anthropic, Google Gemini(Vertex AI / AI Studio), Ollama 등 다양한 LLM 공급자를
환경 변수 설정에 따라 유연하게 초기화하는 팩토리 모듈입니다.

[Google Gemini 공급자 자동 분기 우선순위]
1. `VERTEX_PROJECT` (또는 `GOOGLE_CLOUD_PROJECT`) 설정 시 ➔ Vertex AI (`google_vertexai`, location='global')
2. `GOOGLE_API_KEY` 설정 시 ➔ Google AI Studio (`google_genai`)
"""

import os
import re
from typing import Optional, Any
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model as _native_init_chat_model

# .env 환경변수 자동 로드
load_dotenv()

# OpenAI 추론 모델 패턴 (GPT-5 이후 세대 + o-시리즈): Responses API 필수, temperature 미지원
_OPENAI_REASONING_MODEL_RE = re.compile(r"^(gpt-([5-9]|\d{2,})|o\d)")


def init_chat_model(
    model: str = "gemini-3.8-flash",
    model_provider: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs: Any
):
    """
    LangChain 네이티브 init_chat_model의 모든 기능과 시그니처를 지원하며,
    Google Gemini 모델 지정 시 환경 변수에 따라 google_vertexai(1순위)와 
    google_genai(2순위)를 자동으로 라우팅합니다.

    Args:
        model: 모델 식별자 (예: 'gemini-3.8-flash', 'gpt-6.1-sol', 'claude-3-5-sonnet')
        model_provider: 명시적 공급자 ('openai', 'anthropic', 'google_vertexai', 'google_genai' 등). 생략 시 자동 감지.
        temperature: 샘플링 온도 (기본값: 0.0). OpenAI 추론 모델(GPT-5 이후/o-시리즈)에는 전달되지 않습니다.
        **kwargs: 공급자별 추가 파라미터 (max_tokens, streaming, project, location 등)

    Returns:
        BaseChatModel 인스턴스 (ChatOpenAI, ChatAnthropic, ChatVertexAI, ChatGoogleGenerativeAI 등)
    """
    # 1. 'provider:model' 접두사 파싱 지원 (예: 'openai:gpt-4o', 'google_vertexai:gemini-3.8-flash')
    clean_model_name = model
    parsed_provider = None
    if ":" in model:
        parts = model.split(":", 1)
        parsed_provider = parts[0]
        clean_model_name = parts[1]

    target_provider = model_provider or parsed_provider

    # 2. Google Gemini 모델 계열 자동 감지 및 Provider 라우팅 (Vertex AI 우선)
    is_gemini = (
        "gemini" in clean_model_name.lower()
        or target_provider in ["google", "gemini", "google_genai", "google_vertexai"]
    )

    if is_gemini and (target_provider is None or target_provider in ["google", "gemini"]):
        vertex_project = (
            os.getenv("VERTEX_PROJECT")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCP_PROJECT")
        )
        
        # 1순위: Vertex AI 프로젝트 설정이 있는 경우 -> Vertex AI 우선 사용
        if vertex_project:
            target_provider = "google_vertexai"
        # 2순위: GOOGLE_API_KEY만 있는 경우 -> Google AI Studio 사용
        elif os.getenv("GOOGLE_API_KEY"):
            target_provider = "google_genai"
        else:
            target_provider = "google_vertexai" if vertex_project else "google_genai"

    # 3. Vertex AI 공급자인 경우 GCP 파라미터(project, location) 기본값 자동 보정
    if target_provider == "google_vertexai":
        project = (
            os.getenv("VERTEX_PROJECT")
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCP_PROJECT")
        )
        location = (
            os.getenv("VERTEX_LOCATION")
            or os.getenv("GOOGLE_CLOUD_LOCATION")
            or "global"
        )

        if project and "project" not in kwargs:
            kwargs["project"] = project
        if location and "location" not in kwargs:
            kwargs["location"] = location

    # 4. OpenAI 추론 모델(GPT-5 이후 세대 / o-시리즈, 예: gpt-6.1-sol) 호환 처리
    #    - Chat Completions에서는 추론 모델 + function tools 조합이 거부됨 → Responses API 사용
    #    - temperature 파라미터를 지원하지 않음 → 전달하지 않음
    is_openai = target_provider in (None, "openai") and not is_gemini
    if is_openai and _OPENAI_REASONING_MODEL_RE.match(clean_model_name.lower()):
        kwargs.setdefault("use_responses_api", True)
        return _native_init_chat_model(
            model=clean_model_name,
            model_provider="openai",
            **kwargs
        )

    # 5. LangChain 네이티브 init_chat_model 호출
    return _native_init_chat_model(
        model=clean_model_name,
        model_provider=target_provider,
        temperature=temperature,
        **kwargs
    )


def get_embeddings(model: str = "gemini-embedding-001", **kwargs: Any):
    """
    환경 변수 설정(Vertex AI 1순위, AI Studio 2순위)에 따라 표준 임베딩 인스턴스를 반환합니다.

    Args:
        model: 임베딩 모델명 (기본값: 'gemini-embedding-001')
        **kwargs: 임베딩 클라이언트 추가 옵션

    Returns:
        Embeddings 인스턴스 (VertexAIEmbeddings 또는 GoogleGenerativeAIEmbeddings)
    """
    vertex_project = (
        os.getenv("VERTEX_PROJECT")
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCP_PROJECT")
    )

    # 1순위: Vertex AI 프로젝트 설정이 있는 경우
    if vertex_project:
        from langchain_google_vertexai import VertexAIEmbeddings
        location = (
            os.getenv("VERTEX_LOCATION")
            or os.getenv("GOOGLE_CLOUD_LOCATION")
            or "global"
        )
        clean_model = model.replace("models/", "")
        return VertexAIEmbeddings(
            model_name=clean_model,
            project=vertex_project,
            location=location,
            **kwargs
        )
    # 2순위: GOOGLE_API_KEY만 있는 경우
    elif os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model_name = f"models/{model}" if not model.startswith("models/") else model
        return GoogleGenerativeAIEmbeddings(model=model_name, **kwargs)
    else:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        model_name = f"models/{model}" if not model.startswith("models/") else model
        return GoogleGenerativeAIEmbeddings(model=model_name, **kwargs)
