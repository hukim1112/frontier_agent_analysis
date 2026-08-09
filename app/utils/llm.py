import os
from typing import Optional
from langchain.chat_models import init_chat_model

def get_llm(
    model_name: str = "openai:gpt-4o-mini",
    model_provider: Optional[str] = None,
    temperature: float = 0.1,
    **kwargs
):
    """
    OpenAI, Anthropic, Google GenAI, Vertex AI, Ollama 등 
    모든 LLM 공급자를 유연하게 지원하는 중앙 LLM 팩토리 함수입니다.
    
    [Provider 결정 우선순위]
    1. model_name에 접두사 지정 시 (예: "openai:gpt-4o", "anthropic:claude-3-5-sonnet", "google_vertexai:gemini-2.5-pro")
    2. model_provider 파라미터를 명시적으로 전달받았을 때 (예: model_provider="openai")
    3. .env 환경변수에 MODEL_PROVIDER 또는 LLM_PROVIDER 가 지정되어 있을 때
    4. .env에 VERTEX_PROJECT 가 설정되어 있으면 -> "google_vertexai"
    5. 기본 fallback -> "google_genai"
    """
    # 1. model_name에서 'provider:model' 형태 파싱
    parsed_provider = None
    clean_model_name = model_name
    if ":" in model_name:
        parts = model_name.split(":", 1)
        parsed_provider = parts[0]
        clean_model_name = parts[1]

    # 최종 Provider 결정
    final_provider = (
        model_provider 
        or parsed_provider 
        or os.getenv("MODEL_PROVIDER") 
        or os.getenv("LLM_PROVIDER")
    )
    
    # 구글 모델(Gemini)인 경우, 환경 변수 유무에 따라 google_vertexai와 google_genai를 동적으로 라우팅합니다.
    is_google_model = (
        clean_model_name.startswith("gemini-") 
        or final_provider in ["google_vertexai", "google_genai"]
    )
    
    if is_google_model:
        has_vertex_project = bool(os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT"))
        has_google_api_key = bool(os.getenv("GOOGLE_API_KEY"))
        
        if has_vertex_project:
            final_provider = "google_vertexai"
        elif has_google_api_key:
            final_provider = "google_genai"
        else:
            final_provider = "google_vertexai" if has_vertex_project else "google_genai"
            
    if not final_provider:
        final_provider = "google_genai"

    # 2. Vertex AI인 경우에만 GCP 특화 옵션 (project, location) 자동 보정
    if final_provider == "google_vertexai":
        project = os.getenv("VERTEX_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
        location = os.getenv("VERTEX_LOCATION") or os.getenv("GOOGLE_CLOUD_LOCATION") or "global"
        
        if project and "project" not in kwargs:
            kwargs["project"] = project
        if location and "location" not in kwargs:
            kwargs["location"] = location

    # 3. LangChain 공통 init_chat_model 호출
    return init_chat_model(
        model=clean_model_name,
        model_provider=final_provider,
        temperature=temperature,
        **kwargs
    )
