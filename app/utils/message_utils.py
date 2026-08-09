"""
LLM 모델별 메시지 포맷 차이를 통일하고 깨진 유니코드를 정화하는 유틸리티.

- sanitize_text: UTF-8 직렬화 실패를 유발하는 유니코드 서로게이트(Lone Surrogate, U+D800~U+DFFF)를  문자로 강제 치환
- normalize_content: Gemini 등 리스트 구조체 응답 또는 OpenAI 등의 단일 텍스트를 문자열로 정규화
"""

def sanitize_text(text: str) -> str:
    """서로게이트 문자를 제거하여 JSON 직렬화 오류를 방지합니다.
    
    웹 스크레이핑 데이터나 PDF 추출물에 포함된 잘못된 유니코드 문자를
    정화하여 json.dumps() 시 발생하는 UnicodeEncodeError를 차단합니다.
    """
    if not isinstance(text, str):
        text = str(text)
    return text.encode("utf-8", errors="replace").decode("utf-8")


def normalize_content(content) -> str:
    """다양한 LLM의 content 형식을 단일 문자열로 정규화합니다.
    
    지원하는 형식:
      - str: 그대로 반환 (OpenAI 등)
      - list[dict]: type별로 파싱하여 텍스트 추출 (Gemini 등)
      - list[str]: 줄바꿈으로 결합
      - 기타: str() 변환
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        text_parts.append("[이미지 (Base64)]")
                    else:
                        text_parts.append(f"![image]({url})")
            elif isinstance(item, str):
                text_parts.append(item)
        return "\n".join(text_parts)

    return str(content)
