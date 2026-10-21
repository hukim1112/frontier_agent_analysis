---
name: collection-validation
description: Diagnose shifted rows, missing fields and duplicate records, with a row-scoped extraction example and an optional validator for JSON or JSONL collections.
---

# 수집 결과에서 누락과 행 불일치 찾기

## 사례 1: 제목과 가격 배열을 합쳤더니 다른 상품끼리 묶임

제목 3개와 가격 2개를 각각 수집해 zip하면 가격이 없는 상품 다음부터 값이 어긋날 수 있다. 각 항목의 컨테이너 안에서 필드를 함께 읽으면 빈 필드를 유지한 채 행의 관계를 보존할 수 있다.

아래는 `article.product[data-id]` 안에 `.title`, `.price`, `a.detail`이 있는 HTML의 예제다. 필요한 선택자를 실제 구조로 바꾼다. 가격은 원문 문자열로 유지하므로 통화·소수점·할인 조건에 맞는 숫자 변환은 이후 별도로 할 수 있다.

```python
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def parse_product_cards(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    records = []
    for card in soup.select("article.product"):
        title, price, link = (card.select_one(selector)
                              for selector in (".title", ".price", "a.detail"))
        href = link.get("href") if link else None
        records.append({
            "id": card.get("data-id"),
            "title": title.get_text(" ", strip=True) if title else None,
            "price_text": price.get_text(" ", strip=True) if price else None,
            "url": urljoin(base_url, href) if href else None,
        })
    return records
```

두 번째 상품에 가격이 없으면 그 행의 `price_text`는 `None`이고, 세 번째 상품의 가격을 대신 가져오지 않는다. 이 결과를 필수 필드 검사에 넣으면 문제 위치를 찾을 수 있다. 실제 값이 frame·Shadow DOM에 있다면 HTML 파싱 대신 [해당 영역을 읽는 방식](dynamic_content_diagnosis.md)을 선택한다.

## 사례 2: JSON 파일은 생성됐지만 중복·결측이 있음

반복 검사에는 기존 보조 스크립트를 선택적으로 사용할 수 있다. 자체 코드나 다른 검증 도구가 같은 요구사항을 확인한다면 그것을 사용해도 된다. 아래 필드명은 위 예제 기준이며 실제 output_schema에 맞게 바꾼다.

```bash
python app/agents/scraper/skills/scripts/validate_collection.py artifacts/scrape/items.json --required id title price_text --unique-key id --type id:string --type price_text:string --report artifacts/scrape/quality.json
```

| 관찰한 보고 값 | 해석과 다음 확인 |
|---|---|
| missing_required:price_text | 해당 행의 원본에 가격이 없는지, 선택자가 맞는지 확인 |
| duplicate_count > 0 | 페이지 반복인지, 옵션 상품을 하나의 ID로 합친 것인지 확인; 후자는 id+variant 같은 복합 키 검토 |
| invalid_type | 숫자 문자열·빈 값·오류 문구 등이 섞였는지 표본 확인 |
| coverage=not_established | 파일 품질만 검사했으며 전체 범위를 확인한 것은 아님 |

`--unique-key id variant`로 복합 키를 지정할 수 있다. `--expected-count`는 같은 필터·기간·집계 단위의 총건수를 알고 있을 때 유용하다. 정상 빈 결과가 확인된 경우에는 `--min-count 0 --expected-count 0`처럼 의도를 표현할 수 있다. 숫자 0과 false는 결측으로 취급하지 않는다.

스크립트는 입력 파일을 변경하지 않는다. 종료 코드 0은 지정한 파일 검사 통과, 1은 품질 실패, 2는 입력 오류다. 최상위 필드 타입을 검사하며, 통화 단위나 중첩 데이터의 의미까지 자동 검증하지는 않는다.

## 사례 3: 건수는 맞는데 요청한 범위와 다름

총건수가 같아도 다른 검색 필터, 기본 정렬, 첫 페이지 반복, 수집 중 원본 변경으로 값이 달라질 수 있다. 첫·중간·마지막 구간의 ID와 주요 필드를 실제 화면 또는 원본 응답과 대조하면 이런 문제를 찾는 데 도움이 된다.

전체 수집은 확인된 마지막 페이지/종료 토큰과 대조하고, N건 수집은 요청한 조건의 고유 N건인지 확인한다. [재개 예제](pagination_and_resume.md)의 `done=False`나 오류로 중단된 실행은 파일이 있더라도 부분 수집이다. 데이터와 함께 수집 범위·필터·시각·검증 결과·미완료 이유를 남기면 다음 작업에서 해석할 수 있다.
