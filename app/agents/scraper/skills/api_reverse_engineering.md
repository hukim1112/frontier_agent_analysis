---
name: api-discovery
description: Read embedded JSON, capture a matching browser response, and adapt an observed read-only API for collection when DOM extraction is fragile or incomplete.
---

# 내장 JSON과 네트워크 응답을 실제 수집에 연결하기

DOM 수집으로 충분한 경우 그 방법을 유지할 수 있다. 여기서는 화면에 내장된 데이터나 관찰한 API를 활용할 때 쓸 수 있는 레시피를 다룬다. 아래 `page`는 [기존 탭 연결](references/shared_browser.md) 범위 안에서 사용하는 Playwright 페이지다.

## 사례 1: HTML의 JSON에서 초기 목록 추출

`extract_dom_skeleton`과 일반적인 `get_page_section`은 script를 제거한다. 따라서 정제된 출력에 JSON이 없더라도 원본 script를 직접 확인할 가치가 있다.

다음 함수는 지정한 script들을 JSON으로 읽는다. 기본 선택자는 Next.js용이며, 관찰한 페이지에 따라 `script[type="application/ld+json"]` 등으로 바꿀 수 있다. 빈 목록은 일치하는 비어 있지 않은 script가 없다는 뜻이고, 파싱 오류는 그 내용이 정상 JSON이 아니라는 뜻이다.

```python
import json

async def read_embedded_json(page, selector="script#__NEXT_DATA__"):
    texts = await page.locator(selector).all_text_contents()
    return [json.loads(text) for text in texts if text.strip()]
```

실제로 `props.pageProps.items`에 `id/title`이 있는 구조를 확인했다면 다음처럼 필요한 레코드만 꺼낼 수 있다. 이 경로는 사이트마다 다르며 Nuxt의 참조 배열이나 JSON-LD를 같은 경로로 읽을 수 있다는 뜻은 아니다.

```python
async def next_item_rows(page):
    payloads = await read_embedded_json(page)
    if len(payloads) != 1:
        raise ValueError("Next.js 데이터 script를 하나로 확인해야 합니다.")
    items = payloads[0]["props"]["pageProps"]["items"]
    if not isinstance(items, list):
        raise ValueError("관찰한 items 배열과 응답 구조가 다릅니다.")
    return [{"id": item["id"], "title": item["title"]} for item in items]
```

예를 들어 items가 20개인데 화면의 전체 건수가 250이라면 이 결과는 초기 목록이다. 내장 데이터의 존재만으로 추가 페이지까지 수집된 것은 아니다. JSON-LD는 상세 페이지의 일부 필드만 제공할 수도 있다.

## 사례 2: 더보기가 받은 JSON을 그대로 수집

네트워크 후보를 찾을 때는 `page.on("response", handler)`로 응답을 관찰하거나 [선택적 진단 명령](references/shared_browser.md)을 사용할 수 있다. URL 이름만 보지 말고 화면의 항목과 응답을 대조한다. 이전에 끝난 요청은 새 리스너로 소급 관찰되지 않는다.

관찰한 목록 API를 구분할 조건이 생기면 다음 함수로 **동작 전에** 응답 대기를 등록한다. 반환된 payload에는 실제 값이 있으므로 구조 진단에서 데이터 추출로 바로 이어갈 수 있다.

```python
async def capture_json(page, matches, trigger, timeout_ms=10000):
    async with page.expect_response(matches, timeout=timeout_ms) as pending:
        await trigger()
    response = await pending.value
    content_type = response.headers.get("content-type", "").lower()
    if not response.ok or "json" not in content_type:
        raise ValueError(f"목록 JSON 응답이 아닙니다: HTTP {response.status}")
    payload = await response.json()
    if isinstance(payload, dict) and (payload.get("error") or payload.get("errors")):
        raise ValueError("응답에 API 오류가 있습니다.")
    return response.request, payload
```

아래는 `/api/items`의 GET 응답이 목록이고 `button.more`가 그 요청을 일으킨다고 확인한 경우의 예제다. 실제 origin·경로·메서드·선택자로 바꾼다. GraphQL처럼 같은 URL을 공유하는 경우에는 요청 본문의 `operationName` 등도 조건에 포함할 수 있다.

```python
from urllib.parse import urlsplit

async def capture_more(page, api_url, button_selector):
    expected = urlsplit(api_url)
    def matches(response):
        actual = urlsplit(response.url)
        return (
            (actual.scheme, actual.netloc, actual.path)
            == (expected.scheme, expected.netloc, expected.path)
            and response.request.method == "GET"
        )
    return await capture_json(
        page, matches, lambda: page.locator(button_selector).click()
    )
```

반환된 `request`로 메서드·URL·본문을 확인하고, payload의 실제 항목 ID·필드·필터를 화면과 대조한다. 요청·응답 전체를 로그에 출력하기보다 필요한 필드만 작업 파일에 저장한다. 클릭은 이미 수행되었으므로 후속 코드가 같은 클릭을 무심코 반복하지 않도록 현재 페이지를 기준으로 이어간다.

## 사례 3: 확인한 cursor API를 같은 세션에서 반복 조회

다음 예제의 적용 조건은 **읽기 GET 요청**이며 응답이 `{"items": [...], "nextCursor": null 또는 문자열}`인 경우다. `null`이 종료라는 사실도 실제 응답으로 확인한 뒤 사용한다. page/offset API나 POST 검색에는 관찰한 계약에 맞는 별도 어댑터를 작성한다.

```python
async def fetch_cursor_page(context, endpoint, cursor, *, params=None, headers=None):
    query = dict(params or {})
    if "cursor" in query:
        raise ValueError("cursor는 params가 아니라 별도 인자로 전달합니다.")
    if cursor is not None:
        query["cursor"] = cursor
    response = await context.request.get(
        endpoint, params=query, headers=headers, timeout=10000
    )
    try:
        if not response.ok or "json" not in response.headers.get("content-type", "").lower():
            raise ValueError(f"목록 응답 확인 실패: HTTP {response.status}")
        payload = await response.json()
        if not isinstance(payload, dict) or payload.get("error") or payload.get("errors"):
            raise ValueError("정상 목록 객체가 아닙니다.")
        items, next_cursor = payload["items"], payload["nextCursor"]
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise ValueError("items가 레코드 배열이 아닙니다.")
        if next_cursor is not None and not isinstance(next_cursor, str):
            raise ValueError("관찰한 cursor 형식과 다릅니다.")
        return items, next_cursor
    finally:
        await response.dispose()
```

`context.request`는 브라우저 쿠키를 공유하지만 페이지 JavaScript가 붙이는 bearer·CSRF 헤더까지 복제하지 않는다. 별도 HTTP 클라이언트도 필요한 조건을 재현할 수 있으면 사용할 수 있다. 429 등이 발생하면 [대기·재시도 예제](anti_bot_stealth.md)를 적용할지 판단한다.

이 어댑터는 다음 위치와 레코드를 반환하므로 [페이지네이션·재개 예제](pagination_and_resume.md)에 연결할 수 있다. 한 페이지가 재현되면 인접 페이지의 ID와 필터·정렬도 대조해 같은 목록을 수집하고 있는지 확인한다.
