---
name: dynamic-content-diagnosis
description: Resolve empty extraction, iframe or shadow boundaries and virtualized lists with native-tool examples and adaptable code for an existing page.
---

# 빈 추출과 동적 목록의 해결 패턴

## 사례 1: 셀렉터가 맞는데 값이 비어 있음

`verify_selectors`의 "매칭 12건, 샘플 0건"은 검사한 샘플에서 값을 얻지 못했다는 뜻이다. 매칭 0건과 다르며, 전체 12개의 값이 모두 비었다는 뜻도 아니다. 이미지의 `data-src`, 링크의 `href` 등 실제 값의 위치를 확인할 수 있다.

`::attr(href)`는 `verify_selectors`의 전용 표기다. 생성한 Python 코드에서는 Playwright의 `get_attribute()`나 HTML 파서의 속성 접근으로 옮긴다. 출력이 잘린 경우에는 영역을 좁혀 확인하는 방법도 있다.

로딩 이후 `.result-item`이 나타나는 것을 확인했다면 `interact_page`에 다음 인자를 주어 기다릴 수 있다. 선택자와 시간은 해당 화면에 맞게 바꾼다.

```json
{
  "url": "",
  "actions_json": "[{\"action\":\"wait\",\"selector\":\".result-item\",\"timeout_ms\":5000}]",
  "wait_ms": 0
}
```

이 도구는 한 액션이 실패해도 뒤 액션을 시도할 수 있다. 뒤 동작이 앞 동작의 성공에 의존한다면 호출을 나눠 결과를 확인하는 편이 적합하다. `DOM 변경`은 HTML 길이를 비교하므로 길이가 같은 값 교체를 놓칠 수 있다. 실제 항목·선택 상태가 더 직접적인 근거다.

## 사례 2: 화면의 내용이 iframe 또는 Shadow DOM 안에 있음

부모 HTML에 iframe만 있다면 내부 문서에 별도로 접근한다. 아래는 해당 frame에 적어도 한 항목이 나타나는 화면의 예제다. 정상 빈 결과가 가능한 페이지라면 항목 대신 빈 결과 표시도 대기 조건에 포함하는 방식으로 바꿀 수 있다.

```python
async def read_frame_items(page, frame_selector, item_selector):
    items = page.frame_locator(frame_selector).locator(item_selector)
    await items.first.wait_for(state="attached", timeout=5000)
    return await items.all_text_contents()
```

호출 예: `await read_frame_items(page, "iframe#results", ".item")`. `page`는 [기존 탭 연결 예제](references/shared_browser.md)에서 얻는다. 복잡한 UI 탐색이라면 `browse_web`을 사용하는 방법도 있다. L1 도구에는 frame 선택 인자가 없으므로 해당 인자를 붙이는 것으로 해결되지는 않는다.

open Shadow DOM은 `await page.locator("product-list .item").all_text_contents()`처럼 Playwright locator로 접근할 수 있다. `page.content()`를 BeautifulSoup으로 읽는 방식은 같은 내부 DOM을 얻지 못한다. closed shadow root나 XPath는 같은 방식으로 접근되지 않는다.

## 사례 3: 스크롤해도 항목 수는 20개인데 ID가 교체됨

화면 밖의 노드를 제거하는 가상 목록일 수 있다. 마지막 DOM만 한 번 읽으면 지나간 항목이 빠진다. [목록 API](api_reverse_engineering.md)를 찾는 방법을 검토할 수 있고, 적합한 API가 없다면 보이는 항목을 누적할 수도 있다.

다음은 **내부 스크롤 패널**, 각 행의 **안정적인 data-id**, 행 내부 **.title**을 확인한 경우의 예제다. 이 속성이 없는 사이트에는 선택자·필드·고유 키를 맞춰 바꾼다. DOM을 한 번에 읽어 스크롤 중 서로 다른 행의 필드가 섞이는 가능성을 줄인다.

```python
async def collect_virtual_rows(page, panel_selector, row_selector,
                               max_moves=30, wait_ms=300):
    if max_moves < 0 or wait_ms < 0:
        raise ValueError("max_moves and wait_ms must be nonnegative")
    panel = page.locator(panel_selector)
    seen = {}
    reason = "move_limit"
    for move in range(max_moves + 1):
        rows = await panel.locator(row_selector).evaluate_all("""rows => rows.map(row => ({
            id: row.getAttribute('data-id'),
            title: row.querySelector('.title')?.textContent?.trim() ?? null
        }))""")
        for row in rows:
            if not row["id"] or not row["id"].strip():
                raise ValueError("A stable data-id is required by this example")
            seen.setdefault(row["id"], row)
        if move == max_moves:
            break
        before, after = await panel.evaluate("""element => {
            const before = element.scrollTop;
            element.scrollTop += Math.max(1, element.clientHeight * 0.7);
            return [before, element.scrollTop];
        }""")
        await page.wait_for_timeout(wait_ms)
        if before == after:
            reason = "no_scroll_progress"
            break
    return {"records": list(seen.values()), "stop_reason": reason}
```

이 예제는 제한된 횟수로 화면별 항목을 누적한다. 스크롤 정체는 지연 로딩 때문일 수도 있으므로 반환값 자체는 전체 수집의 증명이 아니다. UI의 전체 건수·종료 표시 등과 대조하고, 로딩 특성에 따라 고정 대기를 새 항목 ID나 로딩 완료 조건으로 바꿀 수 있다. `interact_page`의 scroll은 창 단위이므로 이런 내부 패널 조작과 다를 수 있다.
