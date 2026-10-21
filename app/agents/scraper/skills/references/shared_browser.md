# 기존 Chrome의 상태를 수집 코드에서 사용하기

## 도구에서 로그인했는데 Python 코드에는 세션이 없는 경우

Navigator와 `browse_web`은 같은 Chrome을 사용하지만, `bash_command`의 Python은 별도 프로세스다. 그 안에서 `PlaywrightManager`를 생성하거나 새 브라우저를 띄워도 도구의 로그인·검색 상태가 자동으로 이어지지 않는다.

기존 도구로 필요한 값을 얻을 수 있으면 도구를 그대로 사용한다. 직접 Playwright 코드를 작성해야 할 때는 아래처럼 실행 중인 Chrome에 CDP로 연결할 수 있다. 이 연결은 작업별 브라우저 격리를 제공하지 않으므로 한 브라우저에서 한 수집 흐름을 이어가는 경우를 전제로 한다.

## 기존 탭을 빌려 쓰는 예제

`cdp_url`은 실제 Chrome의 주소다. 기본 포트는 `app/tools/navigator.py`의 `CDP_DEBUG_PORT`이며 현재 기본 주소는 `http://127.0.0.1:9242`다. `current_url`에는 도구가 확인한 최종 URL을 쿼리까지 넣는다.

```python
from contextlib import asynccontextmanager
from playwright.async_api import async_playwright

@asynccontextmanager
async def existing_page(cdp_url, *, current_url=None, target_id=None):
    async with async_playwright() as playwright:
        browser = await playwright.chromium.connect_over_cdp(cdp_url, timeout=8000)
        candidates = []
        for context in browser.contexts:
            for page in context.pages:
                if page.is_closed():
                    continue
                if current_url is not None and page.url != current_url:
                    continue
                if target_id is not None:
                    session = await context.new_cdp_session(page)
                    try:
                        info = (await session.send("Target.getTargetInfo"))["targetInfo"]
                    finally:
                        await session.detach()
                    if info["targetId"] != target_id:
                        continue
                elif current_url is None and not page.url.startswith(("http://", "https://")):
                    continue
                candidates.append(page)
        if len(candidates) != 1:
            raise ValueError("대상 탭을 URL 또는 target ID로 하나만 지정해야 합니다.")
        yield candidates[0]
```

예를 들어 현재 HTML을 가져오는 호출은 다음과 같다. 다른 스킬의 `page`를 받는 함수들도 이 `async with` 안에서 호출할 수 있다.

```python
async def current_html(cdp_url, current_url):
    async with existing_page(cdp_url, current_url=current_url) as page:
        return await page.content()
```

예제는 이동·새로고침·새 탭 생성 없이 연결하며, 블록을 나가면 자신의 Playwright 클라이언트만 정리한다. 빌린 `page/context/browser`에 `close()`를 호출하면 원래 작업에 영향을 줄 수 있다. 동일 URL의 탭이 여러 개면 임의의 첫 탭 대신 식별한 `target_id`를 넘긴다. 연결 실패는 새 Chrome을 시작하라는 신호가 아니며, 도구가 사용하는 주소와 실행 환경을 확인할 상황이다.

## 선택 사항: 빠른 구조 진단

코드로 직접 조사하는 대신 기존 보조 스크립트를 사용할 수도 있다. 다음 명령은 프로젝트 루트와 AAWS Python 환경을 기준으로 한다.

```bash
python app/agents/scraper/skills/scripts/browser_probe.py tabs
python app/agents/scraper/skills/scripts/browser_probe.py inspect --target-id TARGET_ID --output artifacts/scrape/page.json
python app/agents/scraper/skills/scripts/browser_probe.py network --target-id TARGET_ID --click "button.load-more" --output artifacts/scrape/network.json
```

- `tabs`의 ID로 같은 URL의 탭도 구분할 수 있다. ID를 생략하면 HTTP(S) 탭이 하나일 때만 선택된다.
- `inspect`는 frame 목록과 내장 JSON의 구조를 보여 준다. 실제 데이터 값은 [내장 JSON 레시피](../api_reverse_engineering.md)로 읽을 수 있다.
- `network`는 리스너를 등록한 뒤 지정한 클릭을 **실제로 한 번 수행**한다. 클릭을 이미 수행했다면 다음 코드에서 같은 클릭을 반복할 필요가 있는지 현재 상태를 확인한다.
- 기본 관찰은 3초이며 `--duration-ms`로 최대 15초까지 지정할 수 있다. 현재 페이지와 frame의 새 document/xhr/fetch만 대상으로 하고, 과거 요청·새 팝업·WebSocket은 포함하지 않는다.
- URL 쿼리 값·본문 값·인증 헤더는 결과에 포함하지 않는다. JSON 구조는 제한된 깊이와 첫 배열 항목으로 요약하므로 이 출력만으로 요청을 재현하거나 데이터를 저장할 수는 없다.
- 종료 코드 0은 관찰 실행 성공, 1은 요청한 동작 실패, 2는 연결·대상 선택 등의 오류다. 응답에 기록된 401/429는 별도로 해석한다.

이 보조 스크립트도 공유 Chrome을 닫지 않으며, 다른 수집 흐름과의 동시 사용을 차단하는 기능은 없다.
