---
name: anti-bot-recovery
description: Diagnose browser-versus-script access differences and adapt a bounded Retry-After recipe for read-only requests interrupted by rate limits or transient server errors.
---

# 세션 차이와 요청 제한에 대응하는 레시피

## 사례 1: 브라우저에서는 보이는데 코드에서는 403 또는 로그인 HTML

`browse_web`으로 로그인한 뒤 새 `requests` 세션을 사용하면 쿠키가 자동 전달되지 않는다. 이때는 봇 탐지를 가정하기 전에 두 요청의 최종 URL·메서드·인증 조건을 비교하는 것이 도움이 된다. 메인 문서는 200이지만 데이터 API만 401인 경우도 있다.

현재 도구의 화면은 `take_screenshot(url="")` 등으로 확인할 수 있다. HTTP 응답 확인이 더 필요하면 직접 Playwright 응답을 읽거나 [선택적 네트워크 진단](references/shared_browser.md)을 사용할 수 있다.

기존 탭의 `page.context.request`는 그 컨텍스트의 쿠키를 공유한다. [API 요청 예제](api_reverse_engineering.md)로 작은 읽기 요청을 재현하고, 화면과 데이터가 같아지는지 비교할 수 있다. JavaScript가 붙인 Authorization·CSRF 헤더는 별도로 확인해야 하며, 필요한 자격 증명이 없는 상황은 재시도로 해결되지 않는다.

새 브라우저를 띄우는 일반적인 stealth 예제는 이미 확보한 세션을 이어받지 못한다. 고정 User-Agent나 webdriver 변경이 유효한지는 사이트별 근거가 필요하며, 권한 부족·CAPTCHA를 해결하는 범용 처방은 아니다. `--no-sandbox`도 안티봇 기능이 아니다.

## 사례 2: 반복 조회 중 429 또는 일시적인 503

`Retry-After: 120`은 120초 이후 재시도를 뜻하므로 1~3초의 지연만 추가해서는 조건을 충족하지 못한다. 헤더 값이 HTTP 날짜일 수도 있다.

아래 예제는 **읽기 GET 요청**에서 429·502·503·504가 나온 경우에만 제한적으로 재시도한다. 이미 관찰한 API를 대상으로 사용하며, 상태 코드·횟수·대기 예산은 대상에 맞게 조정할 수 있다. 요청 자체의 시간과 별개로 누적 대기 시간도 제한한다.

```python
import asyncio
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

def retry_after_seconds(value, now=None):
    if not value:
        return None
    value = value.strip()
    try:
        if value.isdecimal():
            return int(value)
        deadline = parsedate_to_datetime(value)
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return max(0, (deadline - (now or datetime.now(timezone.utc))).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return None

async def get_json_with_backoff(context, url, *, params=None, headers=None,
                                max_attempts=3, max_wait_s=30):
    if max_attempts < 1 or max_wait_s < 0:
        raise ValueError("max_attempts must be positive and max_wait_s nonnegative")
    remaining_wait = max_wait_s
    for attempt in range(max_attempts):
        response = await context.request.get(
            url, params=params, headers=headers, timeout=10000
        )
        try:
            status = response.status
            if status not in {429, 502, 503, 504}:
                if not response.ok or "json" not in response.headers.get("content-type", "").lower():
                    raise ValueError(f"데이터 응답이 아닙니다: HTTP {status}")
                payload = await response.json()
                if isinstance(payload, dict) and (payload.get("error") or payload.get("errors")):
                    raise ValueError("API가 오류 객체를 반환했습니다.")
                return payload
            delay = retry_after_seconds(response.headers.get("retry-after"))
            if delay is None:
                delay = 2 ** attempt
        finally:
            await response.dispose()
        if attempt + 1 == max_attempts:
            raise RuntimeError(f"재시도 횟수 소진: HTTP {status}")
        if delay > remaining_wait:
            raise RuntimeError(f"필요한 대기 {delay}초가 남은 대기 예산을 초과합니다.")
        await asyncio.sleep(delay)
        remaining_wait -= delay
```

호출 예: `await get_json_with_backoff(page.context, endpoint, params={"page": 1}, max_attempts=3, max_wait_s=30)`. 서버가 120초 대기를 요구하면 이 설정은 30초 뒤에 조기 재요청하지 않고 중단한다. 헤더가 없을 때 사용하는 1초·2초 백오프는 조정 가능한 예시이며 모든 사이트에 충분하다는 의미는 아니다.

이 예제는 401/403, 잘못된 JSON, GraphQL 오류, 전송 예외를 무조건 재시도하지 않는다. POST 조회나 다른 복구가 필요하면 실제 요청의 의미와 실패 원인에 맞게 별도로 구성한다. 실패한 페이지는 [재개 레시피](pagination_and_resume.md)의 완료 위치로 넘기지 않는 방식으로 연결할 수 있다.

## 사례 3: 브라우저 자체에 챌린지 화면이 나타남

HTTP 200이어도 로그인·챌린지 HTML이면 정상 목록이 아니다. 권한 부족, 세션 만료, CAPTCHA·2FA를 구분하면 다음 행동을 결정할 수 있다. 기존 권한으로 다시 로그인할 수 있는 경우에는 현재 탭에서 상태를 회복하고 표본을 확인한다. 사람의 확인이 필요한 화면에서는 탭을 보존하고 필요한 개입을 보고한다.

참고: [HTTP 429](https://www.rfc-editor.org/rfc/rfc6585#section-4), [Retry-After](https://httpwg.org/specs/rfc9110.html#field.retry-after).
