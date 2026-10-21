---
name: scraper-playbooks
description: On-demand scraping cases and adaptable recipes for access failures, embedded data and APIs, dynamic pages, pagination and collection quality.
license: Apache-2.0
---

# Scraper 문제 해결 자료

작업 중 필요한 맥락에 맞는 문서를 읽는다. 예제는 명시된 조건에서 사용할 수 있는 출발점이며, 실제 사이트의 구조와 요청 방식에 맞게 응용한다. 전체 스킬 목록과 읽기 경로는 카탈로그에 있다.

| 필요한 맥락 | 참고 문서 |
|---|---|
| 브라우저와 수집 코드의 응답 차이, 세션 만료, 429 | [접근 실패와 복구](anti_bot_stealth.md) |
| 내장 JSON, 목록 API 발견, 요청 재현 | [내장 데이터와 API](api_reverse_engineering.md) |
| 빈 값, iframe·Shadow DOM, 가상 목록 | [동적 콘텐츠](dynamic_content_diagnosis.md) |
| 여러 페이지 수집, 중단 지점부터 재개 | [페이지네이션과 재개](pagination_and_resume.md) |
| 필드 누락·중복·행 불일치, 수집 범위 확인 | [결과 검증](collection_validation.md) |

생성한 코드에서 현재 브라우저 상태가 필요할 때는 [기존 탭 연결 예제](references/shared_browser.md)를 참고한다.
