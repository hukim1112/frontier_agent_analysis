---
name: pagination-and-resume
description: Adapt observed page or cursor pagination and recover an interrupted collection with a small data-first checkpoint recipe, stable identity and explicit completion.
---

# 페이지 이동과 중단 후 재개의 구현 예제

## 먼저 고를 부분: 사이트의 다음 위치 표현

| 관찰한 방식 | 어댑터에서 반환할 다음 위치 | 확인할 맥락 |
|---|---|---|
| page/offset | 관찰한 다음 번호 또는 offset | 파라미터가 무시되어 첫 페이지가 반복될 수 있음 |
| cursor | 응답이 발급한 불투명 cursor | 숫자로 계산하지 않음; 종료 토큰의 의미 확인 |
| 더보기·가상 목록 | 다음 UI 동작 또는 현재 상태 | API 반복 조회와 다른 방식이므로 [동적 목록 레시피](dynamic_content_diagnosis.md)도 검토 |

아래 레시피는 **같은 위치를 다시 조회할 수 있는 읽기 API**, **안정적인 고유 키**, **단일 수집기**를 전제로 한다. UI 세션 자체를 복원하거나 원본의 시점 일관성을 보장하는 범용 크롤러는 아니다. 사이트가 수집 중 바뀌거나 cursor가 만료되면 별도 복구가 필요할 수 있다.

## 데이터 저장 후 체크포인트 갱신

`fetch_page(position)`이 `(레코드 배열, 다음 위치)`를 반환하고, 확인된 종료에서만 다음 위치를 `None`으로 반환하는 예제다. 시작 위치로 `None`을 쓰는 cursor API도 사용할 수 있다. 오류 응답은 빈 목록으로 바꾸지 않고 예외로 전달한다.

매 페이지마다 전체 데이터를 교체하는 중소규모 수집용 예제이며, 큰 데이터에는 페이지별 파일이나 DB 저장 방식을 선택할 수 있다. 동일 키의 중복은 첫 관찰 값을 유지한다. 복합 키는 `key_fields=("id", "variant")`처럼 지정할 수 있다.

```python
import json
import os
import tempfile
from pathlib import Path

def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

def record_key(record, fields):
    if not isinstance(record, dict):
        raise ValueError("A record must be an object")
    values = [record[field] for field in fields]
    if any(isinstance(value, bool) or not isinstance(value, (str, int))
           or isinstance(value, str) and not value.strip() for value in values):
        raise ValueError("This recipe expects nonempty string or integer identity fields")
    return json.dumps(values, ensure_ascii=False)

async def collect_pages(fetch_page, run_dir, *, source, start=1,
                        key_fields=("id",), max_pages=20):
    if max_pages < 1 or not key_fields or isinstance(key_fields, str):
        raise ValueError("Check max_pages and key_fields")
    root = Path(run_dir)
    data_path, checkpoint_path = root / "data.json", root / "checkpoint.json"
    identity = json.loads(json.dumps({
        "source": source, "start": start, "key_fields": list(key_fields), "schema": 1
    }, allow_nan=False))
    records = []
    if data_path.exists():
        data = json.loads(data_path.read_text(encoding="utf-8"))
        if data["identity"] != identity or not isinstance(data["records"], list):
            raise ValueError("Stored data belongs to a different collection")
        records = data["records"]
    state = {"identity": identity, "next_position": start, "done": False,
             "completed_positions": [], "unique_count": 0}
    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if state["identity"] != identity or not data_path.exists():
            raise ValueError("Checkpoint and data do not describe the same collection")
        if len(records) < state["unique_count"]:
            raise ValueError("Committed records are missing")
    by_key = {record_key(record, key_fields): record for record in records}
    if len(by_key) != len(records):
        raise ValueError("Stored data contains duplicate identity keys")
    completed = set(state["completed_positions"])
    for _ in range(max_pages):
        if state["done"]:
            break
        position = state["next_position"]
        position_key = json.dumps(position, sort_keys=True, allow_nan=False)
        if position_key in completed:
            raise ValueError("Pagination revisited a completed position")
        items, next_position = await fetch_page(position)
        if not isinstance(items, list):
            raise ValueError("The page adapter must return a list of records")
        next_key = json.dumps(next_position, sort_keys=True, allow_nan=False)
        if next_position is not None and next_key in completed | {position_key}:
            raise ValueError("Pagination is cycling instead of advancing")
        for item in items:
            by_key.setdefault(record_key(item, key_fields), item)
        records = list(by_key.values())
        next_state = {
            "identity": identity, "next_position": next_position,
            "done": next_position is None, "unique_count": len(records),
            "completed_positions": [*state["completed_positions"], position_key],
        }
        atomic_json(data_path, {"identity": identity, "records": records})
        atomic_json(checkpoint_path, next_state)
        state = next_state
        completed.add(position_key)
    return {"records": records, "done": state["done"],
            "next_position": state["next_position"],
            "stop_reason": "end" if state["done"] else "page_limit"}
```

데이터 저장 뒤 체크포인트 저장 전에 중단되면, 다음 실행은 이전 위치를 다시 조회한다. 이미 저장된 레코드를 고유 키로 합치므로 재조회 자체가 중복 산출물로 이어지지 않는다. 데이터 파일이 없거나 수집 조건이 다르면 이어서 완료 처리하는 대신 오류를 반환한다. 재개 토큰과 같은 작업 상태는 이 작업 디렉터리에서 유지한다.

이 구현의 `max_pages`는 **이번 호출의 요청 상한**이다. 전체 작업 예산이 있다면 남은 횟수를 넘겨 사용한다. 페이지 상한에 도달하면 `done=False`로 반환하므로 이를 전체 수집 완료와 구분할 수 있다. 원본 ID가 숫자·문자열이 아닌 경우에는 사이트의 고유 키에 맞게 `record_key`를 바꾼다.

## cursor API와 연결하는 예제

[API 문서](api_reverse_engineering.md)의 `fetch_cursor_page`와 위 함수를 같은 작업 스크립트에 두면 다음처럼 연결할 수 있다. `page`는 [기존 탭 연결](references/shared_browser.md) 블록 안의 페이지다. 다음 호출에 같은 디렉터리와 source 조건을 주면 저장된 위치에서 이어간다.

```python
async def collect_cursor_list(page, endpoint, run_dir, *, filters=None, max_pages=20):
    filters = dict(filters or {})
    async def fetch(position):
        return await fetch_cursor_page(page.context, endpoint, position, params=filters)
    result = await collect_pages(
        fetch, run_dir,
        source={"endpoint": endpoint, "filters": filters, "pagination": "cursor"},
        start=None, key_fields=("id",), max_pages=max_pages,
    )
    atomic_json(Path(run_dir) / "items.json", result["records"])
    return result
```

`items.json`은 확인된 레코드의 JSON 배열이며 `data.json`·`checkpoint.json`은 재개용 상태다. 성공적으로 반환된 부분 수집도 저장할 수 있다. 예외로 중단된 경우에는 재개용 파일의 마지막 저장 지점부터 이어갈 수 있다. 품질·실제 범위의 확인에는 [결과 검증](collection_validation.md)을 참고한다.
