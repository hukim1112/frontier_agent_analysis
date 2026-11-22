"""
tests/agent_inspector.py
========================
CompiledStateGraph(`create_agent` 반환 객체) 내부의 노드, 도구, 미들웨어,
체크포인터 커넥션을 직접 탐색하고 리소스를 안전하게 정리하는 테스트/채점 전용 인트로스펙션 유틸리티.

에이전트 팩토리(`agent.py`)에 테스트용 임의 속성(`registered_tools`,
`checkpointer_conn`, `assembler`, `memory_middleware` 등)을 붙이지 않고도
런타임 그래프 자체를 단일 진실 공급원(SSOT)으로 검사·정리합니다.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Type, TypeVar

from langchain.agents.middleware import AgentMiddleware

_M = TypeVar("_M", bound=AgentMiddleware)


def extract_agent_tools(agent: Any) -> List[Any]:
    """컴파일된 에이전트 그래프의 'tools' 노드에서 바인딩된 실제 도구 목록을 추출합니다."""
    if agent is None or not hasattr(agent, "nodes"):
        return []
    tools_node = agent.nodes.get("tools")
    if tools_node is None:
        return []
    bound = getattr(tools_node, "bound", tools_node)
    if bound is not None and hasattr(bound, "tools_by_name"):
        return list(bound.tools_by_name.values())
    return []


def extract_agent_middlewares(agent: Any) -> List[AgentMiddleware]:
    """컴파일된 에이전트 그래프의 노드/래퍼 클로저에서 활성 AgentMiddleware 인스턴스 목록을 추출합니다."""
    if agent is None or not hasattr(agent, "nodes"):
        return []

    found: List[AgentMiddleware] = []
    seen_mw_ids: set[int] = set()
    visited_ids: set[int] = set()

    def _walk(obj: Any, depth: int = 0) -> None:
        if obj is None or depth > 12:
            return
        obj_id = id(obj)
        if obj_id in visited_ids:
            return
        visited_ids.add(obj_id)

        if isinstance(obj, AgentMiddleware):
            if obj_id not in seen_mw_ids:
                seen_mw_ids.add(obj_id)
                found.append(obj)
            return

        owner = getattr(obj, "__self__", None)
        if isinstance(owner, AgentMiddleware):
            owner_id = id(owner)
            if owner_id not in seen_mw_ids:
                seen_mw_ids.add(owner_id)
                found.append(owner)

        for attr in ("bound", "func", "afunc"):
            child = getattr(obj, attr, None)
            if child is not None:
                _walk(child, depth + 1)

        try:
            obj_dict = getattr(obj, "__dict__", None)
        except Exception:
            obj_dict = None
        if isinstance(obj_dict, dict):
            for val in obj_dict.values():
                if callable(val) or isinstance(val, AgentMiddleware):
                    _walk(val, depth + 1)

        closure = getattr(obj, "__closure__", None)
        if closure:
            for cell in closure:
                try:
                    cell_val = cell.cell_contents
                except ValueError:
                    continue
                _walk(cell_val, depth + 1)

    for node in agent.nodes.values():
        _walk(node)

    return found


def find_agent_middleware(agent: Any, middleware_cls: Type[_M]) -> Optional[_M]:
    """컴파일된 에이전트 그래프에서 지정 타입의 미들웨어 인스턴스를 찾아 반환합니다."""
    for mw in extract_agent_middlewares(agent):
        if isinstance(mw, middleware_cls):
            return mw
    return None


def extract_prompt_assembler(agent: Any) -> Optional[Any]:
    """컴파일된 에이전트 그래프에 마운트된 PromptAssembler 인스턴스를 추출합니다."""
    for mw in extract_agent_middlewares(agent):
        assembler = getattr(mw, "assembler", None)
        if assembler is not None:
            return assembler
    return None


async def close_agent_resources(
    agent: Any,
    review_timeout: float = 5.0,
    shutdown_review_queue: bool = False,
) -> None:
    """에이전트 그래프가 보유한 백그라운드 리뷰 큐, 메모리 DB 스토어, 체크포인터 커넥션을 안전하게 종료합니다."""
    if agent is None:
        return

    closed_store_ids: set[int] = set()
    for mw in extract_agent_middlewares(agent):
        if hasattr(mw, "wait_for_reviews"):
            try:
                await asyncio.to_thread(mw.wait_for_reviews, review_timeout)
            except Exception:
                pass
            if shutdown_review_queue and hasattr(mw, "_review_queue"):
                try:
                    mw._review_queue.shutdown()
                except Exception:
                    pass

        for store_attr in ("episodic_store", "transcript_store"):
            store = getattr(mw, store_attr, None)
            if store is not None and id(store) not in closed_store_ids:
                closed_store_ids.add(id(store))
                if hasattr(store, "close"):
                    try:
                        await store.close()
                    except Exception:
                        pass

    cp = getattr(agent, "checkpointer", None)
    conn = getattr(cp, "conn", None) if cp is not None else None
    if conn is not None and getattr(conn, "_running", False):
        await conn.close()
