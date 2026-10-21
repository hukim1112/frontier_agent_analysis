import requests
import json
import sys
import os
import asyncio
from typing import AsyncGenerator, Optional

import httpx
from httpx_sse import aconnect_sse


class AsyncAgentClient:
    """
    Chainlit 등 비동기 프레임워크에서 사용하는 FastAPI 클라이언트.
    httpx + httpx-sse 기반으로 비동기 SSE 스트리밍 및 커넥션 풀(Connection Pool)을 지원합니다.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._client_loop_id: Optional[int] = None

    def _get_client(self) -> httpx.AsyncClient:
        """현재 이벤트 루프에 바인딩된 공유 httpx.AsyncClient 커넥션 풀을 반환합니다."""
        try:
            loop_id = id(asyncio.get_running_loop())
        except RuntimeError:
            loop_id = None

        if (
            self._client is None
            or self._client.is_closed
            or self._client_loop_id != loop_id
        ):
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout, connect=10.0),
                limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
            )
            self._client_loop_id = loop_id
        return self._client

    async def aclose(self) -> None:
        """열려 있는 HTTP 커넥션 풀을 명시적으로 종료합니다."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None
        self._client_loop_id = None

    async def __aenter__(self) -> "AsyncAgentClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()

    async def get_agents(self) -> list:
        """GET /agents — 사용 가능한 에이전트 목록을 비동기로 조회합니다."""
        client = self._get_client()
        try:
            response = await client.get(f"{self.base_url}/agents")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Error fetching agents: {e}")
            return []

    async def stream(
        self,
        agent_name: str,
        message: str,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_thread_id: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        POST /agents/{agent_name}/stream — SSE 스트리밍 이벤트를 비동기로 수신합니다.

        user_id: 사용자 식별자 (멀티유저 메모리·세션 격리용)
        parent_thread_id: 서브에이전트 호출 시 상위 부모(Supervisor) 세션 ID

        Yields:
            dict: {"type": "token"|"tool_start"|"tool_end"|"interrupt"|"queued"|"error", ...}
        """
        url = f"{self.base_url}/agents/{agent_name}/stream"
        payload = {
            "message": message,
            "thread_id": thread_id,
            "stream_tokens": True,
        }
        if user_id:
            payload["user_id"] = user_id
        if parent_thread_id:
            payload["parent_thread_id"] = parent_thread_id

        client = self._get_client()
        try:
            async with aconnect_sse(
                client, "POST", url, json=payload
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.event == "end":
                        break
                    if sse.data:
                        try:
                            data = json.loads(sse.data)
                            yield data
                        except json.JSONDecodeError:
                            pass
        except httpx.HTTPError as e:
            yield {"type": "error", "error": str(e)}

    async def health_check(self) -> bool:
        """GET /health — FastAPI 서버 연결 상태를 확인합니다."""
        client = self._get_client()
        try:
            response = await client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def async_invoke(
        self,
        agent_name: str,
        message: str,
        thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_thread_id: Optional[str] = None,
    ) -> dict:
        """
        POST /agents/{agent_name}/invoke — 단일 에이전트 호출을 비동기로 수행합니다.
        invoke_sub_agent 도구에서 서버 내부 sub-agent를 호출할 때 사용합니다.
        """
        url = f"{self.base_url}/agents/{agent_name}/invoke"
        payload = {"message": message, "thread_id": thread_id}
        if user_id:
            payload["user_id"] = user_id
        if parent_thread_id:
            payload["parent_thread_id"] = parent_thread_id

        client = self._get_client()
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"type": "error", "content": f"{e} — {e.response.text}"}
        except httpx.HTTPError as e:
            return {"type": "error", "content": str(e)}

    async def submit_job(
        self,
        agent_name: str,
        message: str,
        thread_id: Optional[str] = None,
        callback_agent: Optional[str] = "main_agent",
        callback_thread_id: Optional[str] = None,
        user_id: Optional[str] = None,
        parent_thread_id: Optional[str] = None,
    ) -> dict:
        """
        POST /agents/{agent_name}/jobs — 작업을 백그라운드로 등록하고 job_id를 즉시 수신합니다.
        """
        url = f"{self.base_url}/agents/{agent_name}/jobs"
        payload = {
            "message": message,
            "thread_id": thread_id,
            "callback_agent": callback_agent,
            "callback_thread_id": callback_thread_id,
        }
        if user_id:
            payload["user_id"] = user_id
        effective_parent = parent_thread_id or callback_thread_id
        if effective_parent:
            payload["parent_thread_id"] = effective_parent

        client = self._get_client()
        try:
            response = await client.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {"status": "error", "message": f"{e} — {e.response.text}"}
        except httpx.HTTPError as e:
            return {"status": "error", "message": str(e)}

    async def get_job(self, job_id: str) -> dict:
        """
        GET /jobs/{job_id} — 백그라운드 작업의 상태 및 결과를 조회합니다.
        """
        url = f"{self.base_url}/jobs/{job_id}"
        client = self._get_client()
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            return {"status": "error", "message": str(e)}

    async def get_session_jobs(self, thread_id: str) -> list:
        """
        GET /sessions/{thread_id}/jobs — 세션에 연관된 모든 백그라운드 작업을 조회합니다.
        """
        url = f"{self.base_url}/sessions/{thread_id}/jobs"
        client = self._get_client()
        try:
            response = await client.get(url, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return []

    async def get_messages(self, session_id: str) -> list:
        """
        GET /sessions/{session_id}/messages — 세션의 대화 이력을 조회합니다.
        invoke_sub_agent에서 첫 호출 여부를 판단하는 데 사용합니다.
        """
        client = self._get_client()
        try:
            response = await client.get(f"{self.base_url}/sessions/{session_id}/messages")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return []

    async def resume(
        self,
        agent_name: str,
        thread_id: str,
        decisions: list[dict],
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[dict, None]:
        """
        POST /agents/{agent_name}/resume — interrupt된 에이전트를 재개하고 SSE 스트리밍을 수신합니다.
        """
        url = f"{self.base_url}/agents/{agent_name}/resume"
        payload = {"thread_id": thread_id, "decisions": decisions}
        if user_id:
            payload["user_id"] = user_id

        client = self._get_client()
        try:
            async with aconnect_sse(
                client, "POST", url, json=payload
            ) as event_source:
                async for sse in event_source.aiter_sse():
                    if sse.event == "end":
                        break
                    if sse.data:
                        try:
                            data = json.loads(sse.data)
                            yield data
                        except json.JSONDecodeError:
                            pass
        except httpx.HTTPError as e:
            yield {"type": "error", "error": str(e)}


class AgentClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def invoke(self, agent_name: str, message: str, thread_id: str = None) -> dict:
        """
        단일 호출 (Blocking)
        :return: {"type": "ai", "content": "..."}
        """
        url = f"{self.base_url}/agents/{agent_name}/invoke"
        payload = {"message": message, "thread_id": thread_id}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"type": "error", "content": str(e)}

    def stream(self, agent_name: str, message: str, thread_id: str = None):
        """
        스트리밍 호출 (Generator)
        :yield: dict (token, tool_start, error 등)
        """
        url = f"{self.base_url}/agents/{agent_name}/stream"
        payload = {"message": message, "thread_id": thread_id, "stream_tokens": True}

        try:
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()

                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode("utf-8")

                        if decoded_line.startswith("data: "):
                            json_str = decoded_line[6:]
                            if not json_str.strip():
                                continue
                            try:
                                data = json.loads(json_str)
                                yield data
                            except json.JSONDecodeError:
                                pass

                        elif decoded_line.startswith("event: end"):
                            break

        except requests.exceptions.RequestException as e:
            yield {"type": "error", "error": str(e)}

    def create_session(self, session_id: str, agent_name: str, title: str) -> dict:
        url = f"{self.base_url}/sessions"
        payload = {"session_id": session_id, "agent_name": agent_name, "title": title}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"type": "error", "error": str(e)}

    def get_agents(self) -> list:
        url = f"{self.base_url}/agents"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching agents: {e}")
            return []

    def get_sessions(self, agent_name: str = None) -> list:
        url = f"{self.base_url}/sessions"
        params = {"agent_name": agent_name} if agent_name else {}
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching sessions: {e}")
            return []

    def delete_session(self, session_id: str) -> dict:
        url = f"{self.base_url}/sessions/{session_id}"
        try:
            response = requests.delete(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"type": "error", "error": str(e)}

    def get_messages(self, session_id: str) -> list:
        url = f"{self.base_url}/sessions/{session_id}/messages"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching messages: {e}")
            return []


if __name__ == "__main__":
    client = AgentClient()
    available_agents = [a["name"] for a in client.get_agents()]

    print("=" * 50)
    print("🤖 Agent Client Console")
    print(f"Available Agents: {', '.join(available_agents) if available_agents else 'None'}")
    print("Commands:")
    print("  /switch {agent_name} : Switch agent")
    print("  quit / exit          : Exit console")
    print("=" * 50)

    current_agent = available_agents[0] if available_agents else "chatbot"
    thread_id = "cli_test_thread"

    while True:
        try:
            user_input = input(f"\n[{current_agent}] User: ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        if user_input.lower() in ["quit", "exit"]:
            print("Bye!")
            break

        if user_input.startswith("/switch"):
            parts = user_input.split(" ", 1)
            if len(parts) > 1:
                current_agent = parts[1].strip()
                print(f"✅ Switched to agent: {current_agent}")
            else:
                print("⚠️ Usage: /switch {agent_name}")
            continue

        print(f"[{current_agent}] AI: ", end="", flush=True)

        try:
            for chunk in client.stream(current_agent, user_input, thread_id):
                if "type" in chunk:
                    if chunk["type"] == "token":
                        content = chunk.get("content", "")
                        print(content, end="", flush=True)
                    elif chunk["type"] == "tool_start":
                        print(f"\n🛠️ [Tool: {chunk['name']}] Processing...", end="")
                        if "input" in chunk:
                            print(f" Input: {chunk['input']}", end="")
                        print("\n", end="")
                    elif chunk["type"] == "tool_end":
                        print(f"✅ [{chunk['name']}] 완료", end="")
                        if "output" in chunk:
                            print(f" → {chunk['output'][:100]}", end="")
                        print()
                    elif chunk["type"] == "error":
                        print(f"\n❌ Error: {chunk.get('content') or chunk.get('error')}")
                elif "error" in chunk:
                    print(f"\n❌ Error: {chunk['error']}")
            print()

        except KeyboardInterrupt:
            print("\n⛔ Interrupted.")
