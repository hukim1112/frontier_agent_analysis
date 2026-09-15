import requests
import json
import sys
import os
from typing import AsyncGenerator, Optional

import httpx
from httpx_sse import aconnect_sse


class AsyncAgentClient:
    """
    Chainlit 등 비동기 프레임워크에서 사용하는 FastAPI 클라이언트.
    httpx + httpx-sse 기반으로 비동기 SSE 스트리밍을 지원합니다.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def get_agents(self) -> list:
        """GET /agents — 사용 가능한 에이전트 목록을 비동기로 조회합니다."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
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
    ) -> AsyncGenerator[dict, None]:
        """
        POST /agents/{agent_name}/stream — SSE 스트리밍 이벤트를 비동기로 수신합니다.
        
        Yields:
            dict: {"type": "token"|"tool_start"|"tool_end"|"error", ...}
        """
        url = f"{self.base_url}/agents/{agent_name}/stream"
        payload = {"message": message, "thread_id": thread_id, "stream_tokens": True}

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client:
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
        async with httpx.AsyncClient(timeout=5.0) as client:
            try:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
            except httpx.HTTPError:
                return False

    async def async_invoke(
        self,
        agent_name: str,
        message: str,
        thread_id: Optional[str] = None,
    ) -> dict:
        """
        POST /agents/{agent_name}/invoke — 단일 에이전트 호출을 비동기로 수행합니다.
        invoke_sub_agent 도구에서 서버 내부 sub-agent를 호출할 때 사용합니다.

        Args:
            agent_name: 호출할 에이전트 이름 (e.g. 'scraper', 'analyst')
            message: 에이전트에게 전달할 메시지
            thread_id: 세션 연속성을 위한 스레드 ID (None이면 새 세션)

        Returns:
            {"type": "ai", "content": "..."} 형태의 응답 딕셔너리.
            오류 발생 시 {"type": "error", "content": str(e)} 반환.
        """
        url = f"{self.base_url}/agents/{agent_name}/invoke"
        payload = {"message": message, "thread_id": thread_id}

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                return {"type": "error", "content": str(e)}

    async def submit_job(
        self,
        agent_name: str,
        message: str,
        thread_id: Optional[str] = None,
        callback_agent: Optional[str] = "supervisor",
        callback_thread_id: Optional[str] = None,
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

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                return {"status": "error", "message": str(e)}

    async def get_job(self, job_id: str) -> dict:
        """
        GET /jobs/{job_id} — 백그라운드 작업의 상태 및 결과를 조회합니다.
        """
        url = f"{self.base_url}/jobs/{job_id}"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                return {"status": "error", "message": str(e)}

    async def get_session_jobs(self, thread_id: str) -> list:
        """
        GET /sessions/{thread_id}/jobs — 세션에 연관된 모든 백그라운드 작업을 조회합니다.
        """
        url = f"{self.base_url}/sessions/{thread_id}/jobs"
        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                return []

    async def get_messages(self, session_id: str) -> list:
        """
        GET /sessions/{session_id}/messages — 세션의 대화 이력을 조회합니다.
        invoke_sub_agent에서 첫 호출 여부를 판단하는 데 사용합니다.

        Returns:
            메시지 목록 (비어있으면 첫 호출). 오류 시 빈 리스트 반환.
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
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
    ) -> AsyncGenerator[dict, None]:
        """
        POST /agents/{agent_name}/resume — interrupt된 에이전트를 재개하고 SSE 스트리밍을 수신합니다.
        
        Args:
            agent_name: 에이전트 이름
            thread_id: 대화 스레드 ID (interrupt 발생 시와 동일해야 함)
            decisions: 사용자 결정 리스트 [{"type": "approve"}, {"type": "reject", "message": "..."}]
        """
        url = f"{self.base_url}/agents/{agent_name}/resume"
        payload = {"thread_id": thread_id, "decisions": decisions}

        async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout, connect=10.0)) as client:
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
            # stream=True로 연결 유지
            with requests.post(url, json=payload, stream=True) as response:
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        
                        # SSE Format: "data: {...}"
                        if decoded_line.startswith("data: "):
                            json_str = decoded_line[6:] # remove "data: "
                            if not json_str.strip():
                                continue
                            try:
                                data = json.loads(json_str)
                                yield data
                            except json.JSONDecodeError:
                                pass
                                
                        # End Event
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

# --- Interactive Test Loop ---
if __name__ == "__main__":
    client = AgentClient()
    
    # API를 통해 동적으로 에이전트 목록 로드
    client = AgentClient()
    available_agents = [a["name"] for a in client.get_agents()]

    print("="*50)
    print("🤖 Agent Client Console")
    print(f"Available Agents: {', '.join(available_agents) if available_agents else 'None'}")
    print("Commands:")
    print("  /switch {agent_name} : Switch agent")
    print("  quit / exit          : Exit console")
    print("="*50)
    
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
        
        # Stream Output
        try:
            for chunk in client.stream(current_agent, user_input, thread_id):
                if "type" in chunk:
                    if chunk["type"] == "token":
                        content = chunk.get("content", "")
                        print(content, end="", flush=True)
                    elif chunk["type"] == "tool_start":
                        print(f"\n🛠️ [Tool: {chunk['name']}] Processing...", end="")
                        if 'input' in chunk:
                             print(f" Input: {chunk['input']}", end="")
                        print("\n", end="")
                    elif chunk["type"] == "tool_end":
                        print(f"✅ [{chunk['name']}] 완료", end="")
                        if 'output' in chunk:
                            print(f" → {chunk['output'][:100]}", end="")
                        print()
                    elif chunk["type"] == "error":
                        print(f"\n❌ Error: {chunk.get('content') or chunk.get('error')}")
                elif "error" in chunk:
                    print(f"\n❌ Error: {chunk['error']}")
            print() # Newline at end
            
        except KeyboardInterrupt:
            print("\n⛔ Interrupted.")
