"""
===============================================================================
MCP Server Registry — MCP 도구를 API tools 칸에, 서버 지침을 대화 속 알림으로
===============================================================================
Source: Claude Code v2.1.88 (services/mcp/client.ts, tools.ts assembleToolPool,
        utils/mcpInstructionsDelta.ts, constants/prompts.ts getMcpInstructions)

원본에서 MCP 서버 하나를 연결하면 모델에게 전달되는 정보는 두 갈래입니다.
  (a) 도구 스키마  → API tools 칸. 이름은 `mcp__{서버}__{도구}`, 내장 도구 뒤에 따로 정렬.
  (b) 서버 지침    → 서버가 handshake(InitializeResult.instructions)로 준 안내문 +
                     클라이언트가 덧붙이는 안내문(ClientSideInstruction).
                     대화에 저장되는 `mcp_instructions_delta` 알림으로 한 번만 전달.

이 모듈은 configs/mcp.config의 서버 목록으로 (a)와 (b)를 준비합니다.
  - tools                : create_agent(tools=...)에 그대로 넘길 LangChain 도구 목록
  - instruction_blocks() : {서버 이름: 지침} — PromptAssembler의 mcp_instructions로 넘김
연결에 실패한 서버는 건너뛰고 경고만 남깁니다 (에이전트 생성은 실패하지 않음).

configs/mcp.config 형식:
{
  "mcp_enabled": true,
  "connect_timeout_s": 60,
  "servers": {
    "wikipedia": {
      "transport": "stdio", "command": "npx", "args": ["-y", "wikipedia-mcp"],
      "instructions": "클라이언트가 덧붙이는 사용 안내 (선택)",
      "tool_descriptions": {"search": "서버가 설명을 비워 둔 도구의 설명 (선택)"}
    }
  }
}
===============================================================================
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MCP_TOOL_PREFIX = "mcp__"
_CLIENT_ONLY_KEYS = ("instructions", "tool_descriptions", "enabled")


def mcp_tool_name(server: str, tool: str) -> str:
    """원본 buildMcpToolName: mcp__{server}__{tool}."""
    return f"{MCP_TOOL_PREFIX}{server}__{tool}"


def load_mcp_config(path: str = "./configs/mcp.config") -> Dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:  # 설정 파일이 깨져 있어도 에이전트는 떠야 합니다
        logger.warning("[MCP] 설정 파일을 읽지 못했습니다 (%s): %s", path, e)
    return {}


class MCPServerRegistry:
    """configs/mcp.config의 서버에 연결해 도구와 서버 지침을 준비합니다."""

    def __init__(self, servers: Optional[Dict[str, Dict[str, Any]]] = None,
                 connect_timeout_s: float = 60.0):
        self.servers: Dict[str, Dict[str, Any]] = {
            name: cfg for name, cfg in (servers or {}).items() if cfg.get("enabled", True)
        }
        self.connect_timeout_s = connect_timeout_s
        self.tools: List[Any] = []
        self.connected: Dict[str, str] = {}   # {서버 이름: 서버가 준 instructions (없으면 "")}
        self.errors: Dict[str, str] = {}

    @classmethod
    def from_config(cls, path: str = "./configs/mcp.config") -> "MCPServerRegistry":
        cfg = load_mcp_config(path)
        if not cfg.get("mcp_enabled", False):
            return cls({})
        return cls(cfg.get("servers", {}), float(cfg.get("connect_timeout_s", 60)))

    @staticmethod
    def _connection(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """langchain-mcp-adapters 연결 설정 (클라이언트 전용 키 제외)."""
        return {k: v for k, v in cfg.items() if k not in _CLIENT_ONLY_KEYS}

    async def _connect_one(self, client: Any, name: str) -> None:
        cfg = self.servers[name]
        info = await client.get_server_info(server_name=name)
        tools = await client.get_tools(server_name=name)
        overrides = cfg.get("tool_descriptions") or {}
        for tool in tools:
            original = tool.name
            # 원래 이름은 도구 내부 클로저가 기억하므로 LangChain 쪽 이름만 바꿔도 호출은 그대로 동작합니다.
            tool.name = mcp_tool_name(name, original)
            if not (tool.description or "").strip() and overrides.get(original):
                tool.description = overrides[original]
        self.tools.extend(tools)
        result = info.get(name)
        self.connected[name] = ((getattr(result, "instructions", None) or "").strip())

    async def connect(self) -> "MCPServerRegistry":
        """모든 서버에 연결합니다. 실패한 서버는 errors에 기록하고 건너뜁니다."""
        if not self.servers:
            return self
        try:
            from langchain_mcp_adapters.client import MultiServerMCPClient
        except ImportError as e:
            self.errors = {name: f"langchain-mcp-adapters not installed: {e}" for name in self.servers}
            logger.warning("[MCP] langchain-mcp-adapters가 없어 MCP 서버를 연결하지 않습니다.")
            return self

        client = MultiServerMCPClient({n: self._connection(c) for n, c in self.servers.items()})
        for name in self.servers:
            try:
                await asyncio.wait_for(self._connect_one(client, name), self.connect_timeout_s)
            except Exception as e:
                self.errors[name] = f"{type(e).__name__}: {e}"
                logger.warning("[MCP] '%s' 서버 연결 실패, 건너뜁니다: %s", name, e)
        # 내장 도구와 섞이지 않도록 이름순 (최종 정렬은 PromptAssemblerMiddleware가 다시 한 번 수행)
        self.tools.sort(key=lambda t: t.name)
        return self

    def instruction_blocks(self) -> Dict[str, str]:
        """{서버 이름: 지침} — 서버 지침(handshake) + 클라이언트 안내(config instructions).

        원본 getMcpInstructionsDelta의 blocks 계산과 같습니다. 연결된 서버 중 지침이 있는 서버만 포함합니다.
        """
        blocks: Dict[str, str] = {}
        for name, server_text in self.connected.items():
            client_text = (self.servers.get(name, {}).get("instructions") or "").strip()
            text = "\n\n".join(t for t in (server_text, client_text) if t)
            if text:
                blocks[name] = text
        return blocks
