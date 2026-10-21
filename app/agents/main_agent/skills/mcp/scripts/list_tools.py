#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===============================================================================
[Harness Module 03-Skill] MCP Client Discovery (Hybrid SSE / Stdio)
===============================================================================
"""

import sys
import json
import asyncio
import shlex
import argparse
from fastmcp import Client
from fastmcp.client.transports import StdioTransport


async def fetch_tools(target: str):
    if target.startswith("http://") or target.startswith("https://"):
        print(f"[*] Connecting to remote SSE MCP Server: {target} ...", file=sys.stderr)
        client = Client(target)
    else:
        print(f"[*] Spawning local Stdio MCP Server via StdioTransport: {target} ...", file=sys.stderr)
        parts = shlex.split(target)
        # StdioTransport 객체를 생성하여 Client의 transport로 공급
        transport = StdioTransport(command=parts[0], args=parts[1:])
        client = Client(transport)
        
    try:
        async with client:
            tools_response = await client.list_tools()
            
            tools_list = []
            for t in tools_response:
                tools_list.append({
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.inputSchema
                })
            
            return {
                "status": "SUCCESS",
                "mcp_target": target,
                "tools": tools_list
            }
    except Exception as e:
        return {
            "status": "ERROR",
            "message": f"Failed to retrieve tools from '{target}': {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(description="Hybrid MCP tools lister")
    parser.add_argument("--url", required=True, help="MCP SSE URL or Stdio command")
    
    args = parser.parse_args()
    
    try:
        result = asyncio.run(fetch_tools(args.url))
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
