---
name: mcp-client
description: Discover and call tools on any Model Context Protocol (MCP) server — a remote SSE URL or a local stdio command — without dedicated MCP tools, by running the bundled list_tools / execute_tool scripts with bash_command.
---

# Model Context Protocol (MCP) Interface Skill

This skill allows the agent to dynamically inspect and interact with Model Context Protocol (MCP) servers.
The `--url` argument accepts either a remote SSE endpoint (`http://` / `https://`) or a local stdio server command (e.g., `"python demo_workspace/demo_mcp_server.py"`).

## Available Sub-Scripts
All scripts are located in `skills/mcp/scripts/` and should be executed with `bash_command` using Python 3 relative to the workspace root.

---

### 🔍 1. list_tools.py
- **Purpose**: Discovers and prints all available tools and parameter schemas from a target MCP server.
- **Script Path**: `skills/mcp/scripts/list_tools.py`
- **Arguments**:
  - `--url [SSE_ENDPOINT_URL | STDIO_COMMAND]`: Target MCP server (Required)
- **Usage Examples**:
  ```bash
  # Remote SSE server
  python skills/mcp/scripts/list_tools.py --url https://mcp-server-wikipedia.up.railway.app/sse

  # Local Stdio server
  python skills/mcp/scripts/list_tools.py --url "python path/to/mcp_server.py"
  ```

---

### ⚡ 2. execute_tool.py
- **Purpose**: Executes a specific tool on a target MCP server with specified JSON arguments.
- **Script Path**: `skills/mcp/scripts/execute_tool.py`
- **Arguments**:
  - `--url [SSE_ENDPOINT_URL | STDIO_COMMAND]`: Target MCP server (Required)
  - `--tool [TOOL_NAME]`: Name of the tool to execute (Required)
  - `--args [JSON_ARGUMENTS]`: JSON string arguments representing tool parameters (Required)
- **Usage Examples**:
  ```bash
  # Remote SSE server
  python skills/mcp/scripts/execute_tool.py \
    --url https://mcp-server-wikipedia.up.railway.app/sse \
    --tool search \
    --args '{"query": "인공지능"}'

  # Local Stdio server
  python skills/mcp/scripts/execute_tool.py \
    --url "python path/to/mcp_server.py" \
    --tool calculate_roi \
    --args '{"revenue": 125000000, "cost": 80000000}'
  ```

Always run `list_tools.py` first to confirm the exact tool name and input schema before calling `execute_tool.py`.
