# Registered MCP Server Catalog

This catalog registers active Model Context Protocol (MCP) servers. The Universal MCP client scripts can handle local stdio execution commands.

---

### 💿 1. Wikipedia Local Stdio Server (Active)
- **Description**: Spawns a local Wikipedia MCP server process on the host via Node.js/npx. Highly stable, zero server traffic limitations.
- **Connection Command**: `npx -y wikipedia-mcp`
- **Supported Tools**:
  - `search` (Query keyword search. Arguments: `{"query": "string"}`)
  - `readArticle` (Get full page content. Arguments: `{"title": "string"}`)
