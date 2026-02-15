# MCP Integration for Paladin with Qwen/Ollama

This guide shows how to integrate MCP servers with your Paladin tool-calling system.

## Installation

### 1. Install MCP Python SDK

```bash
# In your venv
pip install mcp anthropic-mcp --break-system-packages
```

### 2. Install Obsidian MCP Server

```bash
pip install obsidian-mcp --break-system-packages
```

### 3. Add MCP Integration Module

Copy `mcp_integration.py` to your Paladin package directory:

```bash
cp mcp_integration.py /home/grim/Projects/Paladin/paladin/
```

### 4. Update Tool Registry

Replace your current `tool_registry.py` with the updated version that includes MCP support.

## Configuration

### Option 1: Use Claude Desktop Config

The system will automatically read from `~/.config/Claude/claude_desktop_config.json`

### Option 2: Custom MCP Config

Create a custom config file:

```bash
mkdir -p ~/.config/paladin
cp mcp_config.json ~/.config/paladin/mcp_config.json
```

Edit it with your MCP server settings:

```json
{
  "mcpServers": {
    "obsidian": {
      "command": "/home/grim/your-venv/bin/python",
      "args": ["-m", "obsidian_mcp"],
      "env": {
        "OBSIDIAN_VAULT_PATH": "/home/grim/Documents/ObsidianVault"
      }
    }
  }
}
```

### Option 3: Use Environment Variable

```bash
export PALADIN_MCP_CONFIG="/path/to/your/mcp_config.json"
```

## Usage

### In Your Agent Code

Update `agent.py` to pass MCP config:

```python
from .tool_registry import ToolRegistry

# Enable MCP with default config (Claude Desktop)
registry = ToolRegistry(vector_db=vector_db, enable_mcp=True)

# Or use custom config
registry = ToolRegistry(
    vector_db=vector_db, 
    enable_mcp=True,
    mcp_config_path="~/.config/paladin/mcp_config.json"
)

# Disable MCP if needed
registry = ToolRegistry(vector_db=vector_db, enable_mcp=False)

# Load all tools (DB + MCP + Memory)
tools = registry.load_tools()
```

## Qwen/Ollama Tool Format

The integration automatically converts MCP tools to LangChain's StructuredTool format, which works with Qwen and other Ollama models that support function calling.

### Tool Format Example

MCP tools are converted to this format:

```python
{
    "name": "search_notes",
    "description": "Search Obsidian vault for notes",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    }
}
```

### Qwen Model Compatibility

Make sure you're using a Qwen model that supports function calling:
- `qwen2.5:7b` ✓
- `qwen2.5:14b` ✓
- `qwen2.5:32b` ✓
- `qwen2.5-coder:7b` ✓

## Testing

### Test MCP Integration Standalone

```bash
cd /home/grim/Projects/Paladin
python -c "
from paladin.mcp_integration import load_mcp_tools
tools = load_mcp_tools('~/.config/paladin/mcp_config.json')
print(f'Loaded {len(tools)} MCP tools')
for tool in tools:
    print(f'  - {tool.name}: {tool.description}')
"
```

### Test Full System

```bash
cd /home/grim/Projects/Paladin
python -m paladin.test_system
```

## Troubleshooting

### MCP Tools Not Loading

1. **Check MCP package installed**:
   ```bash
   pip list | grep mcp
   ```

2. **Check config path**:
   ```bash
   cat ~/.config/Claude/claude_desktop_config.json
   # or
   cat ~/.config/paladin/mcp_config.json
   ```

3. **Check Python path in config**:
   ```bash
   which python  # Should match path in config
   ```

4. **Test MCP server manually**:
   ```bash
   python -m obsidian_mcp
   # Should start without errors
   ```

### Async Issues

If you get async/event loop errors, the integration uses `asyncio.run()` to handle async MCP calls synchronously for compatibility with LangChain.

### Tool Name Conflicts

If an MCP tool has the same name as a database tool, the database tool takes precedence. Rename one of them to avoid conflicts.

## Advanced: Custom MCP Server

You can create your own MCP servers in Python:

```python
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("my-custom-server")

@app.list_tools()
async def list_tools():
    return [
        Tool(
            name="my_tool",
            description="Does something cool",
            inputSchema={
                "type": "object",
                "properties": {
                    "param": {"type": "string"}
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name, arguments):
    if name == "my_tool":
        result = f"You said: {arguments['param']}"
        return [TextContent(type="text", text=result)]

# Run with: python my_server.py
```

Then add to your MCP config:

```json
{
  "mcpServers": {
    "custom": {
      "command": "python",
      "args": ["/path/to/my_server.py"]
    }
  }
}
```

## Performance Notes

- MCP tools run asynchronously but are wrapped for sync compatibility
- First call to an MCP tool may be slower due to server initialization
- Consider enabling only needed MCP servers to reduce overhead
- Database tools are faster than MCP tools (no IPC overhead)

## Next Steps

1. Test with Obsidian MCP server
2. Try other MCP servers (filesystem, git, etc.)
3. Create custom MCP servers for your specific needs
4. Integrate with your Qwen agent workflow
