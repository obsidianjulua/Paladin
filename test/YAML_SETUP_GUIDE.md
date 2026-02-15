# YAML MCP Configuration Guide

## Installation

First, install PyYAML to support YAML configs:

```bash
pip install PyYAML --break-system-packages
```

## YAML vs JSON Format

### YAML Format (Recommended - More Readable)
```yaml
name: My MCP Servers
version: 1.0.0
schema: v1

mcpServers:
  - name: obsidian
    command: python
    args:
      - -m
      - obsidian_mcp
    env:
      OBSIDIAN_VAULT_PATH: /home/user/vault

  - name: filesystem
    command: npx
    args:
      - -y
      - "@modelcontextprotocol/server-filesystem"
      - /home/user
    env: {}
```

### JSON Format (Claude Desktop Default)
```json
{
  "mcpServers": {
    "obsidian": {
      "command": "python",
      "args": ["-m", "obsidian_mcp"],
      "env": {
        "OBSIDIAN_VAULT_PATH": "/home/user/vault"
      }
    }
  }
}
```

## Setup Steps

### 1. Create Config Directory

```bash
mkdir -p ~/.config/paladin
```

### 2. Create YAML Config File

```bash
nano ~/.config/paladin/mcp_servers.yaml
```

### 3. Add Your MCP Servers

Copy and customize this template:

```yaml
name: Paladin MCP Servers
version: 1.0.0
schema: v1

mcpServers:
  # Obsidian vault access
  - name: obsidian
    command: /home/grim/venv/bin/python  # Use your venv Python
    args:
      - -m
      - obsidian_mcp
    env:
      OBSIDIAN_VAULT_PATH: /home/grim/Documents/ObsidianVault

  # Filesystem access (requires Node.js)
  - name: filesystem
    command: npx
    args:
      - -y
      - "@modelcontextprotocol/server-filesystem"
      - /home/grim/Documents  # Root directory for access
    env: {}

  # Git operations (requires Node.js)
  - name: git
    command: npx
    args:
      - -y
      - "@modelcontextprotocol/server-git"
      - --repository
      - /home/grim/Projects  # Git repo path
    env: {}
```

## Common MCP Servers (YAML Format)

### Obsidian (Python)
```yaml
- name: obsidian
  command: python
  args:
    - -m
    - obsidian_mcp
  env:
    OBSIDIAN_VAULT_PATH: /path/to/vault
```

### Filesystem (Node.js)
```yaml
- name: filesystem
  command: npx
  args:
    - -y
    - "@modelcontextprotocol/server-filesystem"
    - /allowed/directory
  env: {}
```

### Git (Node.js)
```yaml
- name: git
  command: npx
  args:
    - -y
    - "@modelcontextprotocol/server-git"
    - --repository
    - /path/to/repo
  env: {}
```

### Brave Search (Node.js)
```yaml
- name: brave-search
  command: npx
  args:
    - -y
    - "@modelcontextprotocol/server-brave-search"
  env:
    BRAVE_API_KEY: your-api-key-here
```

### PostgreSQL (Node.js)
```yaml
- name: postgres
  command: npx
  args:
    - -y
    - "@modelcontextprotocol/server-postgres"
  env:
    POSTGRES_CONNECTION_STRING: postgresql://user:pass@localhost/db
```

### GitHub (Node.js)
```yaml
- name: github
  command: npx
  args:
    - -y
    - "@modelcontextprotocol/server-github"
  env:
    GITHUB_PERSONAL_ACCESS_TOKEN: your-token-here
```

### Google Drive (Node.js)
```yaml
- name: gdrive
  command: npx
  args:
    - -y
    - "@modelcontextprotocol/server-gdrive"
  env: {}
```

### Slack (Node.js)
```yaml
- name: slack
  command: npx
  args:
    - -y
    - "@modelcontextprotocol/server-slack"
  env:
    SLACK_BOT_TOKEN: xoxb-your-token
    SLACK_TEAM_ID: T1234567890
```

### Memory/SQLite (Node.js)
```yaml
- name: memory
  command: npx
  args:
    - -y
    - "@modelcontextprotocol/server-memory"
  env: {}
```

### Puppeteer (Browser Automation - Node.js)
```yaml
- name: puppeteer
  command: npx
  args:
    - -y
    - "@modelcontextprotocol/server-puppeteer"
  env: {}
```

### Sequential Thinking (Node.js)
```yaml
- name: sequential-thinking
  command: npx
  args:
    - -y
    - "@modelcontextprotocol/server-sequential-thinking"
  env: {}
```

## Using Custom Python MCP Servers

If you've created your own MCP server in Python:

```yaml
- name: my-custom-server
  command: /home/grim/venv/bin/python
  args:
    - /path/to/my_server.py
  env:
    CUSTOM_CONFIG: value
```

## Usage in Code

### Auto-detect Config File

The system checks these locations in order:
1. `~/.config/paladin/mcp_servers.yaml`
2. `~/.config/paladin/mcp_config.json`
3. `~/.config/Claude/claude_desktop_config.json`

```python
from paladin.mcp_integration import load_mcp_tools

# Auto-detect
tools = load_mcp_tools()
```

### Specify Config Path

```python
# YAML file
tools = load_mcp_tools('~/.config/paladin/mcp_servers.yaml')

# JSON file
tools = load_mcp_tools('~/.config/paladin/mcp_config.json')

# Relative path
tools = load_mcp_tools('./my_servers.yaml')
```

### In Tool Registry

```python
from paladin.tool_registry import ToolRegistry

# Auto-detect config
registry = ToolRegistry(enable_mcp=True)

# Specify YAML config
registry = ToolRegistry(
    enable_mcp=True,
    mcp_config_path='~/.config/paladin/mcp_servers.yaml'
)

# Load all tools
tools = registry.load_tools()
```

## Testing Your YAML Config

### Validate YAML Syntax

```bash
python -c "import yaml; yaml.safe_load(open('~/.config/paladin/mcp_servers.yaml'))"
```

### Test Loading Tools

```bash
cd /home/grim/Projects/Paladin
python -c "
from paladin.mcp_integration import load_mcp_tools
tools = load_mcp_tools('~/.config/paladin/mcp_servers.yaml')
print(f'Loaded {len(tools)} tools:')
for t in tools: print(f'  - {t.name}')
"
```

## YAML Tips

### Environment Variables

```yaml
- name: server
  command: python
  args:
    - -m
    - my_server
  env:
    # String values
    API_KEY: my-secret-key
    
    # Empty dict for no env vars
    # env: {}
    
    # Multiple env vars
    VAR1: value1
    VAR2: value2
```

### Command Paths

```yaml
# Absolute path (recommended for Python in venv)
command: /home/grim/venv/bin/python

# Command in PATH
command: npx

# Relative path (not recommended)
command: ./my_script.sh
```

### Arrays in YAML

```yaml
# Multi-line format (more readable)
args:
  - -y
  - "@package/name"
  - --option
  - value

# Inline format
args: ["-y", "@package/name", "--option", "value"]
```

### Comments

```yaml
# This is a comment
mcpServers:
  # Obsidian server for note-taking
  - name: obsidian
    command: python  # Use venv python
    args:
      - -m
      - obsidian_mcp
    # Environment variables
    env:
      OBSIDIAN_VAULT_PATH: /path/to/vault
```

## Troubleshooting

### "No module named 'yaml'"
```bash
pip install PyYAML --break-system-packages
```

### YAML Syntax Error
- Check indentation (use 2 spaces, not tabs)
- Ensure proper list format with `-`
- Validate at: https://www.yamllint.com/

### Server Not Connecting
1. Check command path: `which python` or `which npx`
2. Test command manually: `python -m obsidian_mcp`
3. Check environment variables are set correctly
4. Look at logs for specific error messages

### Config Not Found
```bash
# Check file exists
ls -la ~/.config/paladin/mcp_servers.yaml

# Check permissions
chmod 644 ~/.config/paladin/mcp_servers.yaml
```

## Converting JSON to YAML

If you have an existing JSON config:

```python
import json
import yaml

# Load JSON
with open('mcp_config.json', 'r') as f:
    config = json.load(f)

# Convert mcpServers from dict to list
servers = []
for name, server in config['mcpServers'].items():
    server['name'] = name
    servers.append(server)

# Create YAML format
yaml_config = {
    'name': 'Converted MCP Servers',
    'version': '1.0.0',
    'schema': 'v1',
    'mcpServers': servers
}

# Save as YAML
with open('mcp_servers.yaml', 'w') as f:
    yaml.dump(yaml_config, f, default_flow_style=False)
```

## Example Full Configuration

```yaml
name: Complete Paladin MCP Setup
version: 1.0.0
schema: v1
description: All MCP servers for Paladin AI system

mcpServers:
  # Knowledge Management
  - name: obsidian
    command: /home/grim/venv/bin/python
    args: ["-m", "obsidian_mcp"]
    env:
      OBSIDIAN_VAULT_PATH: /home/grim/Documents/Vault

  # File Operations
  - name: filesystem
    command: npx
    args: ["-y", "@modelcontextprotocol/server-filesystem", "/home/grim"]
    env: {}

  # Version Control
  - name: git
    command: npx
    args: ["-y", "@modelcontextprotocol/server-git", "--repository", "/home/grim/Projects"]
    env: {}

  # Web Search
  - name: brave-search
    command: npx
    args: ["-y", "@modelcontextprotocol/server-brave-search"]
    env:
      BRAVE_API_KEY: ${BRAVE_API_KEY}  # Use env var

  # Memory/Context
  - name: memory
    command: npx
    args: ["-y", "@modelcontextprotocol/server-memory"]
    env: {}
```
