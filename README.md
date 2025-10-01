# Paladin AI - Modular Tool-Calling System

Version 2.0 - Database-driven tool management with template support

## Overview

Paladin is an AI agent system built on Ollama that uses a database-driven approach to manage hundreds of tools and complex workflows. Tools are stored in a global database, making them easily accessible and maintainable.

## Architecture

### Modular Structure
```
/home/grim/Projects/Paladin/
├── Paladin.py              # Original monolithic version
├── Paladin_v2.py           # New modular entry point
├── paladin/                # Package directory
│   ├── __init__.py
│   ├── agent.py            # Main agent implementation
│   ├── chat_interface.py   # Interactive chat interface
│   ├── tool_registry.py    # Dynamic tool loading from database
│   ├── tools.py            # Tool implementations
│   └── vector_db.py        # Vector database for chat history
```

### Global Tool Database
```
/home/grim/Desktop/op/TOOL_DB/
├── tools.db                # SQLite database
├── init_db.py              # Database initialization
├── populate_tools.py       # Populate with standard tools
└── README.md               # Database documentation
```

## Features

### 1. Database-Driven Tools
- **100+ tools** can be maintained in a central database
- Tools are loaded dynamically at runtime
- Easy to add/remove/update tools without editing code
- Parameter validation and type checking
- Execution history tracking

### 2. Tool Templates (Workflows)
- Define multi-step workflows in the database
- Chain tools together with variable substitution
- Example: `create_python_project` template creates full project structure
- Templates support conditional execution and error handling

### 3. Enhanced Tool Capabilities
- **File Operations**: Create, read, list, search files with content analysis
- **Command Execution**: Run shell commands with output capture
- **File Analysis**: Structure analysis, dependency detection for Python files
- **Compression**: Zip/unzip operations
- **Memory Search**: Vector-based search through conversation history

### 4. Separated Storage
- Vector database for chat history: `~/.paladin/vector_chat.db`
- Tool database: `/home/grim/Desktop/op/TOOL_DB/tools.db`
- Clean separation of concerns

## Installation

### Prerequisites
```bash
pip install langchain langchain-ollama numpy rich sqlite3
```

### Setup
1. Initialize tool database:
```bash
cd /home/grim/Desktop/op/TOOL_DB
python3 init_db.py
python3 populate_tools.py
```

2. Run Paladin:
```bash
cd /home/grim/Projects/Paladin
python3 Paladin_v2.py
```

## Usage

### Basic Chat
```bash
./Paladin_v2.py
> Create a file called test.txt with "Hello World"
> Read the file test.txt
> Find all Python files in the current directory
```

### With Memory Bootstrap
```bash
INIT_SEARCH="python files" python3 Paladin_v2.py
```

### Special Commands
- `/tools` - List all available tools
- `/templates` - List all workflow templates
- `/help` - Show help
- `quit` or `exit` - Exit

## Adding New Tools

### Method 1: Add to Database (Recommended)
```python
cd /home/grim/Desktop/op/TOOL_DB
python3

from init_db import add_tool

add_tool(
    name="convert_case",
    category="text",
    description="Convert text to uppercase or lowercase",
    function_module="paladin.tools",
    function_name="TextTools.convert_case",
    parameters=[
        {"name": "text", "type": "str", "required": 1, "description": "Input text"},
        {"name": "mode", "type": "str", "required": 1, "description": "upper or lower"}
    ]
)
```

### Method 2: Implement in tools.py
```python
# In paladin/tools.py

class TextTools:
    @staticmethod
    def convert_case(text: str, mode: str) -> str:
        """Convert text case."""
        if mode == "upper":
            return text.upper()
        elif mode == "lower":
            return text.lower()
        else:
            return f"Error: Invalid mode '{mode}'"
```

## Adding Templates (Workflows)

```python
from init_db import add_template

add_template(
    name="backup_and_analyze",
    description="Backup a file and analyze its structure",
    category="automation",
    steps=[
        {
            "tool": "zip_files",
            "params": {
                "file_paths": ["{file_path}"],
                "output_zip_path": "{file_path}.backup.zip"
            }
        },
        {
            "tool": "analyze_file",
            "params": {
                "path": "{file_path}",
                "analysis_type": "structure"
            }
        }
    ],
    variables={},
    parameters=[
        {"name": "file_path", "type": "str", "required": 1, "description": "File to backup and analyze"}
    ]
)
```

## Example Workflows

### Create a Python Project
```python
agent.execute_template("create_python_project", {
    "project_path": "/home/user/myproject",
    "project_name": "MyAwesomeProject",
    "description": "An awesome Python project"
})
```

### Analyze Python Module
```python
agent.execute_template("analyze_python_module", {
    "file_path": "/home/user/script.py"
})
```

## Current Tools

### Filesystem (5 tools)
- `create_file` - Create/overwrite files
- `read_file` - Read with AI analysis support
- `list_files` - List with pattern matching
- `find_files` - Search by name/content
- `recursive_file_search` - Recursive pattern search

### Compression (2 tools)
- `zip_files` - Create archives
- `unzip_file` - Extract archives

### System (2 tools)
- `run_command` - Execute shell commands with output capture
- `calculate` - Math expression evaluation

### Analysis (1 tool)
- `analyze_file` - File structure and content analysis

### Memory (1 tool)
- `search_memory` - Vector search through history

## Migration from v1

The original `Paladin.py` is preserved. v2 uses the same underlying LLM and agent logic but with modular architecture:

| v1 (Monolithic) | v2 (Modular) |
|-----------------|--------------|
| Hardcoded tools | Database-driven |
| Single file | Package structure |
| 6-10 tools | Unlimited tools |
| No templates | Workflow templates |
| Fixed location | Configurable paths |

## Configuration

Default settings in `paladin/agent.py`:
```python
DEFAULT_MODEL = "codellama:7b-instruct-q4_K_M"
DEFAULT_BASE_URL = "http://localhost:11434"
```

Default paths in `paladin/tool_registry.py`:
```python
TOOL_DB_PATH = "/home/grim/Desktop/op/TOOL_DB/tools.db"
```

## Extending Paladin

### Add New Tool Categories
1. Create new tool class in `paladin/tools.py`
2. Add tools to database with `add_tool()`
3. Tools automatically available on next run

### Create Custom Templates
1. Design multi-step workflow
2. Add to database with `add_template()`
3. Execute via `agent.execute_template()`

### Add New Analysis Types
Extend `AnalysisTools.analyze_file()` with new analysis types:
- Security scanning
- Performance profiling
- Documentation generation
- Test coverage analysis

## License

Open source - MIT License

## Contributing

To add tools to the global database:
1. Implement in `paladin/tools.py`
2. Register in `/home/grim/Desktop/op/TOOL_DB/populate_tools.py`
3. Run populate script to update database
