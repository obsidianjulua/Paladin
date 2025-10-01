# Tool Database for Paladin AI

Global tool database system for dynamic tool loading and management.

## Location
`/home/grim/Desktop/op/TOOL_DB/`

## Files

- **tools.db** - SQLite database containing tool definitions, parameters, and templates
- **init_db.py** - Database initialization script
- **populate_tools.py** - Script to populate database with standard tools

## Database Schema

### Tables

1. **tools** - Tool definitions
   - id, name, category, description, function_module, function_name, enabled

2. **tool_parameters** - Parameter schemas for tools
   - id, tool_id, param_name, param_type, required, default_value, description, validation_rule

3. **tool_templates** - Multi-step workflows
   - id, name, description, category, steps (JSON), variables (JSON), enabled

4. **template_parameters** - Parameters for templates
   - id, template_id, param_name, param_type, required, default_value, description

5. **tool_usage** - Execution history for analytics
   - id, tool_name, parameters, result_summary, success, execution_time, timestamp

## Usage

### Initialize Database
```bash
cd /home/grim/Desktop/op/TOOL_DB
python3 init_db.py
```

### Populate with Standard Tools
```bash
python3 populate_tools.py
```

### Add New Tool
```python
from init_db import add_tool

add_tool(
    name="my_tool",
    category="custom",
    description="My custom tool",
    function_module="paladin.tools",
    function_name="MyClass.my_method",
    parameters=[
        {"name": "param1", "type": "str", "required": 1, "description": "First parameter"},
        {"name": "param2", "type": "int", "required": 0, "default": "10", "description": "Optional parameter"}
    ]
)
```

### Add New Template
```python
from init_db import add_template

add_template(
    name="my_workflow",
    description="Multi-step workflow",
    category="automation",
    steps=[
        {
            "tool": "create_file",
            "params": {"path": "{output_path}", "content": "{content}"},
            "store_as": "file_result"
        },
        {
            "tool": "run_command",
            "params": {"command": "ls -l {output_path}"}
        }
    ],
    variables={"content": "default content"},
    parameters=[
        {"name": "output_path", "type": "str", "required": 1, "description": "Output file path"}
    ]
)
```

## Categories

- **filesystem** - File operations (create, read, list, search)
- **compression** - Archive operations (zip, unzip)
- **system** - System commands and utilities
- **analysis** - Code and file analysis
- **project** - Project scaffolding and templates
- **network** - Network operations (future)
- **data** - Data processing (future)

## Tool Implementation

Tools are implemented as Python static methods in `/home/grim/Projects/Paladin/paladin/tools.py`:

```python
class MyToolClass:
    @staticmethod
    def my_tool(param1: str, param2: int = 10) -> str:
        """Tool implementation."""
        return f"Result: {param1} - {param2}"
```

Then register in database:
- function_module: `paladin.tools`
- function_name: `MyToolClass.my_tool`

## Integration with Paladin

The ToolRegistry automatically loads tools from this database when Paladin starts.

```python
from paladin import OllamaToolAgent

agent = OllamaToolAgent()
# Tools are loaded automatically from /home/grim/Desktop/op/TOOL_DB/tools.db
```
