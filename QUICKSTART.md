# Paladin v2.0 - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies (2 minutes)
```bash
cd /home/grim/Projects/Paladin
./setup_dependencies.sh
```

### Step 2: Verify Installation (1 minute)
```bash
python3 paladin/test_system.py
```

**Expected**: `✓ ALL TESTS PASSED - System ready to use!`

### Step 3: Run Paladin (immediate)
```bash
python3 Paladin_v2.py
```

---

## 💡 First Commands to Try

```
> /tools
Shows all 10+ available tools

> /templates
Shows workflow templates

> create a file called test.txt with "Hello Paladin"
Tests file creation

> read the file test.txt
Tests file reading

> calculate 25 * 4 + 10
Tests calculator

> find all python files in /home/grim/Projects
Tests file search

> run command: ls -la /tmp
Tests command execution
```

---

## 📚 What You Have

### 10 Built-in Tools
- **create_file** - Create/edit files
- **read_file** - Read files (AI can analyze)
- **list_files** - List directory contents
- **find_files** - Search files by name/content
- **recursive_file_search** - Deep file search
- **zip_files** - Create archives
- **unzip_file** - Extract archives
- **run_command** - Execute shell commands
- **calculate** - Math expressions
- **analyze_file** - Code analysis

### 2 Workflow Templates
- **create_python_project** - Full project scaffolding
- **analyze_python_module** - Code structure analysis

---

## 🎯 Common Tasks

### Create a Project
```
> create a python project called MyAPI in /tmp/myapi with description "REST API service"
```

### Analyze Code
```
> analyze the python file at /home/grim/test.py
```

### Batch Operations
```
> find all .py files in current directory, then analyze each one
```

### Command Execution
```
> run the command git status
> run the command npm install
```

---

## 🔧 Add Your Own Tools

### 1. Add to Database
```bash
cd /home/grim/Desktop/op/TOOL_DB
python3
```

```python
from init_db import add_tool

add_tool(
    name="my_tool",
    category="custom",
    description="My custom tool",
    function_module="paladin.tools",
    function_name="MyTools.my_method",
    parameters=[
        {"name": "input", "type": "str", "required": 1, "description": "Input data"}
    ]
)
```

### 2. Implement Function
Edit `/home/grim/Projects/Paladin/paladin/tools.py`:

```python
class MyTools:
    @staticmethod
    def my_method(input: str) -> str:
        return f"Processed: {input}"
```

### 3. Restart Paladin
Tools are loaded automatically!

---

## 📁 Important Locations

- **Main App**: `/home/grim/Projects/Paladin/Paladin_v2.py`
- **Tool Database**: `/home/grim/Desktop/op/TOOL_DB/tools.db`
- **Vector Database**: `~/.paladin/vector_chat.db`
- **Tool Implementations**: `/home/grim/Projects/Paladin/paladin/tools.py`

---

## ⚡ Pro Tips

1. **Use /tools** to see what's available before asking
2. **Memory persists** across sessions - agent remembers past conversations
3. **Chain operations** - agent can use multiple tools for complex tasks
4. **File analysis** - read_file returns full content for AI reasoning
5. **Safe commands** - dangerous operations are blocked automatically

---

## 🐛 Troubleshooting

### "No module named 'numpy'"
```bash
./setup_dependencies.sh
```

### "Cannot connect to Ollama"
```bash
# Start Ollama in another terminal
ollama serve

# Verify model
ollama list
```

### "Tool not found"
```bash
# Re-populate database
cd /home/grim/Desktop/op/TOOL_DB
python3 populate_tools.py
```

---

## 📖 Full Documentation

- **README.md** - Complete system documentation
- **PROJECT_ANALYSIS.md** - Detailed analysis and architecture
- **TOOL_DB/README.md** - Database schema and examples

---

**Ready to go!** Just run `./setup_dependencies.sh` and `python3 Paladin_v2.py`
