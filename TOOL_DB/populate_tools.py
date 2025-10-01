#!/usr/bin/env python3
"""
Populate the tool database with initial tools
Run this after init_db.py to add tools
"""

from init_db import add_tool, add_template

def populate_file_tools():
    """Add file system tools."""
    print("Adding file system tools...")

    add_tool(
        name="create_file",
        category="filesystem",
        description="Create or overwrite a file with content",
        function_module="paladin.tools",
        function_name="FileTools.create_file",
        parameters=[
            {"name": "path", "type": "str", "required": 1, "description": "File path"},
            {"name": "content", "type": "str", "required": 1, "description": "File content"}
        ]
    )

    add_tool(
        name="read_file",
        category="filesystem",
        description="Read file content for analysis or information retrieval",
        function_module="paladin.tools",
        function_name="FileTools.read_file",
        parameters=[
            {"name": "path", "type": "str", "required": 1, "description": "File path to read"},
            {"name": "max_chars", "type": "int", "required": 0, "default": "5000", "description": "Max characters to read"}
        ]
    )

    add_tool(
        name="list_files",
        category="filesystem",
        description="List files in a directory with optional pattern matching",
        function_module="paladin.tools",
        function_name="FileTools.list_files",
        parameters=[
            {"name": "directory", "type": "str", "required": 1, "description": "Directory path"},
            {"name": "pattern", "type": "str", "required": 0, "default": "*", "description": "Glob pattern (e.g., *.py)"}
        ]
    )

    add_tool(
        name="find_files",
        category="filesystem",
        description="Search for files by name or content across directories",
        function_module="paladin.tools",
        function_name="FileTools.find_files",
        parameters=[
            {"name": "query", "type": "str", "required": 1, "description": "Search query"},
            {"name": "search_directories", "type": "list", "required": 0, "default": None, "description": "Directories to search"},
            {"name": "max_results", "type": "int", "required": 0, "default": "20", "description": "Max results"}
        ]
    )

    add_tool(
        name="recursive_file_search",
        category="filesystem",
        description="Recursively search directory tree for files matching pattern",
        function_module="paladin.tools",
        function_name="FileTools.recursive_file_search",
        parameters=[
            {"name": "directory", "type": "str", "required": 1, "description": "Root directory"},
            {"name": "pattern", "type": "str", "required": 0, "default": "*", "description": "Glob pattern"},
            {"name": "max_results", "type": "int", "required": 0, "default": "50", "description": "Max results"}
        ]
    )


def populate_compression_tools():
    """Add compression tools."""
    print("Adding compression tools...")

    add_tool(
        name="zip_files",
        category="compression",
        description="Compress files and directories into a zip archive",
        function_module="paladin.tools",
        function_name="CompressionTools.zip_files",
        parameters=[
            {"name": "file_paths", "type": "list", "required": 1, "description": "List of file/directory paths"},
            {"name": "output_zip_path", "type": "str", "required": 1, "description": "Output zip file path"}
        ]
    )

    add_tool(
        name="unzip_file",
        category="compression",
        description="Extract contents of a zip archive",
        function_module="paladin.tools",
        function_name="CompressionTools.unzip_file",
        parameters=[
            {"name": "zip_path", "type": "str", "required": 1, "description": "Zip file path"},
            {"name": "output_directory", "type": "str", "required": 1, "description": "Extraction directory"}
        ]
    )


def populate_system_tools():
    """Add system tools."""
    print("Adding system tools...")

    add_tool(
        name="run_command",
        category="system",
        description="Execute a shell command and capture output. Safe commands only.",
        function_module="paladin.tools",
        function_name="SystemTools.run_command",
        parameters=[
            {"name": "command", "type": "str", "required": 1, "description": "Shell command to execute"},
            {"name": "capture_output", "type": "bool", "required": 0, "default": "true", "description": "Capture output"},
            {"name": "timeout", "type": "int", "required": 0, "default": "30", "description": "Timeout in seconds"}
        ]
    )

    add_tool(
        name="calculate",
        category="system",
        description="Safely evaluate mathematical expressions",
        function_module="paladin.tools",
        function_name="SystemTools.calculate",
        parameters=[
            {"name": "expression", "type": "str", "required": 1, "description": "Math expression (e.g., '5 + 3 * 2')"}
        ]
    )


def populate_analysis_tools():
    """Add analysis tools."""
    print("Adding analysis tools...")

    add_tool(
        name="analyze_file",
        category="analysis",
        description="Analyze file structure and content. Supports summary, structure, dependencies, security analysis.",
        function_module="paladin.tools",
        function_name="AnalysisTools.analyze_file",
        parameters=[
            {"name": "path", "type": "str", "required": 1, "description": "File path to analyze"},
            {"name": "analysis_type", "type": "str", "required": 0, "default": "summary", "description": "Type: summary, structure, dependencies, security"}
        ]
    )


def populate_templates():
    """Add tool templates (workflows)."""
    print("Adding tool templates...")

    # Example: Create Python Project template
    add_template(
        name="create_python_project",
        description="Create a complete Python project structure with common files",
        category="project",
        steps=[
            {
                "tool": "run_command",
                "params": {"command": "mkdir -p {project_path}"},
                "store_as": "mkdir_result"
            },
            {
                "tool": "create_file",
                "params": {
                    "path": "{project_path}/README.md",
                    "content": "# {project_name}\n\n{description}\n\n## Installation\n\n```bash\npip install -r requirements.txt\n```\n"
                }
            },
            {
                "tool": "create_file",
                "params": {
                    "path": "{project_path}/requirements.txt",
                    "content": "# Add your dependencies here\n"
                }
            },
            {
                "tool": "create_file",
                "params": {
                    "path": "{project_path}/.gitignore",
                    "content": "__pycache__/\n*.py[cod]\n*$py.class\n.env\nvenv/\n*.log\n"
                }
            },
            {
                "tool": "create_file",
                "params": {
                    "path": "{project_path}/main.py",
                    "content": "#!/usr/bin/env python3\n\"\"\"{project_name}\"\"\"\n\ndef main():\n    print('{project_name} running...')\n\nif __name__ == '__main__':\n    main()\n"
                }
            }
        ],
        variables={
            "project_name": "MyProject",
            "description": "A new Python project"
        },
        parameters=[
            {"name": "project_path", "type": "str", "required": 1, "description": "Root path for project"},
            {"name": "project_name", "type": "str", "required": 1, "description": "Name of the project"},
            {"name": "description", "type": "str", "required": 0, "default": "A new Python project", "description": "Project description"}
        ]
    )

    # Example: Code Analysis template
    add_template(
        name="analyze_python_module",
        description="Comprehensive analysis of a Python module: structure, dependencies, and code quality",
        category="analysis",
        steps=[
            {
                "tool": "read_file",
                "params": {"path": "{file_path}"},
                "store_as": "file_content"
            },
            {
                "tool": "analyze_file",
                "params": {"path": "{file_path}", "analysis_type": "structure"},
                "store_as": "structure_analysis"
            },
            {
                "tool": "analyze_file",
                "params": {"path": "{file_path}", "analysis_type": "summary"},
                "store_as": "summary"
            }
        ],
        variables={},
        parameters=[
            {"name": "file_path", "type": "str", "required": 1, "description": "Path to Python file"}
        ]
    )


def main():
    """Populate all tools and templates."""
    print("=" * 60)
    print("Populating Tool Database")
    print("=" * 60)

    populate_file_tools()
    populate_compression_tools()
    populate_system_tools()
    populate_analysis_tools()
    populate_templates()

    print("=" * 60)
    print("✓ Tool database populated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
