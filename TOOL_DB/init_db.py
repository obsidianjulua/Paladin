#!/usr/bin/env python3
"""
Tool Database Initialization and Management
Creates and manages the global tool database for Paladin AI
"""

import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

DB_PATH = "/home/grim/Desktop/op/TOOL_DB/tools.db"


def init_tool_database():
    """Initialize the tool database with schema."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Tools table - stores tool definitions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tools (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            function_module TEXT NOT NULL,
            function_name TEXT NOT NULL,
            enabled INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Tool parameters table - stores parameter schemas for each tool
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tool_parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_id INTEGER NOT NULL,
            param_name TEXT NOT NULL,
            param_type TEXT NOT NULL,
            required INTEGER DEFAULT 0,
            default_value TEXT,
            description TEXT,
            validation_rule TEXT,
            FOREIGN KEY (tool_id) REFERENCES tools(id) ON DELETE CASCADE,
            UNIQUE(tool_id, param_name)
        )
    """)

    # Tool templates - stores multi-step workflows
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tool_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            description TEXT NOT NULL,
            category TEXT NOT NULL,
            steps TEXT NOT NULL,
            variables TEXT,
            enabled INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)

    # Template parameters - parameters for templates
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS template_parameters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template_id INTEGER NOT NULL,
            param_name TEXT NOT NULL,
            param_type TEXT NOT NULL,
            required INTEGER DEFAULT 0,
            default_value TEXT,
            description TEXT,
            FOREIGN KEY (template_id) REFERENCES tool_templates(id) ON DELETE CASCADE,
            UNIQUE(template_id, param_name)
        )
    """)

    # Tool execution history - for analytics and learning
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tool_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            parameters TEXT,
            result_summary TEXT,
            success INTEGER,
            execution_time REAL,
            timestamp TEXT NOT NULL
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_category ON tools(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_tools_enabled ON tools(enabled)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_templates_category ON tool_templates(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_usage_tool ON tool_usage(tool_name, timestamp)")

    conn.commit()
    conn.close()
    print(f"✓ Tool database initialized at {DB_PATH}")


def add_tool(name: str, category: str, description: str, function_module: str,
             function_name: str, parameters: List[Dict[str, Any]]) -> int:
    """Add a new tool to the database."""
    from datetime import datetime

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()

    try:
        # Insert tool
        cursor.execute("""
            INSERT INTO tools (name, category, description, function_module, function_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, category, description, function_module, function_name, timestamp, timestamp))

        tool_id = cursor.lastrowid

        # Insert parameters
        for param in parameters:
            cursor.execute("""
                INSERT INTO tool_parameters (tool_id, param_name, param_type, required, default_value, description, validation_rule)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                tool_id,
                param['name'],
                param['type'],
                param.get('required', 0),
                param.get('default'),
                param.get('description', ''),
                param.get('validation_rule')
            ))

        conn.commit()
        print(f"✓ Added tool: {name}")
        return tool_id

    except sqlite3.IntegrityError as e:
        print(f"✗ Tool {name} already exists or error: {e}")
        return -1
    finally:
        conn.close()


def add_template(name: str, description: str, category: str, steps: List[Dict[str, Any]],
                 variables: Dict[str, Any], parameters: List[Dict[str, Any]]) -> int:
    """Add a new tool template (workflow) to the database."""
    from datetime import datetime

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    timestamp = datetime.now().isoformat()

    try:
        cursor.execute("""
            INSERT INTO tool_templates (name, description, category, steps, variables, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, description, category, json.dumps(steps), json.dumps(variables), timestamp, timestamp))

        template_id = cursor.lastrowid

        # Insert template parameters
        for param in parameters:
            cursor.execute("""
                INSERT INTO template_parameters (template_id, param_name, param_type, required, default_value, description)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                template_id,
                param['name'],
                param['type'],
                param.get('required', 0),
                param.get('default'),
                param.get('description', '')
            ))

        conn.commit()
        print(f"✓ Added template: {name}")
        return template_id

    except sqlite3.IntegrityError as e:
        print(f"✗ Template {name} already exists or error: {e}")
        return -1
    finally:
        conn.close()


def list_tools(category: Optional[str] = None, enabled_only: bool = True) -> List[Dict[str, Any]]:
    """List all tools, optionally filtered by category."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = "SELECT id, name, category, description, function_module, function_name FROM tools WHERE 1=1"
    params = []

    if enabled_only:
        query += " AND enabled = 1"

    if category:
        query += " AND category = ?"
        params.append(category)

    cursor.execute(query, params)
    tools = []

    for row in cursor.fetchall():
        tool_id, name, cat, desc, module, func = row

        # Get parameters
        cursor.execute("""
            SELECT param_name, param_type, required, default_value, description, validation_rule
            FROM tool_parameters WHERE tool_id = ?
        """, (tool_id,))

        params_list = []
        for param_row in cursor.fetchall():
            params_list.append({
                'name': param_row[0],
                'type': param_row[1],
                'required': bool(param_row[2]),
                'default': param_row[3],
                'description': param_row[4],
                'validation_rule': param_row[5]
            })

        tools.append({
            'id': tool_id,
            'name': name,
            'category': cat,
            'description': desc,
            'function_module': module,
            'function_name': func,
            'parameters': params_list
        })

    conn.close()
    return tools


def list_templates(category: Optional[str] = None) -> List[Dict[str, Any]]:
    """List all tool templates."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    query = "SELECT id, name, description, category, steps, variables FROM tool_templates WHERE enabled = 1"
    params = []

    if category:
        query += " AND category = ?"
        params.append(category)

    cursor.execute(query, params)
    templates = []

    for row in cursor.fetchall():
        template_id, name, desc, cat, steps_json, vars_json = row

        # Get parameters
        cursor.execute("""
            SELECT param_name, param_type, required, default_value, description
            FROM template_parameters WHERE template_id = ?
        """, (template_id,))

        params_list = []
        for param_row in cursor.fetchall():
            params_list.append({
                'name': param_row[0],
                'type': param_row[1],
                'required': bool(param_row[2]),
                'default': param_row[3],
                'description': param_row[4]
            })

        templates.append({
            'id': template_id,
            'name': name,
            'description': desc,
            'category': cat,
            'steps': json.loads(steps_json),
            'variables': json.loads(vars_json) if vars_json else {},
            'parameters': params_list
        })

    conn.close()
    return templates


if __name__ == "__main__":
    init_tool_database()
