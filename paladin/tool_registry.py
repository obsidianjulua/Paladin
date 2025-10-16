#!/usr/bin/env python3
"""
Tool Registry - Dynamic Tool Loading from Database
Manages tool discovery, validation, and execution
"""

import json
import sqlite3
import importlib
import logging
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
from langchain.agents import Tool

logger = logging.getLogger(__name__)

TOOL_DB_PATH = "/home/grim/Desktop/Projects/Paladin/TOOL_DB/tools.db"


class ToolRegistry:
    """Manages dynamic tool loading from the global tool database."""

    def __init__(self, db_path: str = TOOL_DB_PATH, vector_db=None):
        self.db_path = db_path or TOOL_DB_PATH
        self.vector_db = vector_db
        self.tool_cache: Dict[str, Callable] = {}
        self._verify_database()

    def _verify_database(self):
        """Verify the tool database exists."""
        if not Path(self.db_path).exists():
            raise FileNotFoundError(
                f"Tool database not found at {self.db_path}. "
                f"Run init_db.py to initialize it first."
            )

    def _get_tool_function(self, module_path: str, function_name: str) -> Optional[Callable]:
        """Dynamically import and return a tool function."""
        cache_key = f"{module_path}.{function_name}"

        if cache_key in self.tool_cache:
            return self.tool_cache[cache_key]

        try:
            # Import the module
            module = importlib.import_module(module_path)

            # Get the function from the module
            # Support both Class.method and direct function references
            if '.' in function_name:
                class_name, method_name = function_name.split('.', 1)
                tool_class = getattr(module, class_name)
                func = getattr(tool_class, method_name)
            else:
                func = getattr(module, function_name)

            self.tool_cache[cache_key] = func
            return func

        except Exception as e:
            logger.error(f"Failed to load tool function {module_path}.{function_name}: {e}")
            return None

    def _create_tool_wrapper(self, tool_def: Dict[str, Any]) -> Optional[Callable]:
        """Creates a wrapper function for a tool that handles parameter parsing."""
        func = self._get_tool_function(tool_def['function_module'], tool_def['function_name'])

        if func is None:
            return None

        parameters = tool_def['parameters']

        def wrapper(params_input: str) -> str:
            """Wrapper that parses JSON input and calls the tool function."""
            try:
                # Handle simple string input for single-parameter tools
                if len(parameters) == 1 and not params_input.strip().startswith('{'):
                    param_name = parameters[0]['name']
                    parsed_params = {param_name: params_input}
                else:
                    # Parse JSON input
                    parsed_params = json.loads(params_input) if params_input.strip() else {}

                # Validate required parameters
                for param in parameters:
                    if param['required'] and param['name'] not in parsed_params:
                        return f"Error: Missing required parameter '{param['name']}'"

                # Apply defaults
                for param in parameters:
                    if param['name'] not in parsed_params and param['default'] is not None:
                        parsed_params[param['name']] = param['default']

                # Call the actual tool function
                result = func(**parsed_params)
                return result

            except json.JSONDecodeError:
                return "Error: Invalid JSON format for parameters"
            except TypeError as e:
                return f"Error: Invalid parameters - {e}"
            except Exception as e:
                return f"Error executing tool: {e}"

        return wrapper

    def load_tools(self, category: Optional[str] = None, enabled_only: bool = True) -> List[Tool]:
        """Load all tools from the database as LangChain Tool objects."""
        conn = sqlite3.connect(self.db_path)
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
            tool_id, name, cat, desc, module, func_name = row

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

            tool_def = {
                'id': tool_id,
                'name': name,
                'category': cat,
                'description': desc,
                'function_module': module,
                'function_name': func_name,
                'parameters': params_list
            }

            # Create wrapper function
            wrapper_func = self._create_tool_wrapper(tool_def)

            if wrapper_func is None:
                logger.warning(f"Skipping tool '{name}' - failed to load function")
                continue

            # Build description with parameter info
            param_desc = self._build_parameter_description(params_list)
            full_description = f"{desc}\n{param_desc}"

            # Create LangChain Tool
            langchain_tool = Tool(
                name=name,
                func=wrapper_func,
                description=full_description
            )

            tools.append(langchain_tool)

        conn.close()

        # Add memory search tool if vector_db is available
        if self.vector_db:
            tools.append(self._create_memory_tool())

        logger.info(f"Loaded {len(tools)} tools from database")
        return tools

    def _build_parameter_description(self, parameters: List[Dict[str, Any]]) -> str:
        """Build a description of parameters for the tool."""
        if not parameters:
            return "Input: No parameters required"

        if len(parameters) == 1 and parameters[0]['type'] in ['str', 'string']:
            return f"Input: Simple string ({parameters[0]['description']})"

        # Multiple parameters - require JSON
        param_examples = []
        for param in parameters:
            example_val = self._get_example_value(param['type'])
            param_examples.append(f'"{param["name"]}": {example_val}')

        example_json = "{" + ", ".join(param_examples) + "}"
        return f"Input: JSON string like {example_json}"

    def _get_example_value(self, param_type: str) -> str:
        """Get example value for a parameter type."""
        type_examples = {
            'str': '"value"',
            'string': '"value"',
            'int': '42',
            'float': '3.14',
            'bool': 'true',
            'list': '["item1", "item2"]',
            'dict': '{"key": "value"}'
        }
        return type_examples.get(param_type.lower(), '"value"')

    def _create_memory_tool(self) -> Tool:
        """Create the memory search tool using the vector database."""
        def search_memory(query: str) -> str:
            """Search conversation and tool execution history."""
            results = self.vector_db.similarity_search(
                query,
                session_id=getattr(self.vector_db, '_current_session', None),
                include_other_sessions=True
            )

            if not results:
                return "No information found in memory."

            formatted_results = []
            for item in results:
                if item['source'] == 'chat':
                    formatted_results.append(
                        f"[{item['session_id'][:6]} {item['type'].upper()}]: {item['content']}"
                    )
                elif item['source'] == 'tool_execution':
                    try:
                        params = json.loads(item['parameters']) if item.get('parameters') else {}
                    except Exception:
                        params = item.get('parameters')
                    formatted_results.append(
                        f"[{item['session_id'][:6]} TOOL {item['tool'].upper()}]: "
                        f"Params: {params}, Result: {item['result']}"
                    )

            return "Retrieved Memory:\n---\n" + "\n---\n".join(formatted_results)

        return Tool(
            name="search_memory",
            func=search_memory,
            description="Search conversation and tool execution history. Input: simple string query"
        )

    def load_templates(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load tool templates (workflows) from the database."""
        conn = sqlite3.connect(self.db_path)
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

    def execute_template(self, template_name: str, user_params: Dict[str, Any],
                        agent_executor, session_id: str) -> Dict[str, Any]:
        """Execute a tool template (workflow) with user parameters."""
        # Load the specific template
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, steps, variables FROM tool_templates
            WHERE name = ? AND enabled = 1
        """, (template_name,))

        row = cursor.fetchone()
        conn.close()

        if not row:
            return {"success": False, "message": f"Template '{template_name}' not found"}

        template_id, steps_json, vars_json = row
        steps = json.loads(steps_json)
        variables = json.loads(vars_json) if vars_json else {}

        # Merge user params with default variables
        context = {**variables, **user_params}

        # Execute each step
        results = []
        for i, step in enumerate(steps):
            tool_name = step.get('tool')
            params_template = step.get('params', {})

            # Substitute variables in parameters
            params = self._substitute_variables(params_template, context)

            # Execute the step
            try:
                # Find the tool and execute it
                step_result = f"Executing {tool_name} with {params}"
                results.append({
                    'step': i + 1,
                    'tool': tool_name,
                    'params': params,
                    'result': step_result
                })

                # Store result in context for next steps
                if step.get('store_as'):
                    context[step['store_as']] = step_result

            except Exception as e:
                return {
                    "success": False,
                    "message": f"Error at step {i+1}: {e}",
                    "results": results
                }

        return {"success": True, "results": results, "context": context}

    def _substitute_variables(self, template: Any, context: Dict[str, Any]) -> Any:
        """Recursively substitute variables in a template."""
        if isinstance(template, str):
            # Replace {variable_name} with context value
            for key, value in context.items():
                template = template.replace(f"{{{key}}}", str(value))
            return template
        elif isinstance(template, dict):
            return {k: self._substitute_variables(v, context) for k, v in template.items()}
        elif isinstance(template, list):
            return [self._substitute_variables(item, context) for item in template]
        else:
            return template
