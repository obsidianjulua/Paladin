#!/usr/bin/env python3
"""
Tool Registry - Dynamic Tool Loading from Database + MCP Servers
Manages tool discovery, validation, and execution
"""

import json
import sqlite3
import importlib
import logging
import os
from typing import Dict, Any, List, Optional, Callable, Type
from pathlib import Path
from langchain_core.tools import Tool, StructuredTool
from pydantic import create_model, Field, BaseModel

logger = logging.getLogger(__name__)

# Dynamic path resolution for the tool database
def _find_tool_db() -> str:
    env_path = os.environ.get("PALADIN_TOOL_DB")
    if env_path:
        return env_path
    
    base_dir = Path(__file__).resolve().parent.parent
    relative_path = base_dir / "TOOL_DB" / "tools.db"
    if relative_path.exists():
        return str(relative_path)
        
    cwd_path = Path.cwd() / "TOOL_DB" / "tools.db"
    if cwd_path.exists():
        return str(cwd_path)

    return "/home/grim/Desktop/op/TOOL_DB/tools.db"

TOOL_DB_PATH = _find_tool_db()


class ToolRegistry:
    """Manages dynamic tool loading from the global tool database and MCP servers."""

    def __init__(self, db_path: str = TOOL_DB_PATH, vector_db=None, enable_mcp: bool = True, mcp_config_path: str = None):
        self.db_path = db_path or TOOL_DB_PATH
        self.vector_db = vector_db
        self.enable_mcp = enable_mcp
        self.mcp_config_path = mcp_config_path
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
            module = importlib.import_module(module_path)
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

    def _create_pydantic_model(self, tool_name: str, parameters: List[Dict[str, Any]]) -> Type[BaseModel]:
        """Dynamically creates a Pydantic model for tool parameters."""
        fields = {}
        
        type_map = {
            'str': str, 'string': str,
            'int': int, 'integer': int,
            'float': float, 'number': float,
            'bool': bool, 'boolean': bool,
            'list': list, 'array': list,
            'dict': dict, 'object': dict
        }

        for param in parameters:
            name = param['name']
            py_type = type_map.get(param['type'].lower(), str)
            
            # Setup Field kwargs
            field_kwargs = {'description': param['description']}
            
            if not param['required']:
                # Optional field
                default_val = param['default']
                # Cast default if possible
                if default_val is not None:
                    try:
                        if py_type == int: default_val = int(default_val)
                        elif py_type == float: default_val = float(default_val)
                        elif py_type == bool: default_val = str(default_val).lower() in ('true', '1')
                    except:
                        pass
                else:
                    default_val = None
                
                field_kwargs['default'] = default_val
                # Make type Optional
                fields[name] = (Optional[py_type], Field(**field_kwargs))
            else:
                # Required field
                fields[name] = (py_type, Field(**field_kwargs))
        
        # If no parameters, create empty model
        if not fields:
            return create_model(f"{tool_name}Input")
            
        return create_model(f"{tool_name}Input", **fields)

    def load_tools(self, category: Optional[str] = None, enabled_only: bool = True) -> List[StructuredTool]:
        """Load all tools from the database and MCP servers as LangChain StructuredTool objects."""
        all_tools = []
        
        # Load database tools
        db_tools = self._load_database_tools(category, enabled_only)
        all_tools.extend(db_tools)
        
        # Load memory tools if vector_db is available
        if self.vector_db:
            memory_tools = self._create_memory_tools()
            all_tools.extend(memory_tools)
        
        # Load MCP tools
        if self.enable_mcp:
            try:
                from .mcp_integration import load_mcp_tools
                mcp_tools = load_mcp_tools(self.mcp_config_path)
                all_tools.extend(mcp_tools)
                logger.info(f"Loaded {len(mcp_tools)} tools from MCP servers")
            except ImportError:
                logger.warning("MCP integration module not found. Install 'mcp' package: pip install mcp")
            except Exception as e:
                logger.error(f"Error loading MCP tools: {e}")
        
        logger.info(f"Total tools loaded: {len(all_tools)} (DB: {len(db_tools)}, MCP: {len(all_tools) - len(db_tools) - (len(self._create_memory_tools()) if self.vector_db else 0)})")
        return all_tools

    def _load_database_tools(self, category: Optional[str] = None, enabled_only: bool = True) -> List[StructuredTool]:
        """Load tools from the SQLite database."""
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
                SELECT param_name, param_type, required, default_value, description
                FROM tool_parameters WHERE tool_id = ?
            """, (tool_id,))

            params_list = []
            for param_row in cursor.fetchall():
                params_list.append({
                    'name': param_row[0],
                    'type': param_row[1],
                    'required': bool(param_row[2]),
                    'default': param_row[3],
                    'description': param_row[4]
                })

            # Get actual python function
            func = self._get_tool_function(module, func_name)
            if func is None:
                continue

            # Create Pydantic Schema
            input_model = self._create_pydantic_model(name, params_list)

            # Create Structured Tool
            tool = StructuredTool.from_function(
                func=func,
                name=name,
                description=desc,
                args_schema=input_model
            )

            tools.append(tool)

        conn.close()
        logger.info(f"Loaded {len(tools)} tools from database")
        return tools

    def _create_memory_tools(self) -> List[Tool]:
        """Create memory-related tools."""
        
        # 1. Search Memory
        class SearchMemoryInput(BaseModel):
            query: str = Field(description="The search query to find relevant history.")

        def search_memory_func(query: str) -> str:
            results = self.vector_db.similarity_search(
                query,
                session_id=getattr(self.vector_db, '_current_session', None),
                include_other_sessions=True
            )
            if not results: return "No information found."
            formatted = []
            for item in results:
                if item['source'] == 'chat':
                    formatted.append(f"[{item['type'].upper()}]: {item['content']}")
                elif item['source'] == 'tool':
                    formatted.append(f"[TOOL {item['tool']}]: {item['result']}")
                elif item['source'] == 'document':
                    formatted.append(f"[DOC {item['file_path']}]: {item['content'][:200]}...")
                elif item['source'] == 'fact':
                    formatted.append(f"[FACT]: {item['content']}")
            return "\n---\n".join(formatted)

        # 2. Ingest Document
        class IngestDocInput(BaseModel):
            file_path: str = Field(description="Path to the file to ingest.")

        def ingest_func(file_path: str) -> str:
            return self.vector_db.ingest_document(file_path)

        # 3. Ingest Directory
        class IngestDirInput(BaseModel):
            directory_path: str = Field(description="Path to the directory to ingest.")
            recursive: bool = Field(default=True, description="Whether to search recursively.")

        def ingest_dir_func(directory_path: str, recursive: bool = True) -> str:
            return self.vector_db.ingest_directory(directory_path, recursive=recursive)

        # 4. List Documents
        class ListDocsInput(BaseModel):
            pass

        def list_docs_func() -> str:
            docs = self.vector_db.list_documents()
            if not docs:
                return "No documents currently ingested."
            return "Ingested Documents:\n" + "\n".join(docs)

        # 5. Save Fact
        class SaveFactInput(BaseModel):
            fact: str = Field(description="The fact or information to remember.")

        def save_fact_func(fact: str) -> str:
            return self.vector_db.add_fact(fact)

        return [
            StructuredTool.from_function(
                func=search_memory_func,
                name="search_memory",
                description="Search conversation history, facts, and documents.",
                args_schema=SearchMemoryInput
            ),
            StructuredTool.from_function(
                func=ingest_func,
                name="ingest_document",
                description="Ingest a single file into vector memory.",
                args_schema=IngestDocInput
            ),
            StructuredTool.from_function(
                func=ingest_dir_func,
                name="ingest_directory",
                description="Ingest all supported text files in a directory into vector memory.",
                args_schema=IngestDirInput
            ),
            StructuredTool.from_function(
                func=list_docs_func,
                name="list_documents",
                description="List all currently ingested documents in the knowledge base.",
                args_schema=ListDocsInput
            ),
            StructuredTool.from_function(
                func=save_fact_func,
                name="save_fact",
                description="Save a permanent fact or user preference to long-term memory.",
                args_schema=SaveFactInput
            )
        ]
