#!/usr/bin/env python3
"""
Ollama Tool Execution System with Vector Database Storage
Handles explicit tool execution, chat history, and tool chaining for Ollama models.
"""

import os
import json
import logging
import asyncio
import sqlite3
import hashlib
import zipfile
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from fnmatch import fnmatch

# External Libraries
import numpy as np
from langchain_ollama import ChatOllama
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Setup console
console = Console()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MODEL and URL
DEFAULT_MODEL = "codellama:7b-instruct-q4_K_M"
DEFAULT_BASE_URL = "http://localhost:11434"

# --- VectorDatabase Class ---
class VectorDatabase:
    """Simple vector database using SQLite and hash-based embeddings for offline use."""

    def __init__(self, db_path: str = "vector_chat.db", model_name: str = DEFAULT_MODEL):
        self.db_path = db_path
        self.model_name = model_name
        self.embedding_dim = 64 # Hash Lowered
        # keep a single connection to persist DB and reduce locking; enable WAL for durability
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_database()

    def _init_database(self):
        """Initialize the vector database tables."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                message_type TEXT NOT NULL, -- 'human' or 'assistant'
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                content_hash TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_executions (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                parameters TEXT,
                result TEXT,
                embedding BLOB NOT NULL,
                content_hash TEXT NOT NULL
            )
        """)
        # indexes to speed up searches and session filtering
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_session_id ON chat_history(session_id, id DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_session_id ON tool_executions(session_id, id DESC)")
        self.conn.commit()

    def close(self):
        """Close the persistent database connection."""
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates a simple hash-based embedding for the given text (offline compatible)."""
        text_lower = (text or "").lower()
        if not text_lower:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        embeddings = []
        for i in range(self.embedding_dim):
            rotated_text = text_lower[i % len(text_lower):] + text_lower[:i % len(text_lower)]
            hash_obj = hashlib.md5(f"{rotated_text}_{i}".encode())
            hash_int = int(hash_obj.hexdigest()[:8], 16)
            embedding_val = (hash_int % 2000 - 1000) / 1000.0
            embeddings.append(embedding_val)
        return np.array(embeddings, dtype=np.float32)

    def _insert_record(self, table: str, session_id: str, content: str, *, message_type: Optional[str] = None, tool_name: Optional[str] = None, parameters: Optional[str] = None, result: Optional[str] = None):
        """Inserts a record into the database."""
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        embedding = self._get_embedding(content).tobytes()
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if table == "chat_history":
            cursor.execute("""
                INSERT INTO chat_history (session_id, timestamp, message_type, content, embedding, content_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, timestamp, message_type, content, embedding, content_hash))
        elif table == "tool_executions":
            cursor.execute("""
                INSERT INTO tool_executions (session_id, timestamp, tool_name, parameters, result, embedding, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_id, timestamp, tool_name, parameters, result, embedding, content_hash))
        self.conn.commit()

    def add_message(self, session_id: str, message_type: str, content: str):
        """Adds a chat message to the history."""
        self._insert_record("chat_history", session_id, content, message_type=message_type)

    def add_tool_execution(self, session_id: str, tool_name: str, parameters: Dict[str, Any], result: str):
        """Adds a tool execution record."""
        content = f"Tool: {tool_name}, Parameters: {json.dumps(parameters, ensure_ascii=False)}, Result: {result}"
        self._insert_record(
            "tool_executions",
            session_id,
            content,
            tool_name=tool_name,
            parameters=json.dumps(parameters, ensure_ascii=False),
            result=result
        )

    def similarity_search(self, query: str, session_id: Optional[str] = None, limit: int = 5, threshold: float = 0.3, include_other_sessions: bool = True) -> List[Dict[str, Any]]:
        """Performs a vector similarity search across chat and tool history.
        If include_other_sessions is True, it searches all sessions and prioritizes current session."""
        query_embedding = self._get_embedding(query)
        cursor = self.conn.cursor()
        results = []

        def cosine(a: np.ndarray, b: np.ndarray) -> float:
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0.0:
                return 0.0
            return float(np.dot(a, b) / denom)

        # Build WHERE clause
        where_clause = ""
        params: List[Any] = []
        if not include_other_sessions and session_id:
            where_clause = "WHERE session_id = ?"
            params.append(session_id)

        # Search chat history
        cursor.execute(f"SELECT session_id, message_type, content, embedding FROM chat_history {where_clause} ORDER BY id DESC LIMIT 200", params)
        for sess, message_type, content, emb_bytes in cursor.fetchall():
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = cosine(query_embedding, emb)
            if sim >= threshold:
                results.append({
                    "source": "chat",
                    "session_id": sess,
                    "type": message_type,
                    "content": content,
                    "similarity": sim
                })

        # Search tool history
        cursor.execute(f"SELECT session_id, tool_name, parameters, result, embedding FROM tool_executions {where_clause} ORDER BY id DESC LIMIT 200", params)
        for sess, tool_name, parameters, result, emb_bytes in cursor.fetchall():
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = cosine(query_embedding, emb)
            if sim >= threshold:
                results.append({
                    "source": "tool_execution",
                    "session_id": sess,
                    "tool": tool_name,
                    "parameters": parameters,
                    "result": result,
                    "similarity": sim
                })

        # Prefer current session items first, then by similarity
        if session_id:
            results.sort(key=lambda x: ((x.get("session_id") == session_id), x['similarity']), reverse=True)
        else:
            results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]


# --- EnhancedTools Class ---
class EnhancedTools:
    """A collection of advanced static tools for the agent."""

    _SAFE_COMMANDS = ['ls', 'dir', 'cat', 'echo', 'grep', 'find', 'pwd', 'whoami', 'df',
                      'du', 'head', 'tail']
    _DANGEROUS_COMMANDS = ['format', 'shutdown', 'reboot', 'dd', 'mkfs', 'chown', 'chmod']

    @staticmethod
    def create_file(path: str, content: str) -> str:
        """Creates a new file with the specified content. Creates parent directories if they don't exist."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding='utf-8')
            return f"File created/overwritten successfully at {path}."
        except Exception as e:
            return f"Error creating file: {e}"

    @staticmethod
    def read_file(path: str) -> str:
        """Reads the content of a file, truncating output to the first 1000 characters for brevity."""
        try:
            p = Path(path)
            if not p.is_file():
                return f"Error: File not found at {path}"

            content = p.read_text(encoding='utf-8')
            if len(content) > 1000:
                content = content[:1000] + "\n... (content truncated after 1000 characters)"

            return f"Content of {path}:\n---\n{content}\n---"
        except Exception as e:
            return f"Error reading file: {e}"

    @staticmethod
    def list_files(directory: str, pattern: str = "*") -> str:
        """Lists files and directories in a directory, optionally filtering by pattern (e.g., *.txt)."""
        try:
            p = Path(directory)
            if not p.is_dir():
                return f"Error: Directory not found at {directory}"

            files = []
            for item in p.iterdir():
                if fnmatch(item.name, pattern):
                    files.append(str(item.name) + ('/' if item.is_dir() else ''))

            if not files:
                return f"No files found in '{directory}' matching pattern '{pattern}'."

            return f"Contents of {directory} matching '{pattern}':\n" + "\n".join(files)
        except Exception as e:
            return f"Error listing directory: {e}"

    @staticmethod
    def find_files(query: str, search_directories: List[str] = None) -> str:
        """Enhanced file finding with content search and multiple search strategies."""
        if search_directories is None:
            search_directories = ['.']

        try:
            results = []
            search_terms = query.lower().split()

            for search_dir in search_directories:
                search_path = Path(search_dir)
                if not search_path.exists():
                    continue

                # Search by filename
                for file_path in search_path.rglob('*'):
                    if file_path.is_file():
                        filename_lower = file_path.name.lower()

                        # Check if query terms match filename
                        if any(term in filename_lower for term in search_terms):
                            results.append({
                                'type': 'filename_match',
                                'path': str(file_path),
                                'match': f"Filename contains: {query}"
                            })

                        # Search file content for text files
                        if file_path.suffix.lower() in {'.txt', '.py', '.js', '.html', '.css', '.json', '.xml', '.md',
                                                        '.log', '.cfg', '.ini', '.lua', '.cpp', '.jl', 'Modelfile', '.env'}:
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read(1000).lower()  # Read first 1000 chars
                                    if any(term in content for term in search_terms):
                                        # Find the line with the match
                                        lines = content.split('\n')
                                        match_line = next((line for line in lines if any(term in line for term in search_terms)), "")
                                        results.append({
                                            'type': 'content_match',
                                            'path': str(file_path),
                                            'match': f"Content contains: '{match_line[:100]}...'"
                                        })
                            except Exception:
                                pass  # Skip files that can't be read

            if not results:
                return f"No files found matching '{query}' in directories: {search_directories}"

            # Format results
            formatted_results = []
            for i, result in enumerate(results[:20]):  # Limit to 20 results
                formatted_results.append(f"{i+1}. {result['path']} ({result['type']})\n   {result['match']}")

            return f"Found {len(results)} files matching '{query}':\n" + "\n".join(formatted_results)

        except Exception as e:
            return f"Error finding files: {e}"

    @staticmethod
    def execute_command(command: str) -> str:
        """Executes a SAFE shell command and returns the output. Only pre-approved commands are allowed."""
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return "Error: Command is empty."

        # Safety Check: Block dangerous commands
        if cmd_parts[0] in EnhancedTools._DANGEROUS_COMMANDS:
            return f"Error: Command '{cmd_parts[0]}' is explicitly blocked for safety."

        # Safety Check: Allow only approved commands for non-Python execution tools
        if cmd_parts[0] not in EnhancedTools._SAFE_COMMANDS:
             return f"Error: Command '{cmd_parts[0]}' is not an approved safe shell command. Use the Python tools instead."

        try:
            # Execute command (non-blocking)
            result = os.popen(command).read()
            # Truncate output
            if len(result) > 500:
                result = result[:500] + "\n... (output truncated after 500 characters)"
            return f"Command Output:\n---\n{result}\n---"
        except Exception as e:
            return f"Error executing command: {e}"

    @staticmethod
    def calculate(expression: str) -> str:
        """Safely evaluates a simple mathematical expression."""
        safe_dict = {
            'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'pow': pow,
            'sqrt': lambda x: x**0.5, # simple math functions
            '__builtins__': None
        }
        try:
            # Sanitize input: remove dangerous characters or sequences
            safe_expression = expression.replace(';', '').replace('__', '')
            result = str(eval(safe_expression, {"__builtins__": None}, safe_dict))
            return f"Calculation Result: {result}"
        except Exception as e:
            return f"Error evaluating expression: {e}"

    @staticmethod
    def zip_files(file_paths: List[str], output_zip_path: str) -> str:
        """Compresses a list of files or directories into a single zip archive."""
        output_zip_path = Path(output_zip_path)
        try:
            # Create parent directories for the zip file if necessary
            output_zip_path.parent.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                added_count = 0
                for item_path_str in file_paths:
                    item_path = Path(item_path_str)
                    if not item_path.exists():
                        logger.warning(f"Zip skipped: Path not found: {item_path_str}")
                        continue

                    if item_path.is_file():
                        # Add file, using a path relative to the current working directory
                        zipf.write(item_path, item_path.name)
                        added_count += 1
                    elif item_path.is_dir():
                        # Add directory content recursively
                        for root, dirs, files in os.walk(item_path):
                            for file in files:
                                file_path = Path(root) / file
                                # Calculate archive path to preserve directory structure
                                archive_path = file_path.relative_to(Path.cwd())
                                zipf.write(file_path, archive_path)
                                added_count += 1

            return f"Successfully created zip archive at {output_zip_path}. Added {added_count} total files/items."
        except Exception as e:
            return f"Error zipping files: {e}"

    @staticmethod
    def unzip_file(zip_path: str, output_directory: str) -> str:
        """Extracts the contents of a zip file to a specified directory."""
        zip_path = Path(zip_path)
        output_dir = Path(output_directory)

        if not zip_path.is_file():
            return f"Error: Zip file not found at {zip_path}"

        try:
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(output_dir)
            return f"Successfully extracted contents of {zip_path.name} to {output_dir.resolve()}"
        except Exception as e:
            return f"Error unzipping file: {e}"

    @staticmethod
    def recursive_file_search(directory: str, pattern: str = "*") -> str:
        """Recursively searches a directory and all its subdirectories for files matching a glob pattern."""
        search_dir = Path(directory)
        if not search_dir.is_dir():
            return f"Error: Directory not found at {directory}"

        try:
            # Using rglob for recursive glob search
            results = [str(p.resolve()) for p in search_dir.rglob(pattern) if p.is_file()]

            if not results:
                return f"No files found matching pattern '{pattern}' in '{directory}'."

            # Truncate output for LLM context, showing up to 20 results.
            output_list = results[:20]
            summary = f"Found {len(results)} files matching pattern '{pattern}'. Displaying top {len(output_list)} results:\n"
            summary += "\n".join(output_list)

            if len(results) > 20:
                 summary += f"\n... and {len(results) - 20} more results (output truncated)."
            return summary
        except Exception as e:
            return f"Error performing recursive search: {e}"


# --- OllamaToolAgent Class ---
class OllamaToolAgent:
    """Agent that uses Ollama, tools, and vector memory via the ReAct pattern."""

    def __init__(self, model_name: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL, session_id: Optional[str] = None, db_path: Optional[str] = None, init_search_query: Optional[str] = None):
        # support persistent DB path and external session id
        self.session_id = session_id or hashlib.md5(str(datetime.now()).encode()).hexdigest()
        self.db_path = db_path or "vector_chat.db"  # single persistent DB across sessions
        self.db = VectorDatabase(db_path=self.db_path, model_name=model_name)

        # Initialize ChatOllama properly
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.1
        )

        self.tools = self._get_tools()
        self.agent_executor = self._create_agent()

        # On initialization, optionally search memory and show contexts from previous sessions
        if init_search_query:
            past = self.db.similarity_search(init_search_query, session_id=self.session_id, include_other_sessions=True, limit=5, threshold=0.2)
            if past:
                preview = []
                for item in past:
                    if item['source'] == 'chat':
                        preview.append(f"[{item['session_id'][:6]}:{item['type']}] {item['content'][:120]}")
                    else:
                        try:
                            params = json.loads(item['parameters']) if item.get('parameters') else {}
                        except Exception:
                            params = item.get('parameters')
                        preview.append(f"[{item['session_id'][:6]}:TOOL {item['tool']}] {str(params)[:60]} -> {str(item['result'])[:60]}")
                console.print(Panel(Text("Loaded prior context:\n" + "\n".join(preview), style="magenta"), title="Memory Bootstrap", border_style="magenta"))

        console.print(Panel(
            Text(f"Session ID: {self.session_id}\nModel: {model_name}\nDB: {self.db_path}", style="bold green"),
            title="ToolAgent", border_style="green"
        ))

    def cleanup(self):
        """Close DB connection (do not delete persistent DB)."""
        try:
            self.db.close()
            console.print(Text(f"Closed DB connection: {self.db_path}", style="dim"))
        except Exception as e:
            logger.warning(f"Could not close database cleanly {self.db_path}: {e}")

    def _get_tools(self) -> List[Tool]:
        """Defines and returns all available tools for the agent."""

        def search_memory(query: str) -> str:
            """A wrapper for the vector database similarity search."""
            results = self.db.similarity_search(query, self.session_id, include_other_sessions=True)
            if not results:
                return "No information found in memory."
            formatted_results = []
            for item in results:
                if item['source'] == 'chat':
                    formatted_results.append(
                        f"[{item['session_id'][:6]} {item['type'].upper()}]: {item['content']}")
                elif item['source'] == 'tool_execution':
                    try:
                        params = json.loads(item['parameters']) if item.get('parameters') else {}
                    except Exception:
                        params = item.get('parameters')
                    formatted_results.append(
                        f"[{item['session_id'][:6]} TOOL {item['tool'].upper()}]: Params: {params}, Result: {item['result']}")
            return "Retrieved Memory:\n---\n" + "\n---\n".join(formatted_results)

        # Wrapper functions to handle JSON input parsing
        def create_file_wrapper(params_json: str) -> str:
            try:
                params = json.loads(params_json)
                return EnhancedTools.create_file(params['path'], params['content'])
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for parameters"
            except KeyError as e:
                return f"Error: Missing required parameter {e}"

        def read_file_wrapper(params_json: str) -> str:
            try:
                params = json.loads(params_json)
                return EnhancedTools.read_file(params['path'])
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for parameters"
            except KeyError as e:
                return f"Error: Missing required parameter {e}"

        def list_files_wrapper(params_json: str) -> str:
            try:
                params = json.loads(params_json)
                return EnhancedTools.list_files(params['directory'], params.get('pattern', '*'))
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for parameters"
            except KeyError as e:
                return f"Error: Missing required parameter {e}"

        def zip_files_wrapper(params_json: str) -> str:
            try:
                params = json.loads(params_json)
                return EnhancedTools.zip_files(params['file_paths'], params['output_zip_path'])
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for parameters"
            except KeyError as e:
                return f"Error: Missing required parameter {e}"

        def unzip_file_wrapper(params_json: str) -> str:
            try:
                params = json.loads(params_json)
                return EnhancedTools.unzip_file(params['zip_path'], params['output_directory'])
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for parameters"
            except KeyError as e:
                return f"Error: Missing required parameter {e}"

        def recursive_file_search_wrapper(params_json: str) -> str:
            try:
                params = json.loads(params_json)
                return EnhancedTools.recursive_file_search(params['directory'], params.get('pattern', '*'))
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for parameters"
            except KeyError as e:
                return f"Error: Missing required parameter {e}"

        def find_files_wrapper(params_json: str) -> str:
            try:
                params = json.loads(params_json)
                return EnhancedTools.find_files(params['query'], params.get('search_directories'))
            except json.JSONDecodeError:
                return "Error: Invalid JSON format for parameters"
            except KeyError as e:
                return f"Error: Missing required parameter {e}"

        tools = [
            # Memory Tool
            Tool(
                name="search_memory",
                func=search_memory,
                description="Search conversation and tool execution history. Input: simple string query"
            ),
            # File System Tools
            Tool(
                name="create_file",
                func=create_file_wrapper,
                description='Create/overwrite a file. Input: JSON string like {"path": "file.txt", "content": "text"}'
            ),
            Tool(
                name="read_file",
                func=read_file_wrapper,
                description='Read file content (first 1000 chars). Input: JSON string like {"path": "file.txt"}'
            ),
            Tool(
                name="list_files",
                func=list_files_wrapper,
                description='List files in directory. Input: JSON string like {"directory": ".", "pattern": "*.txt"}'
            ),
            Tool(
                name="zip_files",
                func=zip_files_wrapper,
                description='Compress files/dirs to zip. Input: JSON string like {"file_paths": ["file1.txt"], "output_zip_path": "archive.zip"}'
            ),
            Tool(
                name="unzip_file",
                func=unzip_file_wrapper,
                description='Extract zip contents. Input: JSON string like {"zip_path": "file.zip", "output_directory": "extracted"}'
            ),
            Tool(
                name="recursive_file_search",
                func=recursive_file_search_wrapper,
                description='Search directories recursively by pattern. Input: JSON string like {"directory": ".", "pattern": "*.py"}'
            ),
            Tool(
                name="find_files",
                func=find_files_wrapper,
                description='Advanced file finding with content search. Input: JSON string like {"query": "config", "search_directories": ["."]}'
            ),
            # Utility Tools
            Tool(
                name="calculate",
                func=EnhancedTools.calculate,
                description="Evaluate math expressions. Input: simple string like '5 + 3 * 2'"
            ),
            Tool(
                name="execute_command",
                func=EnhancedTools.execute_command,
                description="Execute safe shell commands. Input: simple string like 'ls -l'"
            ),
        ]
        return tools

    def _create_agent(self) -> AgentExecutor:
        """Creates the LangChain ReAct agent."""

        template = """
        You are an expert AI agent designed to execute tasks using a strict sequence of tools.
        Follow the ReAct pattern below.

        CRITICAL RULES:
        1. **MUST use tools** to gather information or complete tasks.
        2. **DO NOT explain what you're going to do** before the first Thought/Action sequence. Start immediately.
        3. **Chain tools together** if one tool's output is needed for the next.
        4. **STOP AND PROVIDE FINAL ANSWER** immediately when you see success messages like "successfully", "created", "completed".
        5. If a tool fails, try a different approach or report the error clearly.
        6. **DO NOT REPEAT** the same action if it already succeeded.

TOOLS:
{tools}

Use the following format:

Thought: I need to think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (follow the exact format specified in tool description)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

BEGIN!
Question: {input}
{agent_scratchpad}
"""

        prompt = PromptTemplate.from_template(template)

        # Create the agent
        agent = create_react_agent(self.llm, self.tools, prompt)

        # Create the executor (run the agent)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=10,
            max_execution_time=300,
            handle_parsing_errors=True
        )
        return agent_executor

    def chat(self, message: str) -> str:
        """Processes a chat message, running the agent and updating memory."""
        self.db.add_message(self.session_id, "human", message)

        console.print(Panel(Text(f"Human: {message}", style="bold blue"), title="Input", border_style="blue"))

        try:
            # Running the agent
            result = self.agent_executor.invoke({"input": message})
            response = result.get('output', 'No response generated.')

            self.db.add_message(self.session_id, "assistant", response)

            console.print(Panel(Text(f"AI: {response}", style="bold green"), title="Output", border_style="green"))
            return response

        except Exception as e:
            error_msg = f"Agent Execution Error: {e}"
            self.db.add_message(self.session_id, "assistant", error_msg)
            console.print(Panel(Text(error_msg, style="bold red"), title="Error", border_style="red"))
            return error_msg

    def execute_tool_chain(self, chain_definition: List[Dict[str, Any]], continue_on_error: bool = False) -> Dict[str, Union[bool, str]]:
        """Executes a predefined sequence of tools, passing results between steps."""
        console.print(Text("Executing tool chain...", style="yellow"))

        context = {}

        for i, step in enumerate(chain_definition):
            tool_name = step.get('tool')
            params = step.get('params', {})
            store_as = step.get('store_as')

            console.print(f"[Step {i+1}]: Calling tool '{tool_name}' with parameters: {params}")

            # Find the tool function
            tool_func = next((t.func for t in self.tools if t.name == tool_name), None)

            if not tool_func:
                error_msg = f"Tool '{tool_name}' not found."
                console.print(f"[red]Error:[/red] {error_msg}")
                if not continue_on_error:
                    return {"success": False, "message": error_msg}
                continue

            # Execute the tool
            try:
                # Handle different input types
                if tool_name in ['calculate', 'execute_command', 'search_memory']:
                    # These tools expect simple string input
                    if isinstance(params, dict) and len(params) == 1:
                        result = tool_func(list(params.values())[0])
                    else:
                        result = tool_func(params)
                else:
                    # These tools expect JSON string input
                    if isinstance(params, dict):
                        result = tool_func(json.dumps(params))
                    else:
                        result = tool_func(params)

                self.db.add_tool_execution(self.session_id, tool_name, params if isinstance(params, dict) else {"input": params}, result)
                console.print(f"[Step {i+1} Result]: {result[:100]}...")

                # Store the result in context if requested
                if store_as:
                    context[store_as] = result
                    console.print(f"[Context]: Stored result as '{store_as}'")

            except Exception as e:
                error_msg = f"Error executing tool '{tool_name}': {e}"
                console.print(f"[red]Error:[/red] {error_msg}")
                if not continue_on_error:
                    return {"success": False, "message": error_msg}

        console.print(Text("Tool chain finished.", style="yellow"))
        return {"success": True, "context": context}


def main():
    """Main function to run the chat interface."""

    agent = None
    try:
        console.print(Panel(
            Text(f"Starting Agent with Model: {DEFAULT_MODEL} at {DEFAULT_BASE_URL}", style="bold cyan"),
            title="Configuration", border_style="cyan"
        ))

        # Allow bootstrap search by environment variable INIT_SEARCH or CLI arg
        init_query = os.environ.get("INIT_SEARCH")
        if len(sys.argv) > 1 and not init_query:
            init_query = " ".join(sys.argv[1:]).strip() or None

        agent = OllamaToolAgent(model_name=DEFAULT_MODEL, base_url=DEFAULT_BASE_URL, init_search_query=init_query)

        console.print(Panel(
            Text("Enter your query or 'quit'/'exit' to stop.", style="bold yellow"),
            title="Chat Interface", border_style="yellow"
        ))

        while True:
            try:
                user_input = input("\n> ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    console.print(Text("Exiting...", style="yellow"))
                    break
                if not user_input:
                    continue
                agent.chat(user_input)
            except KeyboardInterrupt:
                console.print(Text("\nInterrupted. Exiting...", style="yellow"))
                break
            except EOFError:
                console.print(Text("\nEOF detected. Exiting...", style="yellow"))
                break

    finally:
        if agent:
            agent.cleanup()


if __name__ == "__main__":
    main()