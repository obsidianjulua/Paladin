#!/usr/bin/env python3
"""
Ollama Tool Agent - Main agent implementation
"""

import hashlib
import json
import logging
import os
import signal
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional, Union

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .vector_db import VectorDatabase
from .tool_registry import ToolRegistry
from .qwen_tool_formatter import QwenToolFormatter

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_MODEL = os.getenv("PALADIN_MODEL", "qwen3-coder:latest")
DEFAULT_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")


class RichCallbackHandler(BaseCallbackHandler):
    """Custom callback handler to render agent actions nicely with Rich."""
    
    def __init__(self, console: Console):
        self.console = console
        self.status = None
        self.current_tool = None
        self.step_color = "cyan"
    
    def on_chain_start(self, serialized, inputs, **kwargs):
        """Start the spinner when the chain begins."""
        # Only start if it's the main agent chain (heuristic)
        if not self.status:
            self.status = self.console.status("[bold green]Paladin is thinking...", spinner="dots")
            self.status.start()

    def on_agent_action(self, action: AgentAction, **kwargs):
        """Update spinner when a tool is selected."""
        self.current_tool = action.tool
        if self.status:
            self.status.update(f"[bold {self.step_color}]Executing tool: {action.tool}...")

    def on_tool_end(self, output: str, **kwargs):
        """Update spinner when tool finishes."""
        if self.status:
            self.status.update(f"[bold {self.step_color}]Finished {self.current_tool}. Processing results...")
    
    def on_chain_end(self, outputs, **kwargs):
        """Stop spinner when done."""
        if self.status:
            self.status.stop()
            self.status = None

    def on_agent_finish(self, finish: AgentFinish, **kwargs):
        if self.status:
            self.status.stop()
            self.status = None


class OllamaToolAgent:
    """Agent that uses Ollama, tools from database, and vector memory via Native Tool Calling."""

    def __init__(self, model_name: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL,
                 session_id: Optional[str] = None, vector_db_path: Optional[str] = None,
                 tool_db_path: Optional[str] = None, init_search_query: Optional[str] = None,
                 save_history: bool = True):

        self.session_id = session_id or hashlib.md5(str(datetime.now()).encode()).hexdigest()
        self.save_history = save_history

        # Load DB paths from env if not provided
        if not vector_db_path:
            vector_db_path = os.getenv("PALADIN_VECTOR_DB")
        
        if not tool_db_path:
            tool_db_path = os.getenv("PALADIN_TOOL_DB")

        # Initialize Vector Database
        self.db = VectorDatabase(db_path=vector_db_path, model_name=model_name)

        # Initialize Tool Registry
        self.tool_registry = ToolRegistry(db_path=tool_db_path, vector_db=self.db)

        # Store session in vector DB for memory tool
        self.db._current_session = self.session_id

        # Vault logging path
        self.vault_path = os.getenv("PALADIN_VAULT_PATH")

        # Initialize ChatOllama
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.4, # 0.2, 2048
            num_predict=8192, # Larger context for code generation
        )

        # Load tools from database
        self.tools = self.tool_registry.load_tools()
        self.agent_executor = self._create_agent()

        # Bootstrap from memory if requested
        if init_search_query:
            self._bootstrap_memory(init_search_query)

        console.print(Panel(
            Text(
                f"Session ID: {self.session_id}\n"
                f"Model: {model_name}\n"
                f"Vector DB: {self.db.db_path}\n"
                f"Tool DB: {self.tool_registry.db_path}\n"
                f"Loaded Tools: {len(self.tools)}",
                style="bold green"
            ),
            title="Paladin Agent",
            border_style="green"
        ))

    def _bootstrap_memory(self, query: str):
        past = self.db.similarity_search(
            query,
            session_id=self.session_id,
            include_other_sessions=True,
            limit=5,
            threshold=0.2
        )
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
                    preview.append(
                        f"[{item['session_id'][:6]}:TOOL {item['tool']}] "
                        f"{str(params)[:60]} -> {str(item['result'])[:60]}"
                    )
            console.print(Panel(
                Text("Loaded prior context:\n" + "\n".join(preview), style="magenta"),
                title="Memory Bootstrap",
                border_style="magenta"
            ))

    def cleanup(self):
        """Close database connections."""
        try:
            self.db.close()
            console.print(Text(f"Closed DB connection: {self.db.db_path}", style="dim"))
        except Exception as e:
            logger.warning(f"Could not close database cleanly: {e}")

    def _log_to_vault(self, query: str, response: str):
        """Appends the chat interaction to the configured Obsidian vault file."""
        if not self.vault_path:
            return
            
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            os.makedirs(os.path.dirname(os.path.abspath(self.vault_path)), exist_ok=True)
            
            with open(self.vault_path, "a", encoding="utf-8") as f:
                f.write(f"\n## {timestamp}\n")
                f.write(f"**User:** {query}\n\n")
                f.write(f"**AI:**\n{response}\n")
                f.write("\n---\n")
        except Exception as e:
            console.print(f"[yellow]Warning: Failed to log to vault at {self.vault_path}: {e}[/yellow]")

    def _get_cwd_context(self) -> str:
        """Get current working directory context for the prompt."""
        try:
            cwd = os.getcwd()
            files = os.listdir(cwd)
            visible_files = [f for f in files if not f.startswith('.')]
            if len(visible_files) > 20:
                file_list = ", ".join(visible_files[:20]) + f", ... (+{len(visible_files)-20} more)"
            else:
                file_list = ", ".join(visible_files) or "(empty)"
            
            return f"Current Path: {cwd}\nFiles here: {file_list}"
        except Exception as e:
            return f"Error getting path context: {e}"

    def _get_project_instructions(self) -> str:
        """Load project-specific instructions from a local file."""
        try:
            cwd = os.getcwd()
            instruct_path = os.path.join(cwd, "PALADIN_INSTRUCTIONS.md")
            if os.path.exists(instruct_path):
                with open(instruct_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                if content:
                    return f"\nPROJECT SPECIFIC INSTRUCTIONS:\n{content}\n"
            return ""
        except Exception:
            return ""

    def _create_agent(self) -> AgentExecutor:
        """Creates the LangChain Native Tool Calling agent."""
        
        # 1. Define the Chat Prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Paladin, an expert AI software engineer.
Your goal is to complete tasks efficiently and safely using your available tools.

## Core Directives
1. **Verification**: Always verify file locations and contents before editing. Do not guess.
2. **Safety**: Analyze potential risks before executing system commands.
3. **Memory**: Use the provided `Relevant Past Memories` to inform your decisions and maintain continuity.
4. **Tool Usage**: You may call multiple tools in sequence to accomplish a goal.

## Context
### Working Directory
{cwd_context}

### Relevant Past Memories
{memory_context}

{project_instructions}
"""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # 2. Bind tools to the LLM (This enables Native Tool Calling)
        # Qwen supports this natively via Ollama, but we tune it for reliability
        
        # Generate Qwen-specific JSON for debugging/logging
        qwen_tools = QwenToolFormatter.convert_all_tools(self.tools)
        
        # Bind with explicit tool_choice + low temp
        llm_with_tools = self.llm.bind_tools(
            self.tools, 
            tool_choice="auto" 
        )

        # 3. Create the Agent
        # Use ReAct for stability if too many tools, otherwise Tool Calling Agent
        if len(self.tools) > 15: # Raised threshold slightly to prefer tool calling
             # Fallback to ReAct if needed (requires specific prompt structure)
             # For now, we stick to tool calling agent as it is more robust with modern models
             agent = create_tool_calling_agent(llm_with_tools, self.tools, prompt)
        else:
             agent = create_tool_calling_agent(llm_with_tools, self.tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=False,
            max_iterations=15, 
            max_execution_time=900,
            early_stopping_method="generate",
            return_intermediate_steps=False,
            callbacks=[RichCallbackHandler(console)]
        )
        return agent_executor

    def _get_memory_context(self, query: str) -> str:
        """Formatted memory context for Qwen."""
        memories = self.db.similarity_search(query, limit=6, threshold=0.22)
        if not memories:
            return "(No strong matches in memory)"
        
        # Structured, concise for Qwen
        lines = []
        for m in memories:
            content = m.get('content', '')
            if m['source'] == 'chat':
                lines.append(f"📝 RECENT: {content[:180]}")
            elif m['source'] == 'tool':
                # Note: vector_db now returns 'tool' instead of 'tool_execution'
                lines.append(f"🛠️ TOOL [{str(m.get('session_id'))[:6]}]: {content[:120]}")
            elif m['source'] == 'doc':
                lines.append(f"📄 DOC: {content[:120]}")
            elif m['source'] == 'fact':
                lines.append(f"💾 FACT: {content}")
        return "\n".join(lines[:6])

    def chat(self, message: str) -> str:
        """Processes a chat message, running the agent and updating memory."""

        @contextmanager
        def timeout_context(seconds):
            """Context manager for timeout using signals (Unix only)."""
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Agent execution exceeded {seconds} seconds")

            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(seconds)
            try:
                yield
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

        self.db.add_message(self.session_id, "human", message) if self.save_history else None

        console.print(Panel(
            Text(f"Human: {message}", style="bold blue"),
            title="Input",
            border_style="blue"
        ))

        # DEBUG: Print tool format
        try:
            formatted_tools = QwenToolFormatter.format_for_ollama(self.tools)
            logger.debug(f"Tool call format preview: {formatted_tools[:200]}...")
        except Exception:
            pass

        # Handle Command Prefixes
        if message.startswith("//"):
            # Chat Mode: Direct LLM Call (Bypass Agent)
            chat_input = message[2:].strip()
            try:
                response_obj = self.llm.invoke(chat_input)
                response = response_obj.content
            except Exception as e:
                response = f"Chat Error: {e}"
            
            self.db.add_message(self.session_id, "assistant", response) if self.save_history else None
            console.print(Panel(
                Text(f"AI: {response}", style="bold green"),
                title="Output",
                border_style="green"
            ))
            self._log_to_vault(message, response)
            return response

        elif message.startswith("\\\\"):
            # Force Tool Mode
            llm_input = message[2:].strip() + "\n\nCONSTRAINT: You MUST use a tool to handle this request."
        else:
            # Standard Mode
            llm_input = message

        try:
            with timeout_context(330):
                cwd_context = self._get_cwd_context()
                project_instructions = self._get_project_instructions()
                
                # Fetch recent chat history (Short-term memory)
                raw_history = self.db.get_recent_history(self.session_id, limit=10)
                chat_history = []
                for msg in raw_history:
                    if msg['type'] == 'human':
                        chat_history.append(HumanMessage(content=msg['content']))
                    elif msg['type'] == 'assistant':
                        chat_history.append(AIMessage(content=msg['content']))

                # Fetch relevant memories (Long-term memory / RAG)
                memory_context = self._get_memory_context(llm_input)

                result = self.agent_executor.invoke({
                    "input": llm_input,
                    "cwd_context": cwd_context,
                    "project_instructions": project_instructions,
                    "memory_context": memory_context,
                    "chat_history": chat_history
                })
                response = result.get('output', 'No response generated.')

            self.db.add_message(self.session_id, "assistant", response) if self.save_history else None

            console.print(Panel(
                Text(f"AI: {response}", style="bold green"),
                title="Output",
                border_style="green"
            ))
            self._log_to_vault(message, response)
            return response

        except TimeoutError as e:
            error_msg = f"Agent Timeout: {e}. The agent took too long to respond."
            self.db.add_message(self.session_id, "assistant", error_msg) if self.save_history else None
            console.print(Panel(
                Text(error_msg, style="bold red"),
                title="Timeout",
                border_style="red"
            ))
            logger.error(f"Agent timeout on message: {message}")
            return error_msg

        except Exception as e:
            error_msg = f"Agent Execution Error: {e}"
            self.db.add_message(self.session_id, "assistant", error_msg) if self.save_history else None
            console.print(Panel(
                Text(error_msg, style="bold red"),
                title="Error",
                border_style="red"
            ))
            logger.exception(f"Agent error on message: {message}")
            return error_msg

    def execute_tool_chain(self, chain_definition: List[Dict[str, Any]],
                          continue_on_error: bool = False) -> Dict[str, Union[bool, str]]:
        """Executes a predefined sequence of tools."""
        console.print(Text("Executing tool chain...", style="yellow"))
        context = {}

        for i, step in enumerate(chain_definition):
            tool_name = step.get('tool')
            params = step.get('params', {})
            store_as = step.get('store_as')

            console.print(f"[Step {i+1}]: Calling tool '{tool_name}' with parameters: {params}")

            tool_func = next((t.func for t in self.tools if t.name == tool_name), None)

            if not tool_func:
                error_msg = f"Tool '{tool_name}' not found."
                console.print(f"[red]Error:[/red] {error_msg}")
                if not continue_on_error:
                    return {"success": False, "message": error_msg}
                continue

            try:
                if isinstance(params, dict):
                    result = tool_func(json.dumps(params))
                else:
                    result = tool_func(params)

                if self.save_history:
                    self.db.add_tool_execution(
                        self.session_id,
                        tool_name,
                        params if isinstance(params, dict) else {"input": params},
                        result
                    )
                console.print(f"[Step {i+1} Result]: {result[:100]}...")

                if store_as:
                    context[store_as] = result

            except Exception as e:
                error_msg = f"Error executing tool '{tool_name}': {e}"
                console.print(f"[red]Error:[/red] {error_msg}")
                if not continue_on_error:
                    return {"success": False, "message": error_msg}

        console.print(Text("Tool chain finished.", style="yellow"))
        return {"success": True, "context": context}

    def list_available_tools(self) -> List[str]:
        """List all available tool names."""
        return [tool.name for tool in self.tools]
