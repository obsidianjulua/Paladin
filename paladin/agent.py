#!/usr/bin/env python3
"""
Ollama Tool Agent - Main agent implementation
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .vector_db import VectorDatabase
from .tool_registry import ToolRegistry

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_MODEL = "codellama:7b-instruct-q4_K_M"
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaToolAgent:
    """Agent that uses Ollama, tools from database, and vector memory via ReAct pattern."""

    def __init__(self, model_name: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL,
                 session_id: Optional[str] = None, vector_db_path: Optional[str] = None,
                 tool_db_path: Optional[str] = None, init_search_query: Optional[str] = None):

        self.session_id = session_id or hashlib.md5(str(datetime.now()).encode()).hexdigest()

        # Initialize Vector Database
        self.db = VectorDatabase(db_path=vector_db_path, model_name=model_name)

        # Initialize Tool Registry
        self.tool_registry = ToolRegistry(db_path=tool_db_path, vector_db=self.db)

        # Store session in vector DB for memory tool
        self.db._current_session = self.session_id

        # Initialize ChatOllama
        self.llm = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=0.0,  # Zero for more deterministic output
            num_predict=512,  # Limit response length to prevent rambling
            stop=["Observation:", "\nObservation"]  # Stop at observation markers
        )

        # Load tools from database
        self.tools = self.tool_registry.load_tools()
        self.agent_executor = self._create_agent()

        # Bootstrap from memory if requested
        if init_search_query:
            past = self.db.similarity_search(
                init_search_query,
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

    def cleanup(self):
        """Close database connections."""
        try:
            self.db.close()
            console.print(Text(f"Closed DB connection: {self.db.db_path}", style="dim"))
        except Exception as e:
            logger.warning(f"Could not close database cleanly: {e}")

    def _create_agent(self) -> AgentExecutor:
        """Creates the LangChain ReAct agent."""
        template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format EXACTLY:

Thought: I should use a tool to help with this task
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

IMPORTANT:
- Always use the EXACT format above
- Put only the tool name after "Action:"
- Put only the input after "Action Input:"
- Do NOT write "Observation:" yourself - it will be provided to you
- Stop after writing "Action Input:" and wait for the Observation

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

        prompt = PromptTemplate.from_template(template)
        agent = create_react_agent(self.llm, self.tools, prompt)

        # Custom parsing error handler
        def handle_parse_error(error) -> str:
            return f"Could not parse LLM output. Please respond using the exact format:\nThought: [your thought]\nAction: [tool name]\nAction Input: [input]\n\nError details: {str(error)}"

        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            max_iterations=5,  # Reduced from 10 to fail faster
            max_execution_time=60,  # Reduced from 300 to fail faster
            handle_parsing_errors=handle_parse_error,  # Custom error handler
            early_stopping_method="generate",  # Return what we have so far on failure
            return_intermediate_steps=False  # Don't clutter output
        )
        return agent_executor

    def chat(self, message: str) -> str:
        """Processes a chat message, running the agent and updating memory."""
        self.db.add_message(self.session_id, "human", message)

        console.print(Panel(
            Text(f"Human: {message}", style="bold blue"),
            title="Input",
            border_style="blue"
        ))

        try:
            result = self.agent_executor.invoke({"input": message})
            response = result.get('output', 'No response generated.')

            self.db.add_message(self.session_id, "assistant", response)

            console.print(Panel(
                Text(f"AI: {response}", style="bold green"),
                title="Output",
                border_style="green"
            ))
            return response

        except Exception as e:
            error_msg = f"Agent Execution Error: {e}"
            self.db.add_message(self.session_id, "assistant", error_msg)
            console.print(Panel(
                Text(error_msg, style="bold red"),
                title="Error",
                border_style="red"
            ))
            return error_msg

    def execute_tool_chain(self, chain_definition: List[Dict[str, Any]],
                          continue_on_error: bool = False) -> Dict[str, Union[bool, str]]:
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

            try:
                # Execute the tool
                if isinstance(params, dict):
                    result = tool_func(json.dumps(params))
                else:
                    result = tool_func(params)

                self.db.add_tool_execution(
                    self.session_id,
                    tool_name,
                    params if isinstance(params, dict) else {"input": params},
                    result
                )
                console.print(f"[Step {i+1} Result]: {result[:100]}...")

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

    def execute_template(self, template_name: str, user_params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool template (workflow) from the database."""
        return self.tool_registry.execute_template(
            template_name,
            user_params,
            self.agent_executor,
            self.session_id
        )

    def list_available_tools(self) -> List[str]:
        """List all available tool names."""
        return [tool.name for tool in self.tools]

    def list_available_templates(self) -> List[Dict[str, str]]:
        """List all available templates."""
        templates = self.tool_registry.load_templates()
        return [{'name': t['name'], 'description': t['description']} for t in templates]
