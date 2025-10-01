#!/usr/bin/env python3
"""
Chat Interface Module
Handles interactive chat sessions and user input
"""

import os
import sys
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from .agent import OllamaToolAgent, DEFAULT_MODEL, DEFAULT_BASE_URL

console = Console()


class ChatInterface:
    """Interactive chat interface for Paladin."""

    def __init__(self, model_name: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL,
                 init_search_query: str = None):
        self.agent = None
        self.model_name = model_name
        self.base_url = base_url
        self.init_search_query = init_search_query

    def start(self):
        """Start the interactive chat session."""
        try:
            console.print(Panel(
                Text(f"Starting Paladin with Model: {self.model_name} at {self.base_url}", style="bold cyan"),
                title="Configuration",
                border_style="cyan"
            ))

            self.agent = OllamaToolAgent(
                model_name=self.model_name,
                base_url=self.base_url,
                init_search_query=self.init_search_query
            )

            console.print(Panel(
                Text("Enter your query or 'quit'/'exit' to stop.\n"
                     "Special commands:\n"
                     "  /tools - List available tools\n"
                     "  /templates - List available templates\n"
                     "  /help - Show help", style="bold yellow"),
                title="Chat Interface",
                border_style="yellow"
            ))

            self._chat_loop()

        finally:
            if self.agent:
                self.agent.cleanup()

    def _chat_loop(self):
        """Main chat loop."""
        while True:
            try:
                user_input = input("\n> ").strip()

                if user_input.lower() in ['quit', 'exit', 'q']:
                    console.print(Text("Exiting...", style="yellow"))
                    break

                if not user_input:
                    continue

                # Handle special commands
                if user_input.startswith('/'):
                    self._handle_command(user_input)
                    continue

                # Normal chat
                self.agent.chat(user_input)

            except KeyboardInterrupt:
                console.print(Text("\nInterrupted. Exiting...", style="yellow"))
                break
            except EOFError:
                console.print(Text("\nEOF detected. Exiting...", style="yellow"))
                break

    def _handle_command(self, command: str):
        """Handle special slash commands."""
        cmd = command.lower().strip()

        if cmd == '/tools':
            tools = self.agent.list_available_tools()
            console.print(Panel(
                Text(f"Available Tools ({len(tools)}):\n" + "\n".join(f"  • {t}" for t in tools),
                     style="cyan"),
                title="Tools",
                border_style="cyan"
            ))

        elif cmd == '/templates':
            templates = self.agent.list_available_templates()
            if templates:
                template_list = "\n".join(f"  • {t['name']}: {t['description']}" for t in templates)
                console.print(Panel(
                    Text(f"Available Templates ({len(templates)}):\n" + template_list, style="magenta"),
                    title="Templates",
                    border_style="magenta"
                ))
            else:
                console.print(Text("No templates available yet.", style="dim"))

        elif cmd == '/help':
            help_text = """
Paladin AI - Tool-Calling Agent

Commands:
  /tools      - List all available tools
  /templates  - List all available templates
  /help       - Show this help message
  quit, exit  - Exit the program

Usage:
  Just type your request and press Enter. The AI will use tools from the
  database to complete your task. Tools are loaded dynamically from:
  /home/grim/Desktop/op/TOOL_DB/tools.db
"""
            console.print(Panel(Text(help_text.strip(), style="yellow"), title="Help", border_style="yellow"))

        else:
            console.print(Text(f"Unknown command: {command}", style="red"))


def run_chat_interface():
    """Entry point for running the chat interface."""
    # Allow bootstrap search by environment variable or CLI arg
    init_query = os.environ.get("INIT_SEARCH")
    if len(sys.argv) > 1 and not init_query:
        init_query = " ".join(sys.argv[1:]).strip() or None

    interface = ChatInterface(
        model_name=DEFAULT_MODEL,
        base_url=DEFAULT_BASE_URL,
        init_search_query=init_query
    )
    interface.start()


if __name__ == "__main__":
    run_chat_interface()
