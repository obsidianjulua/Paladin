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
                 init_search_query: str = None, save_history: bool = True):
        self.agent = None
        self.model_name = model_name
        self.base_url = base_url
        self.init_search_query = init_search_query
        self.save_history = save_history

    def start(self):
        """Start the interactive chat session."""
        try:
            console.print(Panel(
                Text(f"Model: {self.model_name} at {self.base_url}\nHistory: {self.save_history}", style="bold cyan"),
                title="Configuration",
                border_style="cyan"
            ))

            self.agent = OllamaToolAgent(
                model_name=self.model_name,
                base_url=self.base_url,
                init_search_query=self.init_search_query,
                save_history=self.save_history
            )

            console.print(Panel(
                Text("Enter your query or 'quit'/'exit' to stop.\n"
                     "Special commands:\n"
                     "  /tools\n"
                     "  /help", style="bold yellow"),
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

                if user_input.lower() in ['quit', 'exit', 'bye']:
                    console.print(Text("Exiting...", style="yellow"))
                    break

                if not user_input:
                    continue

                # Handle special commands
                if user_input.startswith(':'):
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

        elif cmd == '/help':
            help_text = """
            Ask the AI!!
"""
            console.print(Panel(Text(help_text.strip(), style="yellow"), title="Help", border_style="yellow"))

        else:
            console.print(Text(f"Unknown command: {command}", style="red"))


def run_chat_interface():
    """Entry point for running the chat interface."""
    # Check for flags
    save_history = True
    if "--no-save" in sys.argv:
        save_history = False
        sys.argv.remove("--no-save")

    # Allow bootstrap search by environment variable or CLI arg
    init_query = os.environ.get("INIT_SEARCH")
    if len(sys.argv) > 1 and not init_query:
        init_query = " ".join(sys.argv[1:]).strip() or None

    interface = ChatInterface(
        model_name=DEFAULT_MODEL,
        base_url=DEFAULT_BASE_URL,
        init_search_query=init_query,
        save_history=save_history
    )
    interface.start()


if __name__ == "__main__":
    run_chat_interface()
