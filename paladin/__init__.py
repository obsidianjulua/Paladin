"""
Paladin AI - Modular Tool-Calling System
"""

__version__ = "2.0.0"

from .vector_db import VectorDatabase
from .tool_registry import ToolRegistry
from .agent import OllamaToolAgent

__all__ = ['VectorDatabase', 'ToolRegistry', 'OllamaToolAgent']
