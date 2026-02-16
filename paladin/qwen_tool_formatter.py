#!/usr/bin/env python3
"""
Qwen Tool Format Converter
Converts LangChain StructuredTools to Qwen's native function calling format
"""

import json
from typing import List, Dict, Any
from langchain_core.tools import StructuredTool
from pydantic import BaseModel


class QwenToolFormatter:
    """Converts tools to Qwen's preferred format."""
    
    @staticmethod
    def pydantic_to_json_schema(model: BaseModel) -> Dict[str, Any]:
        """Convert Pydantic model to JSON schema for Qwen."""
        schema = model.model_json_schema()
        
        # Clean up schema for Qwen
        if '$defs' in schema:
            del schema['$defs']
        if 'title' in schema:
            del schema['title']
        
        return schema
    
    @staticmethod
    def convert_tool(tool: StructuredTool) -> Dict[str, Any]:
        """
        Convert LangChain StructuredTool to Qwen function format.
        
        Returns format compatible with Ollama's tool calling:
        {
            "type": "function",
            "function": {
                "name": "tool_name",
                "description": "Tool description",
                "parameters": {...json schema...}
            }
        }
        """
        # Get the Pydantic model for arguments
        args_schema = tool.args_schema
        
        if args_schema:
            parameters = QwenToolFormatter.pydantic_to_json_schema(args_schema)
        else:
            parameters = {
                "type": "object",
                "properties": {},
                "required": []
            }
        
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": parameters
            }
        }
    
    @staticmethod
    def convert_all_tools(tools: List[StructuredTool]) -> List[Dict[str, Any]]:
        """Convert a list of LangChain tools to Qwen format."""
        return [QwenToolFormatter.convert_tool(tool) for tool in tools]
    
    @staticmethod
    def format_for_ollama(tools: List[StructuredTool]) -> str:
        """
        Format tools as JSON for Ollama API.
        
        Usage with Ollama:
            import ollama
            tools_json = QwenToolFormatter.format_for_ollama(tools)
            response = ollama.chat(
                model='qwen2.5:7b',
                messages=[...],
                tools=json.loads(tools_json)
            )
        """
        qwen_tools = QwenToolFormatter.convert_all_tools(tools)
        return json.dumps(qwen_tools, indent=2)


class QwenResponseParser:
    """Parse Qwen's tool calling responses."""
    
    @staticmethod
    def extract_tool_calls(response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract tool calls from Qwen/Ollama response.
        
        Returns:
            List of {name: str, arguments: dict}
        """
        message = response.get('message', {})
        tool_calls = message.get('tool_calls', [])
        
        calls = []
        for call in tool_calls:
            function = call.get('function', {})
            calls.append({
                'name': function.get('name'),
                'arguments': function.get('arguments', {})
            })
        
        return calls
    
    @staticmethod
    def should_call_tools(response: Dict[str, Any]) -> bool:
        """Check if Qwen wants to call tools."""
        message = response.get('message', {})
        return len(message.get('tool_calls', [])) > 0


# Example usage
if __name__ == "__main__":
    from langchain_core.tools import StructuredTool
    from pydantic import BaseModel, Field
    
    # Example tool
    class SearchInput(BaseModel):
        query: str = Field(description="Search query")
        max_results: int = Field(default=10, description="Maximum results")
    
    def search_func(query: str, max_results: int = 10) -> str:
        return f"Searching for '{query}' (max {max_results} results)..."
    
    tool = StructuredTool.from_function(
        func=search_func,
        name="search",
        description="Search the database",
        args_schema=SearchInput
    )
    
    # Convert to Qwen format
    formatter = QwenToolFormatter()
    qwen_tool = formatter.convert_tool(tool)
    
    print("Qwen Tool Format:")
    print(json.dumps(qwen_tool, indent=2))
    
    # Format for Ollama
    print("\nOllama Tools JSON:")
    print(formatter.format_for_ollama([tool]))
