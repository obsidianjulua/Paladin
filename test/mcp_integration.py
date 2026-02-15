#!/usr/bin/env python3
"""
MCP Integration for Paladin
Loads tools from MCP servers and converts them to LangChain StructuredTools
"""

import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from langchain_core.tools import StructuredTool
from pydantic import create_model, Field, BaseModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPToolLoader:
    """Loads tools from MCP servers and converts them to LangChain format."""
    
    def __init__(self):
        self.sessions: List[ClientSession] = []
        self.tools: List[StructuredTool] = []
        
    async def connect_server(self, server_config: Dict[str, Any]) -> Optional[ClientSession]:
        """
        Connect to an MCP server.
        
        Args:
            server_config: Dict with 'command', 'args', and 'env' keys
        """
        try:
            server_params = StdioServerParameters(
                command=server_config['command'],
                args=server_config.get('args', []),
                env=server_config.get('env', {})
            )
            
            read, write = await stdio_client(server_params).__aenter__()
            session = ClientSession(read, write)
            await session.initialize()
            
            self.sessions.append(session)
            logger.info(f"Connected to MCP server: {server_config['command']}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            return None
    
    def _convert_mcp_param_to_pydantic_type(self, param_schema: Dict[str, Any]) -> tuple:
        """Convert MCP parameter schema to Pydantic field type."""
        param_type = param_schema.get('type', 'string')
        
        type_map = {
            'string': str,
            'number': float,
            'integer': int,
            'boolean': bool,
            'array': list,
            'object': dict
        }
        
        py_type = type_map.get(param_type, str)
        
        # Check if required
        is_required = param_schema.get('required', False)
        description = param_schema.get('description', '')
        
        field_kwargs = {'description': description}
        
        if not is_required:
            default = param_schema.get('default', None)
            field_kwargs['default'] = default
            return (Optional[py_type], Field(**field_kwargs))
        else:
            return (py_type, Field(**field_kwargs))
    
    def _create_pydantic_schema(self, tool_name: str, input_schema: Dict[str, Any]) -> type[BaseModel]:
        """Create Pydantic model from MCP tool input schema."""
        properties = input_schema.get('properties', {})
        required = input_schema.get('required', [])
        
        if not properties:
            return create_model(f"{tool_name}Input")
        
        fields = {}
        for param_name, param_schema in properties.items():
            param_schema['required'] = param_name in required
            fields[param_name] = self._convert_mcp_param_to_pydantic_type(param_schema)
        
        return create_model(f"{tool_name}Input", **fields)
    
    async def load_tools_from_session(self, session: ClientSession) -> List[StructuredTool]:
        """Load all tools from an MCP session and convert to LangChain tools."""
        tools = []
        
        try:
            # List available tools
            tool_list = await session.list_tools()
            
            for mcp_tool in tool_list.tools:
                tool_name = mcp_tool.name
                description = mcp_tool.description or f"MCP tool: {tool_name}"
                input_schema = mcp_tool.inputSchema
                
                # Create Pydantic schema
                args_schema = self._create_pydantic_schema(tool_name, input_schema)
                
                # Create wrapper function
                def create_tool_func(session_ref, tool_name_ref):
                    """Closure to capture session and tool_name."""
                    async def tool_func(**kwargs) -> str:
                        """Execute MCP tool."""
                        try:
                            result = await session_ref.call_tool(tool_name_ref, arguments=kwargs)
                            
                            # Format result
                            if hasattr(result, 'content'):
                                # Extract text content
                                content_parts = []
                                for content in result.content:
                                    if hasattr(content, 'text'):
                                        content_parts.append(content.text)
                                return '\n'.join(content_parts)
                            
                            return str(result)
                            
                        except Exception as e:
                            return f"Error executing MCP tool {tool_name_ref}: {e}"
                    
                    # Make sync wrapper
                    def sync_wrapper(**kwargs) -> str:
                        return asyncio.run(tool_func(**kwargs))
                    
                    return sync_wrapper
                
                tool_func_sync = create_tool_func(session, tool_name)
                
                # Create StructuredTool
                langchain_tool = StructuredTool.from_function(
                    func=tool_func_sync,
                    name=tool_name,
                    description=description,
                    args_schema=args_schema
                )
                
                tools.append(langchain_tool)
                logger.info(f"Loaded MCP tool: {tool_name}")
        
        except Exception as e:
            logger.error(f"Error loading tools from MCP session: {e}")
        
        return tools
    
    async def load_all_servers(self, config_path: Optional[str] = None) -> List[StructuredTool]:
        """
        Load tools from all MCP servers defined in config.
        
        Args:
            config_path: Path to Claude Desktop config or custom MCP config
        """
        if config_path is None:
            # Try to find Claude Desktop config
            config_path = Path.home() / ".config/Claude/claude_desktop_config.json"
            if not config_path.exists():
                config_path = Path.home() / "Library/Application Support/Claude/claude_desktop_config.json"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"MCP config not found at {config_path}")
            return []
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            mcp_servers = config.get('mcpServers', {})
            
            all_tools = []
            for server_name, server_config in mcp_servers.items():
                logger.info(f"Connecting to MCP server: {server_name}")
                session = await self.connect_server(server_config)
                
                if session:
                    tools = await self.load_tools_from_session(session)
                    all_tools.extend(tools)
                    logger.info(f"Loaded {len(tools)} tools from {server_name}")
            
            self.tools = all_tools
            return all_tools
            
        except Exception as e:
            logger.error(f"Error loading MCP servers from config: {e}")
            return []
    
    async def cleanup(self):
        """Close all MCP sessions."""
        for session in self.sessions:
            try:
                await session.close()
            except:
                pass


def load_mcp_tools(config_path: Optional[str] = None) -> List[StructuredTool]:
    """
    Synchronous wrapper to load MCP tools.
    
    Usage:
        mcp_tools = load_mcp_tools()
    """
    loader = MCPToolLoader()
    tools = asyncio.run(loader.load_all_servers(config_path))
    return tools


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load tools
    tools = load_mcp_tools()
    
    print(f"\nLoaded {len(tools)} MCP tools:")
    for tool in tools:
        print(f"  • {tool.name}: {tool.description}")
