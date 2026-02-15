#!/usr/bin/env python3
"""
MCP Config Converter
Converts between JSON and YAML MCP configuration formats
"""

import json
import yaml
import sys
from pathlib import Path
from typing import Dict, Any


def json_to_yaml(json_path: str, yaml_path: str = None):
    """
    Convert JSON MCP config to YAML format.
    
    Args:
        json_path: Path to JSON config file
        yaml_path: Output YAML path (optional, defaults to .yaml version)
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return False
    
    # Load JSON
    with open(json_path, 'r') as f:
        config = json.load(f)
    
    # Convert mcpServers from dict to list format
    if 'mcpServers' in config and isinstance(config['mcpServers'], dict):
        servers = []
        for name, server_config in config['mcpServers'].items():
            server_config['name'] = name
            servers.append(server_config)
        
        # Create YAML format
        yaml_config = {
            'name': 'MCP Servers',
            'version': '1.0.0',
            'schema': 'v1',
            'mcpServers': servers
        }
    else:
        yaml_config = config
    
    # Determine output path
    if yaml_path is None:
        yaml_path = json_path.with_suffix('.yaml')
    else:
        yaml_path = Path(yaml_path)
    
    # Save as YAML
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Converted {json_path} → {yaml_path}")
    print(f"  Servers: {len(yaml_config['mcpServers'])}")
    return True


def yaml_to_json(yaml_path: str, json_path: str = None):
    """
    Convert YAML MCP config to JSON format.
    
    Args:
        yaml_path: Path to YAML config file
        json_path: Output JSON path (optional, defaults to .json version)
    """
    yaml_path = Path(yaml_path)
    
    if not yaml_path.exists():
        print(f"Error: YAML file not found: {yaml_path}")
        return False
    
    # Load YAML
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Convert mcpServers from list to dict format
    if 'mcpServers' in config and isinstance(config['mcpServers'], list):
        servers_dict = {}
        for server in config['mcpServers']:
            name = server.pop('name', 'unnamed')
            servers_dict[name] = server
        
        json_config = {
            'mcpServers': servers_dict
        }
    else:
        json_config = config
    
    # Determine output path
    if json_path is None:
        json_path = yaml_path.with_suffix('.json')
    else:
        json_path = Path(json_path)
    
    # Save as JSON
    with open(json_path, 'w') as f:
        json.dump(json_config, f, indent=2)
    
    print(f"✓ Converted {yaml_path} → {json_path}")
    print(f"  Servers: {len(json_config['mcpServers'])}")
    return True


def validate_config(config_path: str):
    """Validate an MCP config file (JSON or YAML)."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        print(f"Error: File not found: {config_path}")
        return False
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix == '.json':
                config = json.load(f)
            else:
                print(f"Error: Unsupported format: {config_path.suffix}")
                return False
        
        # Validate structure
        if 'mcpServers' not in config:
            print("Error: Missing 'mcpServers' key")
            return False
        
        servers = config['mcpServers']
        
        # Check format
        if isinstance(servers, dict):
            format_type = "JSON (dict)"
            server_list = [{'name': k, **v} for k, v in servers.items()]
        elif isinstance(servers, list):
            format_type = "YAML (list)"
            server_list = servers
        else:
            print("Error: Invalid mcpServers format")
            return False
        
        # Validate each server
        print(f"\n✓ Valid {format_type} config: {config_path}")
        print(f"  Servers: {len(server_list)}\n")
        
        for i, server in enumerate(server_list, 1):
            name = server.get('name', f'server_{i}')
            command = server.get('command', '?')
            args = server.get('args', [])
            env = server.get('env', {})
            
            print(f"  {i}. {name}")
            print(f"     Command: {command}")
            print(f"     Args: {len(args)} arguments")
            print(f"     Env: {len(env)} variables")
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """CLI interface for config conversion."""
    if len(sys.argv) < 2:
        print("""
MCP Config Converter

Usage:
  python mcp_config_converter.py <command> <input_file> [output_file]

Commands:
  json2yaml <file.json> [output.yaml]  - Convert JSON to YAML
  yaml2json <file.yaml> [output.json]  - Convert YAML to JSON  
  validate <file>                      - Validate config file

Examples:
  python mcp_config_converter.py json2yaml mcp_config.json
  python mcp_config_converter.py yaml2json mcp_servers.yaml
  python mcp_config_converter.py validate mcp_servers.yaml
        """)
        return
    
    command = sys.argv[1].lower()
    
    if command == 'json2yaml':
        if len(sys.argv) < 3:
            print("Error: Missing input file")
            return
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        json_to_yaml(input_file, output_file)
    
    elif command == 'yaml2json':
        if len(sys.argv) < 3:
            print("Error: Missing input file")
            return
        input_file = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        yaml_to_json(input_file, output_file)
    
    elif command == 'validate':
        if len(sys.argv) < 3:
            print("Error: Missing input file")
            return
        input_file = sys.argv[2]
        validate_config(input_file)
    
    else:
        print(f"Error: Unknown command '{command}'")
        print("Use: json2yaml, yaml2json, or validate")


if __name__ == "__main__":
    main()
