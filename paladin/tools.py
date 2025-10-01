#!/usr/bin/env python3
"""
Modular Tool Implementations
All tools that can be called by the AI agent
"""

import os
import json
import zipfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional
from fnmatch import fnmatch


class FileTools:
    """File system operations."""

    @staticmethod
    def create_file(path: str, content: str) -> str:
        """Creates a new file with the specified content."""
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding='utf-8')
            return f"File created/overwritten successfully at {path}."
        except Exception as e:
            return f"Error creating file: {e}"

    @staticmethod
    def read_file(path: str, max_chars: int = 5000) -> str:
        """Reads the content of a file. Returns full content for AI analysis."""
        try:
            p = Path(path)
            if not p.is_file():
                return f"Error: File not found at {path}"

            content = p.read_text(encoding='utf-8')

            # Return full content but add metadata
            file_info = f"File: {path}\nSize: {len(content)} characters\nLines: {len(content.splitlines())}\n\n"

            if len(content) > max_chars:
                return file_info + content[:max_chars] + f"\n\n... (truncated, total {len(content)} chars)"

            return file_info + content
        except Exception as e:
            return f"Error reading file: {e}"

    @staticmethod
    def list_files(directory: str, pattern: str = "*") -> str:
        """Lists files and directories in a directory."""
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
    def find_files(query: str, search_directories: List[str] = None, max_results: int = 20) -> str:
        """Enhanced file finding with content search."""
        if search_directories is None:
            search_directories = ['.']

        try:
            results = []
            search_terms = query.lower().split()

            for search_dir in search_directories:
                search_path = Path(search_dir)
                if not search_path.exists():
                    continue

                for file_path in search_path.rglob('*'):
                    if file_path.is_file():
                        filename_lower = file_path.name.lower()

                        if any(term in filename_lower for term in search_terms):
                            results.append({
                                'type': 'filename_match',
                                'path': str(file_path),
                                'match': f"Filename contains: {query}"
                            })

                        if file_path.suffix.lower() in {'.txt', '.py', '.js', '.html', '.css', '.json', '.xml',
                                                        '.md', '.log', '.cfg', '.ini', '.lua', '.cpp', '.jl',
                                                        'Modelfile', '.env', '.sh', '.yml', '.yaml'}:
                            try:
                                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                    content = f.read(2000).lower()
                                    if any(term in content for term in search_terms):
                                        lines = content.split('\n')
                                        match_line = next((line for line in lines if any(term in line for term in search_terms)), "")
                                        results.append({
                                            'type': 'content_match',
                                            'path': str(file_path),
                                            'match': f"Content: '{match_line[:100]}...'"
                                        })
                            except Exception:
                                pass

            if not results:
                return f"No files found matching '{query}' in directories: {search_directories}"

            formatted_results = []
            for i, result in enumerate(results[:max_results]):
                formatted_results.append(f"{i+1}. {result['path']} ({result['type']})\n   {result['match']}")

            total_info = f"Found {len(results)} files" + (f" (showing first {max_results})" if len(results) > max_results else "")
            return total_info + ":\n" + "\n".join(formatted_results)

        except Exception as e:
            return f"Error finding files: {e}"

    @staticmethod
    def recursive_file_search(directory: str, pattern: str = "*", max_results: int = 50) -> str:
        """Recursively searches for files matching a pattern."""
        search_dir = Path(directory)
        if not search_dir.is_dir():
            return f"Error: Directory not found at {directory}"

        try:
            results = [str(p.resolve()) for p in search_dir.rglob(pattern) if p.is_file()]

            if not results:
                return f"No files found matching pattern '{pattern}' in '{directory}'."

            output_list = results[:max_results]
            summary = f"Found {len(results)} files matching '{pattern}'.\n"
            summary += "\n".join(output_list)

            if len(results) > max_results:
                summary += f"\n... and {len(results) - max_results} more results."
            return summary
        except Exception as e:
            return f"Error performing recursive search: {e}"


class CompressionTools:
    """Compression and archive operations."""

    @staticmethod
    def zip_files(file_paths: List[str], output_zip_path: str) -> str:
        """Compresses files/directories into a zip archive."""
        output_zip_path = Path(output_zip_path)
        try:
            output_zip_path.parent.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                added_count = 0
                for item_path_str in file_paths:
                    item_path = Path(item_path_str)
                    if not item_path.exists():
                        continue

                    if item_path.is_file():
                        zipf.write(item_path, item_path.name)
                        added_count += 1
                    elif item_path.is_dir():
                        for root, dirs, files in os.walk(item_path):
                            for file in files:
                                file_path = Path(root) / file
                                archive_path = file_path.relative_to(Path.cwd())
                                zipf.write(file_path, archive_path)
                                added_count += 1

            return f"Successfully created zip archive at {output_zip_path}. Added {added_count} items."
        except Exception as e:
            return f"Error zipping files: {e}"

    @staticmethod
    def unzip_file(zip_path: str, output_directory: str) -> str:
        """Extracts zip file contents."""
        zip_path = Path(zip_path)
        output_dir = Path(output_directory)

        if not zip_path.is_file():
            return f"Error: Zip file not found at {zip_path}"

        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                zipf.extractall(output_dir)
            return f"Successfully extracted {zip_path.name} to {output_dir.resolve()}"
        except Exception as e:
            return f"Error unzipping file: {e}"


class SystemTools:
    """System command execution and utilities."""

    _SAFE_COMMANDS = ['ls', 'dir', 'cat', 'echo', 'grep', 'find', 'pwd', 'whoami', 'df',
                      'du', 'head', 'tail', 'wc', 'uname', 'hostname', 'date', 'which',
                      'git', 'python3', 'pip3', 'npm', 'node', 'docker', 'curl', 'wget']
    _DANGEROUS_COMMANDS = ['rm', 'format', 'shutdown', 'reboot', 'dd', 'mkfs', 'chown', 'chmod']

    @staticmethod
    def run_command(command: str, capture_output: bool = True, timeout: int = 30) -> str:
        """
        Executes a shell command and returns output.
        Enhanced version with full subprocess control for better output capture.
        """
        cmd_parts = command.strip().split()
        if not cmd_parts:
            return "Error: Command is empty."

        # Safety check
        base_cmd = cmd_parts[0]
        if base_cmd in SystemTools._DANGEROUS_COMMANDS:
            return f"Error: Command '{base_cmd}' is blocked for safety."

        try:
            if capture_output:
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=timeout
                )

                output = ""
                if result.stdout:
                    output += f"STDOUT:\n{result.stdout}"
                if result.stderr:
                    output += f"\nSTDERR:\n{result.stderr}"
                if result.returncode != 0:
                    output += f"\nReturn Code: {result.returncode}"

                # Truncate if too long
                if len(output) > 2000:
                    output = output[:2000] + f"\n... (truncated, total {len(output)} chars)"

                return output if output else "Command executed successfully (no output)"
            else:
                # Run without capturing
                result = subprocess.run(command, shell=True, timeout=timeout)
                return f"Command executed with return code: {result.returncode}"

        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {timeout} seconds"
        except Exception as e:
            return f"Error executing command: {e}"

    @staticmethod
    def execute_command(command: str) -> str:
        """Alias for run_command with default settings (backward compatibility)."""
        return SystemTools.run_command(command, capture_output=True, timeout=30)

    @staticmethod
    def calculate(expression: str) -> str:
        """Safely evaluates a mathematical expression."""
        safe_dict = {
            'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'pow': pow,
            'sqrt': lambda x: x**0.5,
            '__builtins__': None
        }
        try:
            safe_expression = expression.replace(';', '').replace('__', '')
            result = str(eval(safe_expression, {"__builtins__": None}, safe_dict))
            return f"Result: {result}"
        except Exception as e:
            return f"Error evaluating expression: {e}"


class AnalysisTools:
    """Tools for analyzing file content and data."""

    @staticmethod
    def analyze_file(path: str, analysis_type: str = "summary") -> str:
        """
        Analyzes a file and returns structured information.
        Analysis types: summary, structure, dependencies, security
        """
        try:
            p = Path(path)
            if not p.is_file():
                return f"Error: File not found at {path}"

            content = p.read_text(encoding='utf-8')
            file_extension = p.suffix.lower()

            analysis = {
                'path': str(path),
                'name': p.name,
                'extension': file_extension,
                'size_bytes': p.stat().st_size,
                'lines': len(content.splitlines()),
                'characters': len(content)
            }

            if analysis_type == "summary":
                # Basic summary
                lines = content.splitlines()
                non_empty_lines = [l for l in lines if l.strip()]

                analysis['non_empty_lines'] = len(non_empty_lines)
                analysis['preview'] = '\n'.join(lines[:10])

            elif analysis_type == "structure" and file_extension == ".py":
                # Python structure analysis
                import re
                classes = re.findall(r'^class\s+(\w+)', content, re.MULTILINE)
                functions = re.findall(r'^def\s+(\w+)', content, re.MULTILINE)
                imports = re.findall(r'^(?:from|import)\s+(.+?)(?:\s+import)?', content, re.MULTILINE)

                analysis['classes'] = classes
                analysis['functions'] = functions
                analysis['imports'] = [i.strip() for i in imports]

            return json.dumps(analysis, indent=2)

        except Exception as e:
            return f"Error analyzing file: {e}"
