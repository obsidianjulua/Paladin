#!/usr/bin/env python3
"""
Comprehensive System Test for Paladin v2
Tests all components without requiring Ollama to be running
"""

import sys
import os
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test all module imports."""
    print("=" * 60)
    print("TEST 1: Module Imports")
    print("=" * 60)

    try:
        import numpy as np
        print("✓ numpy imported")
    except ImportError as e:
        print(f"✗ numpy import failed: {e}")
        return False

    try:
        from rich.console import Console
        print("✓ rich imported")
    except ImportError as e:
        print(f"✗ rich import failed: {e}")
        return False

    try:
        from langchain.agents import Tool
        print("✓ langchain imported")
    except ImportError as e:
        print(f"✗ langchain import failed: {e}")
        return False

    try:
        from langchain_ollama import ChatOllama
        print("✓ langchain-ollama imported")
    except ImportError as e:
        print(f"✗ langchain-ollama import failed: {e}")
        return False

    try:
        from paladin.vector_db import VectorDatabase
        from paladin.tools import FileTools, SystemTools, CompressionTools, AnalysisTools
        from paladin.tool_registry import ToolRegistry
        print("✓ All paladin modules imported")
    except ImportError as e:
        print(f"✗ Paladin module import failed: {e}")
        return False

    print("✓ All imports successful\n")
    return True


def test_tool_database():
    """Test tool database access."""
    print("=" * 60)
    print("TEST 2: Tool Database")
    print("=" * 60)

    db_path = "/home/grim/Desktop/op/TOOL_DB/tools.db"

    if not Path(db_path).exists():
        print(f"✗ Tool database not found at {db_path}")
        return False

    print(f"✓ Tool database exists at {db_path}")

    import sqlite3
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Count tools
        cursor.execute("SELECT COUNT(*) FROM tools WHERE enabled = 1")
        tool_count = cursor.fetchone()[0]
        print(f"✓ Found {tool_count} enabled tools")

        # Count templates
        cursor.execute("SELECT COUNT(*) FROM tool_templates WHERE enabled = 1")
        template_count = cursor.fetchone()[0]
        print(f"✓ Found {template_count} templates")

        # List tool categories
        cursor.execute("SELECT DISTINCT category FROM tools")
        categories = [row[0] for row in cursor.fetchall()]
        print(f"✓ Tool categories: {', '.join(categories)}")

        conn.close()
        print("✓ Tool database verified\n")
        return True

    except Exception as e:
        print(f"✗ Database error: {e}")
        return False


def test_tool_implementations():
    """Test individual tool implementations."""
    print("=" * 60)
    print("TEST 3: Tool Implementations")
    print("=" * 60)

    try:
        from paladin.tools import FileTools, SystemTools, CompressionTools, AnalysisTools

        # Test calculate
        result = SystemTools.calculate("2 + 2")
        assert "4" in result
        print("✓ SystemTools.calculate works")

        # Test create_file
        test_path = "/tmp/paladin_test.txt"
        result = FileTools.create_file(test_path, "Test content")
        assert "successfully" in result.lower()
        print("✓ FileTools.create_file works")

        # Test read_file
        result = FileTools.read_file(test_path)
        assert "Test content" in result
        print("✓ FileTools.read_file works")

        # Test list_files
        result = FileTools.list_files("/tmp", "paladin_*")
        assert "paladin_test.txt" in result
        print("✓ FileTools.list_files works")

        # Cleanup
        Path(test_path).unlink()

        print("✓ All tool implementations verified\n")
        return True

    except Exception as e:
        print(f"✗ Tool test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_registry():
    """Test tool registry loading."""
    print("=" * 60)
    print("TEST 4: Tool Registry")
    print("=" * 60)

    try:
        from paladin.tool_registry import ToolRegistry

        registry = ToolRegistry()
        print("✓ ToolRegistry initialized")

        tools = registry.load_tools()
        print(f"✓ Loaded {len(tools)} tools from database")

        # Verify tools are LangChain Tool objects
        from langchain.agents import Tool
        for tool in tools[:3]:  # Check first 3
            assert isinstance(tool, Tool)
            print(f"  • {tool.name}: {tool.description[:50]}...")

        print("✓ Tool registry verified\n")
        return True

    except Exception as e:
        print(f"✗ Tool registry test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vector_database():
    """Test vector database."""
    print("=" * 60)
    print("TEST 5: Vector Database")
    print("=" * 60)

    try:
        from paladin.vector_db import VectorDatabase

        # Use test database
        test_db = "/tmp/test_vector.db"
        if Path(test_db).exists():
            Path(test_db).unlink()

        db = VectorDatabase(db_path=test_db)
        print("✓ VectorDatabase initialized")

        # Test adding message
        db.add_message("test_session", "human", "Hello world")
        print("✓ Message added")

        # Test search
        results = db.similarity_search("hello", session_id="test_session")
        print(f"✓ Similarity search returned {len(results)} results")

        db.close()
        Path(test_db).unlink()

        print("✓ Vector database verified\n")
        return True

    except Exception as e:
        print(f"✗ Vector database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tool_wrappers():
    """Test tool wrapper execution."""
    print("=" * 60)
    print("TEST 6: Tool Wrapper Execution")
    print("=" * 60)

    try:
        from paladin.tool_registry import ToolRegistry
        import json

        registry = ToolRegistry()
        tools = registry.load_tools()

        # Find calculate tool
        calc_tool = next((t for t in tools if t.name == "calculate"), None)
        if calc_tool:
            result = calc_tool.func("5 + 3")
            assert "8" in result
            print(f"✓ calculate tool wrapper: {result}")

        # Find create_file tool
        create_tool = next((t for t in tools if t.name == "create_file"), None)
        if create_tool:
            params = json.dumps({"path": "/tmp/wrapper_test.txt", "content": "wrapper test"})
            result = create_tool.func(params)
            assert "successfully" in result.lower()
            print(f"✓ create_file tool wrapper: {result}")
            Path("/tmp/wrapper_test.txt").unlink()

        print("✓ Tool wrappers verified\n")
        return True

    except Exception as e:
        print(f"✗ Tool wrapper test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_permissions():
    """Test file and directory permissions."""
    print("=" * 60)
    print("TEST 7: File Permissions")
    print("=" * 60)

    tool_db = Path("/home/grim/Desktop/op/TOOL_DB/tools.db")
    paladin_dir = Path("/home/grim/Projects/Paladin/paladin")

    if not tool_db.exists():
        print(f"✗ Tool database missing: {tool_db}")
        return False
    print(f"✓ Tool database exists: {tool_db}")

    if not tool_db.is_file():
        print(f"✗ Tool database is not a file: {tool_db}")
        return False

    if not os.access(tool_db, os.R_OK):
        print(f"✗ Tool database not readable: {tool_db}")
        return False
    print(f"✓ Tool database is readable")

    if not paladin_dir.exists():
        print(f"✗ Paladin package directory missing: {paladin_dir}")
        return False
    print(f"✓ Paladin package directory exists")

    required_files = ["__init__.py", "agent.py", "tools.py", "tool_registry.py", "vector_db.py", "chat_interface.py"]
    for file in required_files:
        file_path = paladin_dir / file
        if not file_path.exists():
            print(f"✗ Missing file: {file}")
            return False
    print(f"✓ All required module files present")

    print("✓ File permissions verified\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("PALADIN v2.0 SYSTEM TEST")
    print("=" * 60 + "\n")

    tests = [
        ("Module Imports", test_imports),
        ("Tool Database", test_tool_database),
        ("Tool Implementations", test_tool_implementations),
        ("Tool Registry", test_tool_registry),
        ("Vector Database", test_vector_database),
        ("Tool Wrappers", test_tool_wrappers),
        ("File Permissions", test_file_permissions),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)

    if passed == total:
        print("\n✓ ALL TESTS PASSED - System ready to use!\n")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed - check errors above\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
