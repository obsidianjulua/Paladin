#!/usr/bin/env python3
"""
Diagnostic script to test agent execution and identify hanging issues
"""

import sys
import logging
from paladin.agent import OllamaToolAgent

# Enable debug logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_simple_command():
    """Test a simple command to see where it hangs"""
    print("=" * 80)
    print("TESTING AGENT EXECUTION")
    print("=" * 80)

    try:
        agent = OllamaToolAgent(model_name='codellama:7b-instruct-q4_K_M')
        print(f"\nAgent initialized with {len(agent.tools)} tools")
        print("\nSending test query: 'List files in the current directory'")
        print("-" * 80)

        # Try a simple command
        response = agent.chat("List files in the current directory")

        print("-" * 80)
        print(f"RESPONSE: {response}")
        print("=" * 80)
        print("SUCCESS!")

    except KeyboardInterrupt:
        print("\n\nINTERRUPTED BY USER")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if 'agent' in locals():
            agent.cleanup()

if __name__ == "__main__":
    test_simple_command()
