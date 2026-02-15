import sys
import os
import time

# Ensure we can find the paladin package
sys.path.append(os.getcwd())

from paladin.agent import OllamaToolAgent, DEFAULT_MODEL

def test():
    print(f"Initializing agent with model: {DEFAULT_MODEL}...")
    
    try:
        # Initialize with history saving DISABLED for this test
        agent = OllamaToolAgent(model_name=DEFAULT_MODEL, save_history=False)
        print("Agent initialized (History Saving: DISABLED).")
        
        # Admin Task 1: Grep task
        # "Find all TODOs in the paladin directory recursively"
        # We'll plant a dummy TODO first just in case
        dummy_file = "paladin/todo_test.py"
        with open(dummy_file, "w") as f:
            f.write("# TODO: Refactor this later\nprint('test')")
            
        print("\n[TEST] Executing Admin Task (Grep)...")
        query = "Find all lines containing 'TODO' in the paladin directory and show me the line numbers."
        print(f"Query: {query}")
        
        response = agent.chat(query)
        print(f"Response: {response}")
        
        # Cleanup
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
            
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()