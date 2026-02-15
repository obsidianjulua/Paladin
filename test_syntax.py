import sys
import os
import time

# Ensure we can find the paladin package
sys.path.append(os.getcwd())

from paladin.agent import OllamaToolAgent, DEFAULT_MODEL

def test():
    print(f"Initializing agent with model: {DEFAULT_MODEL}...")
    
    try:
        agent = OllamaToolAgent(model_name=DEFAULT_MODEL, save_history=False)
        print("Agent initialized.")
        
        # Test 1: Chat Mode (//)
        print("\n[TEST 1] Chat Mode (//Hello)")
        query1 = "//Hello! How are you?"
        print(f"Query: {query1}")
        response1 = agent.chat(query1)
        print(f"Response: {response1}")

        # Test 2: Force Tool Mode (\\)
        print("\n[TEST 2] Force Tool Mode (\\\\echo)")
        # Note: double backslash in python string is single backslash
        query2 = "\\\\echo 'Force Tool Execution'"
        print(f"Query: {query2}")
        response2 = agent.chat(query2)
        print(f"Response: {response2}")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
