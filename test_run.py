
import sys
import os

# Ensure we can find the paladin package
sys.path.append(os.getcwd())

from paladin.agent import OllamaToolAgent, DEFAULT_MODEL

def test():
    print(f"Initializing agent with model: {DEFAULT_MODEL}...")
    try:
        agent = OllamaToolAgent(model_name=DEFAULT_MODEL)
        print("Agent initialized successfully.")
        
        # Explicit path to ensure it lands where you want it
        target_path = os.path.expanduser("~/Desktop/data.txt")
        query = f"Create a file at {target_path} with the content 'Paladin Tool Test Successful'."
        print(f"\nSending query: {query}")
        
        response = agent.chat(query)
        print(f"\nResponse: {response}")
        
        # Verification
        if os.path.exists(target_path):
            with open(target_path, 'r') as f:
                content = f.read()
            print(f"\n[SUCCESS] File verified at {target_path}")
            print(f"Content: {content}")
            # Cleanup
            os.remove(target_path)
            print("File removed after test.")
        else:
            print(f"\n[FAILURE] File was not found at {target_path}")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test()
