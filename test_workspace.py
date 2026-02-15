import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure we can find the paladin package
sys.path.append(os.getcwd())

# Load env before import to ensure it's available
load_dotenv(override=True)

from paladin.agent import OllamaToolAgent

def test_simple_file_creation():
    model = os.getenv("PALADIN_MODEL", "qwen3:latest")
    print(f"Using Model: {model}")
    
    print("Initializing Paladin Agent...")
    try:
        # Pass the model explicitly
        agent = OllamaToolAgent(model_name=model)
        print("Agent initialized.")
        
        target_path = Path.home() / "Desktop" / "data.txt"
        prompt = f"Create a file at {target_path} with the text: 'Hello from Paladin! This confirms the tool works.'"
        
        print(f"\n[TEST] Sending prompt: {prompt}")
        response = agent.chat(prompt)
        print(f"Response: {response}")

    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_file_creation()
