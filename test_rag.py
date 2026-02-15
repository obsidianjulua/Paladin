import sys
import os
import time

# Ensure we can find the paladin package
sys.path.append(os.getcwd())

from paladin.agent import OllamaToolAgent, DEFAULT_MODEL

def test():
    print(f"Initializing agent with model: {DEFAULT_MODEL}...")
    
    # Create dummy doc
    with open("dummy_rag.md", "w") as f:
        f.write("# About Paladin\nPaladin is an AI agent that uses Ollama and local tools.\nIt was created by Grim.\nIt uses vector embeddings for memory.")
        
    try:
        agent = OllamaToolAgent(model_name=DEFAULT_MODEL)
        print("Agent initialized.")
        
        # 1. Ingest
        print("\n[TEST] Ingesting document...")
        query_ingest = "Ingest the file dummy_rag.md"
        response = agent.chat(query_ingest)
        print(f"Response: {response}")
        
        time.sleep(1) 
        
        # 2. Search
        print("\n[TEST] Searching document...")
        query_search = "Who created Paladin and what does it use?"
        response = agent.chat(query_search)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists("dummy_rag.md"):
            os.remove("dummy_rag.md")

if __name__ == "__main__":
    test()