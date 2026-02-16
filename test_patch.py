from paladin.agent import OllamaToolAgent
import sys

try:
    print("Initializing agent...")
    agent = OllamaToolAgent(model_name="qwen3-coder:latest")
    print("🔧 Testing tool call...")
    response = agent.chat("Search my Obsidian vault for Paladin MCP guide")
    print("
Response:", response)
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
