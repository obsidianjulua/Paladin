#!/bin/bash
# Quick script to update the model in agent.py

echo "Current model configuration:"
grep "DEFAULT_MODEL" paladin/agent.py

echo ""
echo "Available better models for ReAct agents:"
echo "1. mistral:instruct (Recommended - fast and reliable)"
echo "2. llama3:8b-instruct (Good quality, moderate speed)"
echo "3. qwen2.5:7b-instruct (Best for tool calling)"
echo ""
echo "To change the model, edit paladin/agent.py line 24:"
echo '  DEFAULT_MODEL = "mistral:instruct"'
echo ""
echo "Then pull the model:"
echo '  ollama pull mistral:instruct'
