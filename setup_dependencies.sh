#!/bin/bash
# Setup script for Paladin dependencies

echo "Installing Paladin dependencies..."
echo "=================================="

pip3 install --user numpy>=1.24.0
pip3 install --user langchain>=0.1.0
pip3 install --user langchain-ollama>=0.1.0
pip3 install --user rich>=13.0.0

echo ""
echo "=================================="
echo "Dependencies installed!"
echo "=================================="
echo ""
echo "To verify installation, run:"
echo "  python3 paladin/test_system.py"
