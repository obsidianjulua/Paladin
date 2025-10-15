#!/usr/bin/env python3
"""
Paladin AI - Modular Tool-Calling System v2.0
Main entry point using new modular architecture
"""

import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import modular components
from paladin.chat_interface import run_chat_interface

def main():
    """Main entry point for Paladin v2."""
    run_chat_interface()

if __name__ == "__main__":
    main()
