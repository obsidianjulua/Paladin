# Paladin

An advanced AI scripting agent and assistant built for Linux, designed to integrate deeply with your local environment, Obsidian vault, and development workflows.

## Features

- **Local LLM Execution**: Uses [Ollama](https://ollama.com/) with Qwen-based models (default: `qwen3-coder:latest`) for high-performance code generation and tool usage.
- **Native Tool Calling**: Implements a robust tool execution system compatible with Qwen's function calling format.
- **Vector Memory (RAG)**:
    - **Hybrid Search**: Combines FAISS (Facebook AI Similarity Search) for blazing-fast vector retrieval with SQLite for persistent storage.
    - **Embeddings**: Utilizes `qwen3-embedding:4b` (2560-dim) for high-quality semantic search.
    - **Context Awareness**: Retrieves relevant past conversations, tool outputs, and documents to provide continuity.
- **Obsidian Integration**:
    - **Vault Logging**: Automatically logs chat sessions to your daily notes or a specified file in your Obsidian vault.
    - **MCP Support**: Initial integration for Model Context Protocol to interact with external tools and data sources.
- **Project Context**: Aware of the current working directory and project-specific instructions (`PALADIN_INSTRUCTIONS.md`).

## Installation

1.  **Prerequisites**:
    - Python 3.10+
    - [Ollama](https://ollama.com/) installed and running.

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/obsidianjulua/Paladin.git
    cd Paladin
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `faiss-cpu` is required for vector search.*

4.  **Pull Models**:
    ```bash
    ollama pull qwen3-coder:latest
    ollama pull qwen3-embedding:4b
    ```

5.  **Configuration**:
    - Create a `.env` file based on the example (or use the defaults).
    - Key variables:
        - `PALADIN_MODEL`: The LLM model to use.
        - `PALADIN_EMBED_MODEL`: The embedding model.
        - `PALADIN_VAULT_PATH`: Absolute path to your Obsidian log file.

## Usage

Run the agent interactively:

```bash
python Paladin.py
```

### Modes
- **Chat Mode**: Standard conversation. The agent decides when to use tools.
- **Direct LLM (`//`)**: Bypass the agent loop and talk directly to the model (faster, no tools).
  - Example: `// What is the capital of France?`
- **Force Tool (``)**: Force the agent to use a tool.
  - Example: `\ Search my notes for "Linux"`

## Project Structure

- `paladin/`: Core package containing the agent logic, vector DB, and tool registry.
- `TOOL_DB/`: SQLite database storing tool definitions.
- `vector_chat.db`: SQLite database storing chat history and embeddings (mirrored to FAISS in-memory).

## License

[MIT License](LICENSE)
