#!/usr/bin/env python3
"""
Vector Database for Chat History Storage
Separated from main tool database for modularity
"""

import sqlite3
import hashlib
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings

load_dotenv()


class VectorDatabase:
    """Simple vector database using SQLite and Ollama embeddings for offline use."""

    def __init__(self, db_path: str = None, model_name: str = "default"):
        # Default to env var or user's data directory
        if db_path is None:
            db_path = os.getenv("PALADIN_VECTOR_DB") or str(Path.home() / ".paladin" / "vector_chat.db")

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.model_name = model_name
        
        # Initialize real embeddings
        # Use nomic-embed-text for speed/quality balance if available, else fall back to the main model
        embed_model = os.getenv("PALADIN_EMBED_MODEL", "nomic-embed-text:latest")
        try:
            self.embedding_model = OllamaEmbeddings(model=embed_model)
            # Test embedding to get dimension
            test_emb = self.embedding_model.embed_query("test")
            self.embedding_dim = len(test_emb)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Warning: Could not initialize OllamaEmbeddings: {e}. Falling back to random.")
            self.embedding_model = None
            self.embedding_dim = 768  # Assumption

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_database()

    def _init_database(self):
        """Initialize the vector database tables."""
        cursor = self.conn.cursor()
        
        # We need to recreate tables if dimensions don't match or for v2 upgrade
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chat_history'")
        if cursor.fetchone():
            # Check a sample embedding size if possible, or just drop for this upgrade
            # For this "2.0" upgrade, we'll force a clean slate to ensure embedding consistency
            # Comment out the DROP lines if you want to preserve old (incompatible) data
            cursor.execute("DROP TABLE IF EXISTS chat_history")
            cursor.execute("DROP TABLE IF EXISTS tool_executions")
            cursor.execute("DROP TABLE IF EXISTS documents")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                message_type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                content_hash TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_executions (
                id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tool_name TEXT NOT NULL,
                parameters TEXT,
                result TEXT,
                embedding BLOB NOT NULL,
                content_hash TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY,
                content TEXT NOT NULL,
                embedding BLOB NOT NULL,
                timestamp TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_session_id ON chat_history(session_id, id DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_session_id ON tool_executions(session_id, id DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_docs_path ON documents(file_path)")
        self.conn.commit()

    def close(self):
        """Close the persistent database connection."""
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates embedding for the given text."""
        if not text:
            return np.zeros(self.embedding_dim, dtype=np.float32)
            
        if self.embedding_model:
            try:
                embedding_val = self.embedding_model.embed_query(text)
                return np.array(embedding_val, dtype=np.float32)
            except Exception as e:
                print(f"Embedding error: {e}")
                return np.zeros(self.embedding_dim, dtype=np.float32)
        
        # Fallback dummy (should not happen in prod)
        np.random.seed(len(text))
        return np.random.rand(self.embedding_dim).astype(np.float32)

    def ingest_document(self, file_path: str, chunk_size: int = 1000, overlap: int = 100) -> str:
        """Ingests a document, chunks it, and stores it in the vector DB."""
        path = Path(file_path)
        if not path.exists():
            return f"Error: File {file_path} not found."
            
        try:
            content = path.read_text(encoding='utf-8')
            
            # Simple chunking
            chunks = []
            start = 0
            while start < len(content):
                end = start + chunk_size
                chunk = content[start:end]
                chunks.append(chunk)
                start += (chunk_size - overlap)
                
            cursor = self.conn.cursor()
            timestamp = datetime.now().isoformat()
            
            # Delete old versions of this file
            cursor.execute("DELETE FROM documents WHERE file_path = ?", (str(path),))
            
            count = 0
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                    
                emb = self._get_embedding(chunk).tobytes()
                cursor.execute("""
                    INSERT INTO documents (file_path, chunk_index, content, embedding, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (str(path), i, chunk, emb, timestamp))
                count += 1
                
            self.conn.commit()
            return f"Successfully ingested {count} chunks from {file_path} into vector database."
            
        except Exception as e:
            return f"Error ingesting document: {e}"

    def ingest_directory(self, directory_path: str, recursive: bool = True) -> str:
        """Ingests all supported text files in a directory."""
        path = Path(directory_path)
        if not path.exists():
            return f"Error: Directory {directory_path} not found."
        
        allowed_extensions = {
            '.md', '.txt', '.py', '.js', '.json', '.html', '.css', '.sh', 
            '.yml', '.yaml', '.ini', '.cfg', '.toml', '.xml', '.java', 
            '.cpp', '.c', '.h', '.hpp', '.rs', '.go', '.ts', '.tsx', '.jsx',
            '.jl'
        }
        
        files_to_ingest = []
        if recursive:
            for ext in allowed_extensions:
                files_to_ingest.extend(path.rglob(f"*{ext}"))
        else:
            for ext in allowed_extensions:
                files_to_ingest.extend(path.glob(f"*{ext}"))
        
        results = []
        success_count = 0
        
        for file_path in files_to_ingest:
            if file_path.is_file():
                # Skip dotfiles/folders implicitly via rglob/glob usually, but good to be safe
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                    
                res = self.ingest_document(str(file_path))
                if "Successfully" in res:
                    success_count += 1
                results.append(f"{file_path.name}: {res}")
        
        summary = f"Ingestion complete. Processed {len(files_to_ingest)} files. Successfully ingested: {success_count}.\n"
        details = "\n".join(results[:10]) + ("\n..." if len(results) > 10 else "")
        return summary + details

    def list_documents(self) -> List[str]:
        """Lists all currently ingested documents."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT DISTINCT file_path FROM documents ORDER BY file_path")
        return [row[0] for row in cursor.fetchall()]

    def add_fact(self, content: str) -> str:
        """Adds a permanent fact to the database."""
        try:
            cursor = self.conn.cursor()
            timestamp = datetime.now().isoformat()
            emb = self._get_embedding(content).tobytes()
            
            cursor.execute("""
                INSERT INTO facts (content, embedding, timestamp)
                VALUES (?, ?, ?)
            """, (content, emb, timestamp))
            self.conn.commit()
            return "Fact saved successfully."
        except Exception as e:
            return f"Error saving fact: {e}"

    def get_all_facts(self) -> List[str]:
        """Retrieves all stored facts."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT content FROM facts ORDER BY timestamp DESC")
        return [row[0] for row in cursor.fetchall()]

    def search_documents(self, query: str, limit: int = 5, threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Search ingested documents."""
        query_embedding = self._get_embedding(query)
        cursor = self.conn.cursor()
        results = []

        def cosine(a: np.ndarray, b: np.ndarray) -> float:
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0.0:
                return 0.0
            return float(np.dot(a, b) / denom)

        # Retrieve candidates (dumb scan for now, optimized later if needed)
        cursor.execute("SELECT file_path, chunk_index, content, embedding FROM documents")
        for file_path, idx, content, emb_bytes in cursor.fetchall():
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = cosine(query_embedding, emb)
            if sim >= threshold:
                results.append({
                    "source": "document",
                    "file_path": file_path,
                    "chunk": idx,
                    "content": content,
                    "similarity": sim
                })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]

    def _insert_record(self, table: str, session_id: str, content: str, *, message_type: Optional[str] = None,
                      tool_name: Optional[str] = None, parameters: Optional[str] = None, result: Optional[str] = None):
        """Inserts a record into the database."""
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        embedding = self._get_embedding(content).tobytes()
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if table == "chat_history":
            cursor.execute("""
                INSERT INTO chat_history (session_id, timestamp, message_type, content, embedding, content_hash)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, timestamp, message_type, content, embedding, content_hash))
        elif table == "tool_executions":
            cursor.execute("""
                INSERT INTO tool_executions (session_id, timestamp, tool_name, parameters, result, embedding, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (session_id, timestamp, tool_name, parameters, result, embedding, content_hash))
        self.conn.commit()


    def add_message(self, session_id: str, message_type: str, content: str):
        """Adds a chat message to the history."""
        self._insert_record("chat_history", session_id, content, message_type=message_type)

    def add_tool_execution(self, session_id: str, tool_name: str, parameters: Dict[str, Any], result: str):
        """Adds a tool execution record."""
        import json
        content = f"Tool: {tool_name}, Parameters: {json.dumps(parameters, ensure_ascii=False)}, Result: {result}"
        self._insert_record(
            "tool_executions",
            session_id,
            content,
            tool_name=tool_name,
            parameters=json.dumps(parameters, ensure_ascii=False),
            result=result
        )

    def get_recent_history(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        """Retrieves the most recent chat messages for a session."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT message_type, content 
            FROM chat_history 
            WHERE session_id = ? 
            ORDER BY id DESC 
            LIMIT ?
        """, (session_id, limit))
        
        # Reverse to get chronological order (oldest -> newest)
        rows = cursor.fetchall()
        history = [{"type": row[0], "content": row[1]} for row in reversed(rows)]
        return history

    def similarity_search(self, query: str, session_id: Optional[str] = None, limit: int = 5,
                         threshold: float = 0.3, include_other_sessions: bool = True) -> List[Dict[str, Any]]:
        """Performs a vector similarity search across chat and tool history."""
        query_embedding = self._get_embedding(query)
        cursor = self.conn.cursor()
        results = []

        def cosine(a: np.ndarray, b: np.ndarray) -> float:
            denom = float(np.linalg.norm(a) * np.linalg.norm(b))
            if denom == 0.0:
                return 0.0
            return float(np.dot(a, b) / denom)

        where_clause = ""
        params: List[Any] = []
        if not include_other_sessions and session_id:
            where_clause = "WHERE session_id = ?"
            params.append(session_id)

        # Search chat history
        cursor.execute(f"SELECT session_id, message_type, content, embedding FROM chat_history {where_clause} ORDER BY id DESC LIMIT 200", params)
        for sess, message_type, content, emb_bytes in cursor.fetchall():
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = cosine(query_embedding, emb)
            if sim >= threshold:
                results.append({
                    "source": "chat",
                    "session_id": sess,
                    "type": message_type,
                    "content": content,
                    "similarity": sim
                })

        # Search tool history
        cursor.execute(f"SELECT session_id, tool_name, parameters, result, embedding FROM tool_executions {where_clause} ORDER BY id DESC LIMIT 200", params)
        for sess, tool_name, parameters, result, emb_bytes in cursor.fetchall():
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = cosine(query_embedding, emb)
            if sim >= threshold:
                results.append({
                    "source": "tool_execution",
                    "session_id": sess,
                    "tool": tool_name,
                    "parameters": parameters,
                    "result": result,
                    "similarity": sim
                })

        # Search documents (global knowledge, not bound by session)
        cursor.execute("SELECT file_path, chunk_index, content, embedding FROM documents")
        for file_path, idx, content, emb_bytes in cursor.fetchall():
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = cosine(query_embedding, emb)
            if sim >= threshold:
                results.append({
                    "source": "document",
                    "file_path": file_path,
                    "chunk": idx,
                    "content": content,
                    "similarity": sim
                })

        # Search facts (permanent knowledge)
        cursor.execute("SELECT content, embedding FROM facts")
        for content, emb_bytes in cursor.fetchall():
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            sim = cosine(query_embedding, emb)
            if sim >= threshold:
                results.append({
                    "source": "fact",
                    "content": content,
                    "similarity": sim
                })

        if session_id:
            results.sort(key=lambda x: ((x.get("session_id") == session_id), x['similarity']), reverse=True)
        else:
            results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
