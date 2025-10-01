#!/usr/bin/env python3
"""
Vector Database for Chat History Storage
Separated from main tool database for modularity
"""

import sqlite3
import hashlib
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path


class VectorDatabase:
    """Simple vector database using SQLite and hash-based embeddings for offline use."""

    def __init__(self, db_path: str = None, model_name: str = "default"):
        # Default to user's data directory
        if db_path is None:
            db_path = str(Path.home() / ".paladin" / "vector_chat.db")

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.model_name = model_name
        self.embedding_dim = 64
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_database()

    def _init_database(self):
        """Initialize the vector database tables."""
        cursor = self.conn.cursor()
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chat_session_id ON chat_history(session_id, id DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_session_id ON tool_executions(session_id, id DESC)")
        self.conn.commit()

    def close(self):
        """Close the persistent database connection."""
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    def _get_embedding(self, text: str) -> np.ndarray:
        """Generates a simple hash-based embedding for the given text (offline compatible)."""
        text_lower = (text or "").lower()
        if not text_lower:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        embeddings = []
        for i in range(self.embedding_dim):
            rotated_text = text_lower[i % len(text_lower):] + text_lower[:i % len(text_lower)]
            hash_obj = hashlib.md5(f"{rotated_text}_{i}".encode())
            hash_int = int(hash_obj.hexdigest()[:8], 16)
            embedding_val = (hash_int % 2000 - 1000) / 1000.0
            embeddings.append(embedding_val)
        return np.array(embeddings, dtype=np.float32)

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

        if session_id:
            results.sort(key=lambda x: ((x.get("session_id") == session_id), x['similarity']), reverse=True)
        else:
            results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:limit]
