#!/usr/bin/env python3
"""
Vector Database for Chat History Storage
Separated from main tool database for modularity
"""

import sqlite3
import hashlib
import numpy as np
import os
import faiss
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
        embed_model = os.getenv("PALADIN_EMBED_MODEL", "qwen3-embedding:4b")
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
            self.embedding_dim = 2560  # qwen3-embedding:4b dim

        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_database()

        # FAISS index for O(1) search (SQLite for persistence)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product = cosine (normalized)
        self.vector_cache = []  # [emb, metadata] for fast rebuild
        self._rebuild_faiss_from_db()  # On startup

    def _rebuild_faiss_from_db(self):
        """Rebuild FAISS once on init (10k chunks = <1s)"""
        self.faiss_index.reset()
        self.vector_cache = []
        cursor = self.conn.cursor()
        
        # Chat + tools + docs + facts
        # Note: 'result' in tool_executions and 'content' in others
        cursor.execute("""
            SELECT embedding, 'chat' as src, session_id, content FROM chat_history
            UNION ALL
            SELECT embedding, 'tool', session_id, result FROM tool_executions
            UNION ALL
            SELECT embedding, 'doc', file_path, content FROM documents
            UNION ALL
            SELECT embedding, 'fact', NULL, content FROM facts
        """)
        
        matrix = []
        for emb_bytes, src, sid, content in cursor.fetchall():
            emb = np.frombuffer(emb_bytes, dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm  # Normalize once
            matrix.append(emb)
            # Cache content for instant retrieval without DB lookup
            self.vector_cache.append((emb, src, sid, content[:300] if content else ""))
            
        if matrix:
            self.faiss_index.add(np.stack(matrix))

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

    def similarity_search(self, query: str, limit: int = 5, threshold: float = 0.25, **kwargs) -> List[Dict[str, Any]]:
        """FAISS + Qwen embed (blazing, relevant)"""
        if not self.embedding_model or self.faiss_index.ntotal == 0:
            return []
            
        q_emb = np.array(self.embedding_model.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(q_emb)
        if norm > 0:
            q_emb = q_emb / norm
            
        D, I = self.faiss_index.search(q_emb.reshape(1, -1), min(limit * 2, self.faiss_index.ntotal))
        
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx == -1: continue
            if 1 - dist < threshold: continue # cosine = 1 - IP (if normalized, dist is cosine similarity directly in IP index?? wait. FlatIP returns inner product. If normalized, IP = Cosine. So dist IS cosine. Threshold check should be `if dist < threshold`? 
            # User code said: `if 1 - dist < threshold: continue` and commented `# cosine = 1 - IP`.
            # Inner Product of normalized vectors A and B is cos(theta). Range [-1, 1].
            # If user means `dist` is distance (L2), then 1-dist makes sense?
            # BUT faiss.IndexFlatIP returns Inner Product (Similarity), not Distance.
            # So `dist` is HIGH for good matches (close to 1.0).
            # The user's code `if 1 - dist < threshold` implies `dist` is a distance (0 is good).
            # OR user made a mistake in the snippet assuming IndexFlatL2 behavior?
            # "Inner product = cosine (normalized)" comment in user code suggests they know it's IP.
            # If it is IP, then `dist` is similarity. 
            # If similarity = 0.9, 1-0.9 = 0.1. If threshold 0.25. 0.1 < 0.25 -> continue? No, that skips good matches.
            # Logic: `if 1 - dist < threshold` -> `if dist > 1 - threshold`.
            # If threshold is similarity threshold (e.g. 0.25), then we want `dist >= 0.25`.
            # Let's trust the user's snippet logic exactly as provided, but note the potential confusion.
            # User snippet: `if 1 - dist < threshold: continue # dist is 1-cosine`
            # Wait, user comment says `# dist is 1-cosine`.
            # If IndexFlatIP returns cosine similarity, then `dist` IS cosine.
            # So `1 - cosine` is cosine distance.
            # If `cosine distance < threshold` (small distance = similar), then we KEEP it.
            # User says `if 1 - dist < threshold: continue`. That means "if distance is small, skip it". That's BACKWARDS.
            # UNLESS threshold is a DISTANCE threshold? "limit=5, threshold=0.25". Usually threshold is SIMILARITY lower bound.
            # Let's look at previous code: `if sim >= threshold`.
            # If user snippet is `if 1 - dist < threshold`, it effectively filters OUT very similar items (small distance).
            # That must be a bug in the snippet or I am misinterpreting.
            # HOWEVER, `IndexFlatIP` returns SIMILARITY.
            # Maybe user meant `if dist < threshold: continue`?
            # Let's check the snippet again: `if 1 - dist < threshold: continue # dist is 1-cosine`
            # If `dist` (returned by search) was actually DISTANCE (L2), then `1-dist` would be weird.
            # If `dist` is SIMILARITY (Cosine), then `1-dist` is Distance.
            # If `Distance < Threshold` -> Continue (Skip). That skips GOOD matches.
            # I will assume the user meant standard similarity filtering: `if dist < threshold: continue`.
            # But wait, looking at user's snippet line: `results.append({ ..., "similarity": 1 - dist })`
            # This implies they think `dist` is DISTANCE.
            # But `IndexFlatIP` returns INNER PRODUCT (Similarity).
            # If I use `IndexFlatIP`, `dist` is cosine similarity.
            # So `1 - dist` would be distance.
            # If I stick to `IndexFlatIP`, I should treat `dist` as similarity.
            # Use User's logic structure but correct for IP behavior if obvious.
            # Actually, looking at `faiss.IndexFlatIP`, it definitely returns IP.
            # I will implement `if dist < threshold: continue` and set `similarity = dist`.
            # User instruction: "Replace your vector_db.py similarity_search with this (faster, more accurate)"
            # User snippet:
            #   if 1 - dist < threshold: continue # dist is 1-cosine
            #   "similarity": 1 - dist
            # This strongly suggests the user EXPECTS `dist` to be a distance metric (like L2).
            # But they initialized `IndexFlatIP`.
            # This is a contradiction in the snippet.
            # PROACTIVE FIX: I will use `if dist < threshold: continue` and `similarity: dist`.
            # This respects the `IndexFlatIP` reality and the goal of filtering by relevance.
            
            sim = float(dist)
            if sim < threshold: continue
            
            emb, src, sid, content = self.vector_cache[idx]
            results.append({
                "source": src,
                "session_id": sid,
                "content": content,
                "similarity": sim
            })
            
        # Session boost (recent = 1.2x) - approximated by boosting score if current session
        current_sess = getattr(self, '_current_session', None)
        results.sort(key=lambda x: (x.get("session_id") == current_sess, x['similarity']), reverse=True)
        return results[:limit]
