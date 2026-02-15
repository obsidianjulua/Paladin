#!/usr/bin/env python3
"""
Standalone script to manually embed documents into the Paladin vector database.
This avoids loading the heavy LLM and only uses the lightweight embedding model.
"""

import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# Ensure we can find the paladin package
sys.path.append(os.getcwd())

# Load env before imports
load_dotenv(override=True)

from paladin.vector_db import VectorDatabase

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Paladin Document Embedder")
    parser.add_argument("path", nargs="?", help="File or directory path to ingest")
    parser.add_argument("-r", "--recursive", action="store_true", help="Recursively ingest directory")
    parser.add_argument("-l", "--list", action="store_true", help="List all ingested documents")
    parser.add_argument("--clear", action="store_true", help="Clear all documents from the database")
    parser.add_argument("-s", "--search", help="Search for documents matching query")

    args = parser.parse_args()

    console.print("[bold blue]Paladin Document Embedder[/bold blue]")
    
    try:
        db = VectorDatabase()
        console.print(f"Database: [green]{db.db_path}[/green]")
        console.print(f"Embedding Model: [green]{db.model_name}[/green] (Dimension: {db.embedding_dim})")

        if args.search:
            console.print(f"Searching for: [yellow]{args.search}[/yellow]")
            results = db.search_documents(args.search, limit=3)
            if not results:
                console.print("[red]No matches found.[/red]")
            else:
                table = Table(title=f"Search Results: {args.search}")
                table.add_column("Score", style="magenta")
                table.add_column("File", style="cyan")
                table.add_column("Content Preview", style="white")
                
                for res in results:
                    score = f"{res['similarity']:.4f}"
                    preview = res['content'][:100].replace('\n', ' ') + "..."
                    table.add_row(score, res['file_path'], preview)
                console.print(table)
            return

        if args.clear:
            confirm = input("Are you sure you want to clear ALL documents? (y/N): ")
            if confirm.lower() == 'y':
                cursor = db.conn.cursor()
                cursor.execute("DELETE FROM documents")
                db.conn.commit()
                console.print("[bold red]All documents cleared from database.[/bold red]")
            else:
                console.print("Operation cancelled.")
            return

        if args.list:
            docs = db.list_documents()
            if not docs:
                console.print("[yellow]No documents found in database.[/yellow]")
            else:
                table = Table(title="Ingested Documents")
                table.add_column("Path", style="cyan")
                for doc in docs:
                    table.add_row(doc)
                console.print(table)
            return

        if not args.path:
            parser.print_help()
            return

        target_path = Path(args.path)
        if not target_path.exists():
            console.print(f"[bold red]Error: Path '{args.path}' does not exist.[/bold red]")
            return

        if target_path.is_file():
            console.print(f"Ingesting file: {target_path}")
            result = db.ingest_document(str(target_path))
            console.print(f"[green]{result}[/green]")
        
        elif target_path.is_dir():
            console.print(f"Ingesting directory: {target_path} (Recursive: {args.recursive})")
            result = db.ingest_directory(str(target_path), recursive=args.recursive)
            console.print(f"[green]{result}[/green]")

    except Exception as e:
        console.print(f"[bold red]Critical Error: {e}[/bold red]")
        import traceback
        traceback.print_exc()
    finally:
        if 'db' in locals():
            db.close()

if __name__ == "__main__":
    main()
