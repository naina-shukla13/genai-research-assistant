"""
DuckDB-backed vector store.

Design note (important for the README): DuckDB does not have a native
ANN/vector index in the version we pin here. For this case study's scale
(a handful of documents, low hundreds of chunks), brute-force cosine
similarity computed in Python over arrays fetched from DuckDB is the
right tradeoff — it's simple, correct, and fast enough. We call this out
explicitly rather than pretending DuckDB does something it doesn't.
"""

import duckdb
import numpy as np
from typing import List
import os

from config.settings import settings
from core.schemas import Chunk
from rag.chunker import TextChunk
from utils.exceptions import VectorStoreError
from utils.logger import trace_step, log_event


class DuckDBVectorStore:
    def __init__(self, db_path: str = None) -> None:
        self.db_path = db_path or settings.duckdb_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        try:
            self._conn = duckdb.connect(self.db_path)
            self._init_schema()
        except Exception as exc:
            raise VectorStoreError(f"Failed to connect to DuckDB at {self.db_path}: {exc}") from exc

    def _init_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id VARCHAR PRIMARY KEY,
                text VARCHAR,
                source_document VARCHAR,
                embedding DOUBLE[]
            )
            """
        )

    def is_empty(self) -> bool:
        count = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return count == 0

    def clear(self) -> None:
        self._conn.execute("DELETE FROM chunks")
        log_event("VectorStore", "store_cleared")

    def insert_chunks(self, chunks: List[TextChunk], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Chunk/embedding count mismatch: {len(chunks)} vs {len(embeddings)}"
            )

        with trace_step("VectorStore", "insert_chunks", count=len(chunks)):
            try:
                rows = [
                    (chunk.chunk_id, chunk.text, chunk.source_document, embedding.tolist())
                    for chunk, embedding in zip(chunks, embeddings)
                ]
                self._conn.executemany(
                    "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?)", rows
                )
            except Exception as exc:
                raise VectorStoreError(f"Failed to insert chunks: {exc}") from exc

        log_event("VectorStore", "chunks_inserted", count=len(chunks))

    def fetch_all(self):
        """Returns all (chunk_id, text, source_document, embedding) rows."""
        try:
            return self._conn.execute(
                "SELECT chunk_id, text, source_document, embedding FROM chunks"
            ).fetchall()
        except Exception as exc:
            raise VectorStoreError(f"Failed to fetch chunks: {exc}") from exc

    def close(self) -> None:
        self._conn.close()


# Singleton — import as `from rag.vector_store import vector_store`
vector_store = DuckDBVectorStore()