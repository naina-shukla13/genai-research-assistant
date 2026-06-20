"""
One-time (or re-runnable) ingestion script: loads documents, chunks them,
embeds them, and stores them in DuckDB.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.loader import load_knowledge_base
from rag.chunker import chunk_documents
from rag.embedder import embed_texts
from rag.vector_store import vector_store
from utils.logger import log_event, trace_step
from utils.exceptions import AssistantBaseException


def run_ingestion(clear_existing: bool = True) -> None:
    with trace_step("Ingestion", "full_pipeline"):
        try:
            documents = load_knowledge_base()
            log_event("Ingestion", "documents_loaded", count=len(documents))

            chunks = chunk_documents(documents)
            log_event("Ingestion", "chunks_created", count=len(chunks))

            texts = [c.text for c in chunks]
            embeddings = embed_texts(texts)
            log_event("Ingestion", "embeddings_generated", count=len(embeddings))

            if clear_existing:
                vector_store.clear()

            vector_store.insert_chunks(chunks, embeddings)
            log_event("Ingestion", "ingestion_completed", total_chunks=len(chunks))

            print(f"✅ Ingestion complete: {len(documents)} documents → {len(chunks)} chunks stored in DuckDB.")

        except AssistantBaseException as exc:
            print(f"❌ Ingestion failed: {exc}")
            raise


if __name__ == "__main__":
    run_ingestion()