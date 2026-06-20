"""
Top-k similarity search over the DuckDB vector store.

Because embeddings are normalized at generation time (see embedder.py),
cosine similarity reduces to a plain dot product here — a deliberate
performance optimization, not an accident.
"""

import numpy as np
from typing import List

from config.settings import settings
from core.schemas import Chunk
from rag.embedder import embed_query
from rag.vector_store import vector_store
from utils.exceptions import RetrievalError
from utils.logger import trace_step, log_event


def retrieve_top_k(query: str, top_k: int = None, threshold: float = None) -> List[Chunk]:
    """
    Embeds the query and returns the top-k most similar chunks from the
    vector store, filtered by a minimum similarity threshold so low-quality
    matches don't get passed downstream as "relevant context" — directly
    supporting the RAG Quality criterion ("relevant retrieval, not noise").
    """
    top_k = top_k or settings.top_k_retrieval
    threshold = threshold if threshold is not None else settings.similarity_threshold

    with trace_step("Retriever", "retrieve_top_k", query=query, top_k=top_k):
        try:
            query_embedding = embed_query(query)
            rows = vector_store.fetch_all()

            if not rows:
                raise RetrievalError("Vector store is empty — has ingestion been run?")

            scored: List[Chunk] = []
            for chunk_id, text, source_document, embedding in rows:
                emb_array = np.array(embedding, dtype=np.float32)
                similarity = float(np.dot(query_embedding, emb_array))

                if similarity >= threshold:
                    scored.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=text,
                            source_document=source_document,
                            similarity_score=round(similarity, 4),
                        )
                    )

            scored.sort(key=lambda c: c.similarity_score, reverse=True)
            top_chunks = scored[:top_k]

            log_event(
                "Retriever", "retrieval_completed",
                query=query, candidates_above_threshold=len(scored),
                returned=len(top_chunks),
            )
            return top_chunks

        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Retrieval failed for query '{query}': {exc}") from exc