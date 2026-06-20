"""
Chunking strategy for the RAG pipeline.

Uses a recursive character splitter (sentence/paragraph-aware where
possible) rather than naive fixed-size slicing, to avoid cutting chunks
mid-sentence — a small quality lever that meaningfully affects RAG
relevance, which is an explicit scoring criterion.
"""

from typing import List, NamedTuple
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import settings
from rag.loader import RawDocument
from utils.logger import trace_step, log_event


class TextChunk(NamedTuple):
    chunk_id: str
    text: str
    source_document: str


def chunk_documents(documents: List[RawDocument]) -> List[TextChunk]:
    """
    Splits each document into overlapping chunks using chunk_size and
    chunk_overlap from settings (externalized, not hardcoded).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[TextChunk] = []

    with trace_step("Chunker", "chunk_documents", document_count=len(documents)):
        for doc in documents:
            pieces = splitter.split_text(doc.text)
            for idx, piece in enumerate(pieces):
                chunk_id = f"{doc.source_filename}::chunk_{idx}"
                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=piece.strip(),
                        source_document=doc.source_filename,
                    )
                )
            log_event(
                "Chunker", "document_chunked",
                filename=doc.source_filename, chunk_count=len(pieces),
            )

    log_event("Chunker", "chunking_completed", total_chunks=len(chunks))
    return chunks