"""
Document loader for the knowledge base.

Supports PDF, Markdown, and plain text — matching the case study's
explicit "Load documents (PDF, text, or markdown)" requirement.
Each loader returns raw text tagged with its source filename, so every
chunk downstream can be traced back to exactly which document it came
from (needed for the Chunk.source_document field in our schema).
"""

import os
from pathlib import Path
from typing import List, NamedTuple

from pypdf import PdfReader

from config.settings import settings
from utils.exceptions import DocumentLoadError
from utils.logger import trace_step, log_event


class RawDocument(NamedTuple):
    source_filename: str
    text: str


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def _load_pdf(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise DocumentLoadError(f"Failed to read PDF {path.name}: {exc}") from exc


def _load_text_like(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:
        raise DocumentLoadError(f"Failed to read file {path.name}: {exc}") from exc


def load_knowledge_base(directory: str = None) -> List[RawDocument]:
    """
    Loads every supported document from the knowledge base directory.

    Skips unsupported file types with a warning log rather than failing
    the whole pipeline — partial knowledge base availability is better
    than a hard crash on one bad file.
    """
    kb_dir = Path(directory or settings.knowledge_base_dir)

    if not kb_dir.exists():
        raise DocumentLoadError(f"Knowledge base directory not found: {kb_dir}")

    documents: List[RawDocument] = []

    with trace_step("Loader", "load_knowledge_base", directory=str(kb_dir)):
        for file_path in sorted(kb_dir.iterdir()):
            ext = file_path.suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                log_event("Loader", "skipped_unsupported_file", filename=file_path.name)
                continue

            try:
                text = (
                    _load_pdf(file_path)
                    if ext == ".pdf"
                    else _load_text_like(file_path)
                )
                if not text.strip():
                    log_event("Loader", "empty_document_skipped", filename=file_path.name)
                    continue

                documents.append(RawDocument(source_filename=file_path.name, text=text))
                log_event(
                    "Loader", "document_loaded",
                    filename=file_path.name, char_count=len(text),
                )
            except DocumentLoadError as exc:
                # Log and continue — one bad file shouldn't kill the whole ingest
                log_event("Loader", "document_load_failed", filename=file_path.name, error=str(exc))
                continue

    if not documents:
        raise DocumentLoadError(f"No valid documents found in {kb_dir}")

    return documents