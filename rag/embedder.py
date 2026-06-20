"""
Embedding generation using a local sentence-transformers model.

Why local embeddings instead of calling OpenRouter for embeddings:
- Free OpenRouter models are chat-completion models, not embedding
  endpoints — using them for embeddings would be a misuse of the API.
- Local embeddings (all-MiniLM-L6-v2, ~80MB) are fast, free, deterministic,
  and don't introduce an external dependency into the retrieval critical
  path. This is a tradeoff we explicitly document in the README.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

from config.settings import settings
from utils.exceptions import EmbeddingError
from utils.logger import trace_step, log_event

_model_cache: dict = {}


def _get_model() -> SentenceTransformer:
    """Lazily loads and caches the embedding model (loaded once per process)."""
    model_name = settings.embedding_model
    if model_name not in _model_cache:
        with trace_step("Embedder", "load_model", model_name=model_name):
            _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embeds a batch of texts. Returns an (N, dim) float32 numpy array."""
    if not texts:
        raise EmbeddingError("No texts provided for embedding")

    try:
        with trace_step("Embedder", "embed_texts", count=len(texts)):
            model = _get_model()
            embeddings = model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )
        return embeddings.astype(np.float32)
    except Exception as exc:
        raise EmbeddingError(f"Failed to embed {len(texts)} text(s): {exc}") from exc


def embed_query(query: str) -> np.ndarray:
    """Embeds a single query string. Returns a (dim,) float32 numpy array."""
    return embed_texts([query])[0]