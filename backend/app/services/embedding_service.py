"""
Embedding service — generates vector embeddings using
``sentence-transformers``.

The model is loaded once at construction time and reused across requests.
Embedding is done in configurable batches for memory efficiency.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.interfaces import EmbeddingProvider

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Generate embeddings using a ``sentence-transformers`` model."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
    ) -> None:
        logger.info("Loading embedding model '%s' …", model_name)
        self._model = SentenceTransformer(model_name)
        self._batch_size = batch_size
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model ready (dim=%d, batch_size=%d)",
            self._dimension,
            self._batch_size,
        )

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            vectors = self._model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            # Convert numpy to plain list[float] for ChromaDB compatibility
            for vec in vectors:
                all_embeddings.append(vec.tolist())

        logger.debug("Generated %d embeddings", len(all_embeddings))
        return all_embeddings
