"""
Retriever service — encapsulates vector search logic, filtering, and scoring.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.interfaces import MetadataStore, VectorStore, EmbeddingProvider
from app.core.models import Citation

logger = logging.getLogger(__name__)


class RetrieverService:
    """Handles semantic search and context building."""

    def __init__(
        self,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        embedder: EmbeddingProvider,
        similarity_threshold: float = 0.5,
    ) -> None:
        self._vector_store = vector_store
        self._metadata_store = metadata_store
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold

    async def retrieve(
        self, query: str, top_k: int = 5, document_ids: list[str] | None = None
    ) -> list[Citation]:
        """Search for relevant chunks and build rich citations."""
        logger.info("Searching for top %d chunks for query", top_k)
        
        # 1. Embed query
        query_embeddings = await self._embedder.embed([query])
        if not query_embeddings:
            return []
        query_vector = query_embeddings[0]

        # 2. Setup filters
        filter_metadata = None
        if document_ids:
            if len(document_ids) == 1:
                filter_metadata = {"document_id": document_ids[0]}
            else:
                filter_metadata = {"document_id": {"$in": document_ids}}

        # 3. Search
        results = await self._vector_store.search(
            query_embedding=query_vector,
            k=top_k,
            filter_metadata=filter_metadata,
        )

        if not results:
            return []

        # 4. Filter by threshold and build citations
        citations: list[Citation] = []
        
        # We need to map document_id -> document_name to avoid repeated DB calls
        doc_names: dict[str, str] = {}

        for item in results:
            distance = item.get("distance", 0.0)
            if distance > self._similarity_threshold:
                logger.debug("Skipping chunk due to high distance: %f", distance)
                continue

            metadata = item.get("metadata", {})
            doc_id = metadata.get("document_id", "unknown")
            page_num = metadata.get("page_number", 0)

            # Resolve document name
            if doc_id not in doc_names:
                doc = await self._metadata_store.get_document(doc_id)
                doc_names[doc_id] = doc.original_name if doc else f"Document {doc_id}"

            citations.append(
                Citation(
                    document_id=doc_id,
                    document_name=doc_names[doc_id],
                    page_number=page_num,
                    text=item.get("text", ""),
                    score=1.0 - distance,  # simple inversion for cosine distance
                )
            )

        logger.info("Retrieved %d valid chunks", len(citations))
        return citations
