"""
ChromaDB-backed vector store.

Persists chunk embeddings with metadata (document_id, page_number, chunk_index)
for similarity search.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.interfaces import VectorStore
from app.core.models import Chunk

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """Concrete ``VectorStore`` backed by a persistent ChromaDB instance."""

    COLLECTION_NAME = "pdf_chunks"

    def __init__(self, persist_directory: str) -> None:
        self._client = chromadb.PersistentClient(
            path=persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB vector store ready at %s (collection=%s, count=%d)",
            persist_directory,
            self.COLLECTION_NAME,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    async def add(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        if not chunks:
            return

        ids = [c.id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {
                "document_id": c.document_id,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "token_count": c.token_count,
            }
            for c in chunks
        ]

        # ChromaDB supports batch upsert
        self._collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Stored %d chunks in vector store", len(chunks))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    async def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        where = filter_metadata if filter_metadata else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Flatten ChromaDB's nested list structure
        items: list[dict[str, Any]] = []
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"][0]):
                items.append(
                    {
                        "id": chunk_id,
                        "text": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else None,
                    }
                )
        return items

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    async def delete_by_document(self, document_id: str) -> None:
        # ChromaDB requires fetching IDs that match a where filter, then deleting
        try:
            results = self._collection.get(
                where={"document_id": document_id},
                include=[],
            )
            if results and results["ids"]:
                self._collection.delete(ids=results["ids"])
                logger.info(
                    "Deleted %d chunks for document %s",
                    len(results["ids"]),
                    document_id,
                )
        except Exception as exc:
            logger.warning(
                "Failed to delete chunks for document %s: %s", document_id, exc
            )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return total number of stored chunks."""
        return self._collection.count()

    def get_chunks_by_document(
        self, document_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Retrieve all chunks for a given document (for inspection)."""
        results = self._collection.get(
            where={"document_id": document_id},
            include=["documents", "metadatas"],
            limit=limit,
        )
        items: list[dict[str, Any]] = []
        if results and results["ids"]:
            for i, chunk_id in enumerate(results["ids"]):
                items.append(
                    {
                        "id": chunk_id,
                        "text": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    }
                )
        return items