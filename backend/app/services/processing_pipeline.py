"""
Processing pipeline — orchestrates the full ingestion flow:

    extract → chunk → embed → store

Each stage updates the document status so the frontend can
display real-time progress.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.exceptions import ProcessingError
from app.core.interfaces import (
    EmbeddingProvider,
    MetadataStore,
    PDFExtractor,
    TextChunker,
    VectorStore,
)
from app.core.models import Chunk, ProcessingStatus

logger = logging.getLogger(__name__)


class ProcessingPipeline:
    """End-to-end document ingestion pipeline."""

    def __init__(
        self,
        extractor: PDFExtractor,
        chunker: TextChunker,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
    ) -> None:
        self._extractor = extractor
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._metadata_store = metadata_store

    async def process(self, document_id: str, file_path: str) -> None:
        """Run the full pipeline for a single document.

        Raises ``ProcessingError`` on unrecoverable failures; the
        document status is set to FAILED automatically.
        """
        try:
            # ── 1. Extract ───────────────────────────────────────────
            await self._metadata_store.update_status(
                document_id, ProcessingStatus.EXTRACTING
            )
            logger.info("[%s] Extracting text …", document_id)

            extraction = await self._extractor.extract(file_path)
            await self._metadata_store.update_status(
                document_id,
                ProcessingStatus.EXTRACTING,
                page_count=extraction.total_pages,
            )
            logger.info(
                "[%s] Extracted %d pages (%d tables)",
                document_id,
                extraction.total_pages,
                len(extraction.tables),
            )

            # ── 2. Chunk ────────────────────────────────────────────
            await self._metadata_store.update_status(
                document_id, ProcessingStatus.CHUNKING
            )

            all_chunks: list[Chunk] = []
            chunk_index = 0

            # Chunk page text
            for page in extraction.pages:
                page_chunks = self._chunker.chunk(
                    text=page.text,
                    document_id=document_id,
                    page_number=page.page_number,
                    start_index=chunk_index,
                    metadata={
                        "extraction_method": page.extraction_method.value,
                    },
                )
                all_chunks.extend(page_chunks)
                chunk_index += len(page_chunks)

            # Chunk tables (as markdown text)
            for table in extraction.tables:
                table_chunks = self._chunker.chunk(
                    text=table.markdown,
                    document_id=document_id,
                    page_number=table.page_number,
                    start_index=chunk_index,
                    metadata={"content_type": "table"},
                )
                all_chunks.extend(table_chunks)
                chunk_index += len(table_chunks)

            logger.info("[%s] Created %d chunks", document_id, len(all_chunks))

            if not all_chunks:
                await self._metadata_store.update_status(
                    document_id,
                    ProcessingStatus.COMPLETED,
                    chunk_count=0,
                )
                logger.warning("[%s] No chunks produced — nothing to embed", document_id)
                return

            # ── 3. Embed ────────────────────────────────────────────
            await self._metadata_store.update_status(
                document_id, ProcessingStatus.EMBEDDING
            )

            texts = [c.text for c in all_chunks]
            embeddings = await self._embedder.embed(texts)

            logger.info("[%s] Generated %d embeddings", document_id, len(embeddings))

            # ── 4. Store ────────────────────────────────────────────
            await self._metadata_store.update_status(
                document_id, ProcessingStatus.STORING
            )

            await self._vector_store.add(all_chunks, embeddings)

            # ── 5. Done ─────────────────────────────────────────────
            await self._metadata_store.update_status(
                document_id,
                ProcessingStatus.COMPLETED,
                chunk_count=len(all_chunks),
                page_count=extraction.total_pages,
            )
            logger.info(
                "[%s] Processing complete — %d pages, %d chunks",
                document_id,
                extraction.total_pages,
                len(all_chunks),
            )

        except ProcessingError:
            raise
        except Exception as exc:
            error_msg = str(exc)
            await self._metadata_store.update_status(
                document_id,
                ProcessingStatus.FAILED,
                error_message=error_msg,
            )
            logger.exception("[%s] Processing failed", document_id)
            raise ProcessingError(document_id, "pipeline", error_msg) from exc
