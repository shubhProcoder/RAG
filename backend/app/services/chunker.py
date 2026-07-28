"""
Recursive text chunker with page-number preservation.

Splits on paragraph → sentence → word boundaries to keep chunks
semantically coherent.  Overlap is applied between adjacent chunks
to avoid losing context at boundaries.
"""

from __future__ import annotations

import re
import logging
from typing import Any

from app.core.interfaces import TextChunker
from app.core.models import Chunk

logger = logging.getLogger(__name__)

# Ordered from coarsest to finest boundary
_SEPARATORS: list[str] = [
    "\n\n",   # paragraph break
    "\n",     # line break
    ". ",     # sentence boundary
    "? ",
    "! ",
    "; ",
    ", ",
    " ",      # word boundary
    "",        # character level (last resort)
]


class RecursiveChunker(TextChunker):
    """Split text recursively using a hierarchy of separators."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._overlap = chunk_overlap

    def chunk(
        self,
        text: str,
        document_id: str,
        page_number: int,
        start_index: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not text.strip():
            return []

        raw_chunks = self._split_recursive(text, _SEPARATORS)
        chunks: list[Chunk] = []

        for i, chunk_text in enumerate(raw_chunks):
            chunks.append(
                Chunk(
                    document_id=document_id,
                    text=chunk_text,
                    page_number=page_number,
                    chunk_index=start_index + i,
                    token_count=self._estimate_tokens(chunk_text),
                    metadata=metadata or {},
                )
            )

        return chunks

    # ------------------------------------------------------------------
    # Core recursive splitting
    # ------------------------------------------------------------------

    def _split_recursive(
        self, text: str, separators: list[str]
    ) -> list[str]:
        """Split *text* using the first separator that produces segments,
        then recurse on segments that are still too long."""

        if len(text) <= self._chunk_size:
            return [text.strip()] if text.strip() else []

        # Find the best separator for this text
        separator = separators[-1]  # fallback: character-level
        for sep in separators:
            if sep and sep in text:
                separator = sep
                break

        # Split and recombine with overlap
        parts = text.split(separator) if separator else list(text)
        remaining_seps = (
            separators[separators.index(separator) + 1:]
            if separator in separators and separators.index(separator) + 1 < len(separators)
            else [separators[-1]]
        )

        merged = self._merge_splits(parts, separator)
        final: list[str] = []

        for segment in merged:
            if len(segment) <= self._chunk_size:
                if segment.strip():
                    final.append(segment.strip())
            else:
                # Recurse with finer separators
                final.extend(self._split_recursive(segment, remaining_seps))

        return final

    def _merge_splits(
        self, splits: list[str], separator: str
    ) -> list[str]:
        """Merge small splits together up to ``chunk_size``, applying overlap."""

        merged: list[str] = []
        current: list[str] = []
        current_len = 0

        for part in splits:
            part_len = len(part)

            if current_len + part_len + (len(separator) if current else 0) > self._chunk_size:
                if current:
                    merged.append(separator.join(current))

                    # Apply overlap: keep trailing parts that fit within overlap budget
                    overlap_parts: list[str] = []
                    overlap_len = 0
                    for prev in reversed(current):
                        if overlap_len + len(prev) + len(separator) > self._overlap:
                            break
                        overlap_parts.insert(0, prev)
                        overlap_len += len(prev) + len(separator)

                    current = overlap_parts
                    current_len = overlap_len
                else:
                    current = []
                    current_len = 0

            current.append(part)
            current_len += part_len + (len(separator) if len(current) > 1 else 0)

        if current:
            merged.append(separator.join(current))

        return merged

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 characters per token (English)."""
        return max(1, len(text) // 4)
