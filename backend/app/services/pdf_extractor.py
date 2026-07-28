"""
PDF text extractor — PyMuPDF (primary) with Tesseract OCR fallback
and pdfplumber table extraction.

Gracefully degrades if Tesseract is not installed.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import pdfplumber

from app.config import settings
from app.core.exceptions import ExtractionError
from app.core.interfaces import PDFExtractor
from app.core.models import (
    ExtractionMethod,
    ExtractionResult,
    PageContent,
    TableContent,
)

logger = logging.getLogger(__name__)

# Lazy-check Tesseract availability
_TESSERACT_AVAILABLE: bool | None = None


def _check_tesseract() -> bool:
    global _TESSERACT_AVAILABLE
    if _TESSERACT_AVAILABLE is not None:
        return _TESSERACT_AVAILABLE
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        _TESSERACT_AVAILABLE = True
        logger.info("Tesseract OCR is available")
    except Exception:
        _TESSERACT_AVAILABLE = False
        logger.warning(
            "Tesseract OCR is NOT available — scanned PDF pages will be skipped"
        )
    return _TESSERACT_AVAILABLE


class PyMuPDFExtractor(PDFExtractor):
    """Extract text from PDFs using PyMuPDF with optional OCR fallback."""

    async def extract(self, file_path: str) -> ExtractionResult:
        path = Path(file_path)
        if not path.exists():
            raise ExtractionError(path.name, "File does not exist")

        try:
            pages = self._extract_text(path)
            tables = self._extract_tables(path)

            return ExtractionResult(
                pages=pages,
                tables=tables,
                total_pages=len(pages),
                metadata=self._extract_metadata(path),
            )
        except ExtractionError:
            raise
        except Exception as exc:
            raise ExtractionError(path.name, str(exc)) from exc

    # ------------------------------------------------------------------
    # Native text extraction via PyMuPDF
    # ------------------------------------------------------------------

    def _extract_text(self, path: Path) -> list[PageContent]:
        pages: list[PageContent] = []

        doc = fitz.open(str(path))
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()

                if len(text) >= settings.OCR_MIN_TEXT_LENGTH:
                    pages.append(
                        PageContent(
                            page_number=page_num + 1,
                            text=text,
                            extraction_method=ExtractionMethod.NATIVE,
                        )
                    )
                else:
                    # Attempt OCR if text is sparse
                    ocr_text = self._ocr_page(path, page_num)
                    if ocr_text:
                        pages.append(
                            PageContent(
                                page_number=page_num + 1,
                                text=ocr_text,
                                extraction_method=ExtractionMethod.OCR,
                            )
                        )
                    elif text:
                        # Fall back to whatever we got natively
                        pages.append(
                            PageContent(
                                page_number=page_num + 1,
                                text=text,
                                extraction_method=ExtractionMethod.NATIVE,
                            )
                        )
                    else:
                        logger.warning(
                            "Page %d of '%s' has no extractable text",
                            page_num + 1,
                            path.name,
                        )
        finally:
            doc.close()

        return pages

    # ------------------------------------------------------------------
    # OCR fallback via Tesseract
    # ------------------------------------------------------------------

    def _ocr_page(self, path: Path, page_index: int) -> str | None:
        if not settings.OCR_ENABLED or not _check_tesseract():
            return None

        try:
            from pdf2image import convert_from_path
            import pytesseract

            images = convert_from_path(
                str(path),
                first_page=page_index + 1,
                last_page=page_index + 1,
                dpi=300,
            )
            if not images:
                return None

            text = pytesseract.image_to_string(
                images[0], lang=settings.OCR_LANGUAGE
            ).strip()
            return text if text else None
        except Exception as exc:
            logger.warning("OCR failed for page %d: %s", page_index + 1, exc)
            return None

    # ------------------------------------------------------------------
    # Table extraction via pdfplumber
    # ------------------------------------------------------------------

    def _extract_tables(self, path: Path) -> list[TableContent]:
        tables: list[TableContent] = []

        try:
            with pdfplumber.open(str(path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if not table or not table[0]:
                            continue
                        md = self._table_to_markdown(table)
                        tables.append(
                            TableContent(
                                page_number=page_num,
                                table_index=table_idx,
                                markdown=md,
                                row_count=len(table),
                                col_count=len(table[0]),
                            )
                        )
        except Exception as exc:
            logger.warning("Table extraction failed: %s", exc)

        return tables

    @staticmethod
    def _table_to_markdown(table: list[list[str | None]]) -> str:
        """Convert a pdfplumber table to Markdown format."""
        if not table:
            return ""

        def _clean(cell: str | None) -> str:
            return (cell or "").replace("\n", " ").strip()

        lines: list[str] = []
        header = [_clean(c) for c in table[0]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        for row in table[1:]:
            cells = [_clean(c) for c in row]
            # Pad or truncate to match header length
            while len(cells) < len(header):
                cells.append("")
            lines.append("| " + " | ".join(cells[: len(header)]) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # PDF metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metadata(path: Path) -> dict[str, Any]:
        doc = fitz.open(str(path))
        try:
            meta = doc.metadata or {}
            return {
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "subject": meta.get("subject", ""),
                "creator": meta.get("creator", ""),
                "producer": meta.get("producer", ""),
                "page_count": len(doc),
            }
        finally:
            doc.close()
