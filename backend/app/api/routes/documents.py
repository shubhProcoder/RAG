"""
Document management endpoints — upload, list, get, delete, chunk inspection.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, File, UploadFile, Query

from app.core.models import (
    ChunkResponse,
    DocumentListResponse,
    DocumentResponse,
    UploadResponse,
    InsightType,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])


def _get_document_service():
    """Lazy import to avoid circular dependency."""
    from app.dependencies import get_document_service
    return get_document_service()

def _get_insights_service():
    from app.dependencies import get_insights_service
    return get_insights_service()


# ------------------------------------------------------------------
# Upload
# ------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: list[UploadFile] = File(..., description="One or more PDF files"),
):
    """Upload one or more PDF files for processing."""
    service = _get_document_service()
    document_ids: list[str] = []

    for upload_file in files:
        content = await upload_file.read()
        doc_id = await service.upload(
            filename=upload_file.filename or "unknown.pdf",
            content=content,
        )
        document_ids.append(doc_id)

    return UploadResponse(
        document_ids=document_ids,
        message=f"Accepted {len(document_ids)} file(s) for processing",
    )


# ------------------------------------------------------------------
# List
# ------------------------------------------------------------------

@router.get("", response_model=DocumentListResponse)
async def list_documents(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """List all uploaded documents with their processing status."""
    service = _get_document_service()
    return await service.list_documents(limit=limit, offset=offset)


# ------------------------------------------------------------------
# Get
# ------------------------------------------------------------------

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """Get a single document's metadata and processing status."""
    service = _get_document_service()
    return await service.get_document(document_id)


# ------------------------------------------------------------------
# Delete
# ------------------------------------------------------------------

@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and all associated data (file, vectors, metadata)."""
    service = _get_document_service()
    await service.delete_document(document_id)
    return {"message": f"Document {document_id} deleted successfully"}


# ------------------------------------------------------------------
# Chunk inspection
# ------------------------------------------------------------------

@router.get("/{document_id}/chunks", response_model=list[ChunkResponse])
async def get_document_chunks(
    document_id: str,
    limit: int = Query(100, ge=1, le=500),
):
    """List stored chunks for a document (for debugging / inspection)."""
    service = _get_document_service()

    # Verify document exists
    await service.get_document(document_id)

    raw_chunks = service.get_chunks(document_id, limit=limit)
    return [
        ChunkResponse(
            id=c["id"],
            document_id=c.get("metadata", {}).get("document_id", document_id),
            text=c.get("text", ""),
            page_number=c.get("metadata", {}).get("page_number", 0),
            chunk_index=c.get("metadata", {}).get("chunk_index", 0),
            token_count=c.get("metadata", {}).get("token_count", 0),
            metadata=c.get("metadata", {}),
        )
        for c in raw_chunks
    ]

# ------------------------------------------------------------------
# Insights
# ------------------------------------------------------------------

@router.get("/{document_id}/insights/{insight_type}")
async def get_document_insight(document_id: str, insight_type: InsightType):
    """Get or generate a specific insight for a document."""
    service = _get_insights_service()
    doc_service = _get_document_service()
    
    # Ensure document exists
    await doc_service.get_document(document_id)
    
    insight = await service.get_or_generate_insight(document_id, insight_type)
    return insight

from pydantic import BaseModel

class CompareRequest(BaseModel):
    document_ids: list[str]
    query: str = "Compare these documents."

@router.post("/compare")
async def compare_documents(request: CompareRequest):
    """Compare multiple documents based on a query."""
    if len(request.document_ids) < 2:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="At least 2 documents are required for comparison.")
    
    service = _get_insights_service()
    result = await service.compare_documents(request.document_ids, request.query)
    return {"comparison": result}
