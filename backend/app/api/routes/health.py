"""
Health check endpoint — enhanced with LLM and insights checks.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.config import settings
from app.core.models import HealthResponse

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return system health status."""
    from app.dependencies import (
        get_metadata_store,
        get_vector_store,
        get_embedder,
    )

    components: dict = {}

    # Check metadata store
    try:
        meta_store = get_metadata_store()
        docs, total = await meta_store.list_documents(limit=1, offset=0)
        components["metadata_store"] = {
            "status": "healthy",
            "total_documents": total,
        }
    except Exception as exc:
        components["metadata_store"] = f"unhealthy: {exc}"

    # Check vector store
    try:
        vec_store = get_vector_store()
        components["vector_store"] = {
            "status": "healthy",
            "total_chunks": vec_store.count(),
        }
    except Exception as exc:
        components["vector_store"] = f"unhealthy: {exc}"

    # Check embedding model
    try:
        embedder = get_embedder()
        components["embedding_model"] = {
            "status": "healthy",
            "model": settings.EMBEDDING_MODEL,
            "dimension": embedder.dimension,
        }
    except Exception as exc:
        components["embedding_model"] = f"unhealthy: {exc}"

    # Check LLM connectivity
    try:
        llm_configured = bool(settings.OPENAI_API_KEY)
        components["llm"] = {
            "status": "healthy" if llm_configured else "unconfigured",
            "model": settings.LLM_MODEL,
        }
    except Exception as exc:
        components["llm"] = f"unhealthy: {exc}"

    overall = "healthy" if all(
        (isinstance(v, dict) and v.get("status") in ("healthy", "unconfigured")) or v == "healthy"
        for v in components.values()
    ) else "degraded"

    return HealthResponse(
        status=overall,
        version=settings.APP_VERSION,
        components=components,
    )
