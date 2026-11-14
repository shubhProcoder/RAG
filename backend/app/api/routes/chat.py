"""
Chat API Routes — handling conversations, streaming SSE, and search.
"""

from __future__ import annotations

import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import StreamingResponse

from app.core.models import ChatRequest, Conversation
from app.dependencies import get_chat_service, get_conversation_store
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["chat"])


@router.post("/chat")
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    """
    Stream a chat response via Server-Sent Events (SSE).
    The first event will contain citations metadata, the conversation id and the Tutor State.
    Subsequent events contain text chunks.
    """
    async def sse_generator() -> AsyncGenerator[str, None]:
        try:
            async for chunk, citations, meta in chat_service.chat_stream(request):
                # Build an SSE payload
                data = {
                    "text": chunk,
                }
                if citations:
                    data["citations"] = [c.model_dump() for c in citations]
                if meta:
                    data["topic"] = meta["topic"]
                    data["tutor_state"] = meta["tutor_state"].model_dump(mode="json")

                yield f"data: {json.dumps(data)}\n\n"
            
            yield "event: done\ndata: {}\n\n"
        except Exception as e:
            logger.exception("Error in chat streaming")
            yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable NGINX buffering
        },
    )


@router.get("/conversations", response_model=list[Conversation])
async def list_conversations(
    limit: int = Query(50, ge=1),
    offset: int = Query(0, ge=0),
    store=Depends(get_conversation_store),
):
    """List all previous conversations."""
    convs, _ = await store.list_conversations(limit=limit, offset=offset)
    return convs


@router.post("/conversations", response_model=Conversation)
async def create_conversation(
    title: str = Query("New Conversation"),
    store=Depends(get_conversation_store),
):
    """Create a new, empty conversation."""
    conv = await store.create_conversation(title=title)
    return conv


@router.get("/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(
    conversation_id: str,
    store=Depends(get_conversation_store),
):
    """Retrieve full conversation history."""
    conv = await store.get_conversation(conversation_id)
    if not conv:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.get("/search")
async def search(
    query: str,
    top_k: int = Query(5, ge=1, le=20),
    document_ids: list[str] = Query(None),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Direct semantic search endpoint."""
    citations = await chat_service._retriever.retrieve(
        query=query,
        top_k=top_k,
        document_ids=document_ids,
    )
    return {"results": [c.model_dump() for c in citations]}
