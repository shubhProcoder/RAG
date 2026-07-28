"""
Chat Service — Orchestrates Retrieval, Prompt Building, and LLM Generation.
"""

from __future__ import annotations

import logging
import typing

from app.core.interfaces import ConversationStore, LLMProvider
from app.core.models import ChatMessage, ChatRequest, Citation
from app.services.prompt_builder import PromptBuilder
from app.services.retriever_service import RetrieverService

logger = logging.getLogger(__name__)


class ChatService:
    """Orchestrates the RAG pipeline for chat interactions."""

    def __init__(
        self,
        retriever: RetrieverService,
        prompt_builder: PromptBuilder,
        llm: LLMProvider,
        conversation_store: ConversationStore,
    ) -> None:
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._conversation_store = conversation_store

    async def chat_stream(
        self, request: ChatRequest
    ) -> typing.AsyncGenerator[tuple[str, list[Citation]], None]:
        """
        Executes the RAG pipeline and yields text chunks.
        The very first yield includes the full list of citations so the frontend can display them.
        Subsequent yields only contain text chunks.
        """
        # 1. Retrieve or create conversation
        if request.conversation_id:
            conv = await self._conversation_store.get_conversation(request.conversation_id)
            if not conv:
                conv = await self._conversation_store.create_conversation()
        else:
            conv = await self._conversation_store.create_conversation()

        history = conv.messages[-10:] if conv else [] # Keep last 10 messages for context

        # 2. Retrieve context
        citations = await self._retriever.retrieve(
            query=request.message,
            document_ids=request.document_ids,
        )

        # 3. Build prompt
        messages = self._prompt_builder.build_messages(
            history=history,
            query=request.message,
            citations=citations,
        )

        # Save user message
        user_msg = ChatMessage(role="user", content=request.message)
        await self._conversation_store.add_message(conv.id, user_msg)

        # 4. Stream response
        full_response = ""
        first_chunk = True
        
        async for chunk in self._llm.generate_stream(messages, context=""):
            full_response += chunk
            if first_chunk:
                yield (chunk, citations)
                first_chunk = False
            else:
                yield (chunk, [])

        if first_chunk:
            # If no chunks were yielded at all (e.g., error)
            yield ("", citations)

        # Save assistant message
        assistant_msg = ChatMessage(role="assistant", content=full_response)
        await self._conversation_store.add_message(conv.id, assistant_msg)
