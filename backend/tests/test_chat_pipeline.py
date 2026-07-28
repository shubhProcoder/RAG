import pytest
from unittest.mock import AsyncMock

from app.core.models import ChatRequest, Conversation, ChatMessage
from app.services.chat_service import ChatService
from app.services.prompt_builder import PromptBuilder

@pytest.fixture
def mock_retriever():
    retriever = AsyncMock()
    retriever.retrieve.return_value = []
    return retriever

@pytest.fixture
def mock_conversation_store():
    store = AsyncMock()
    store.create_conversation.return_value = Conversation(id="conv1", title="New Conv")
    store.get_conversation.return_value = None
    store.add_message = AsyncMock()
    return store

class MockLLMProvider:
    async def generate_stream(self, messages, context):
        # Simply echo back the system message to verify PromptBuilder behavior
        system_content = messages[0].content
        if "No relevant documents found" in system_content:
            yield "Information not found in uploaded documents."
        else:
            yield "Here is the answer."

@pytest.mark.asyncio
async def test_chat_pipeline_fallback(mock_retriever, mock_conversation_store):
    prompt_builder = PromptBuilder()
    llm = MockLLMProvider()
    
    chat_service = ChatService(
        retriever=mock_retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        conversation_store=mock_conversation_store
    )

    req = ChatRequest(message="What is the capital of France?")
    
    chunks = []
    citations = None
    
    async for chunk, cites in chat_service.chat_stream(req):
        chunks.append(chunk)
        if cites is not None and citations is None:
            citations = cites

    full_response = "".join(chunks)
    
    # Verify fallback message was triggered due to empty retrieval
    assert full_response == "Information not found in uploaded documents."
    assert citations == []
