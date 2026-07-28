import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.models import Document, Citation
from app.services.retriever_service import RetrieverService

@pytest.fixture
def mock_vector_store():
    store = AsyncMock()
    store.search.return_value = [
        {"text": "Chunk 1", "distance": 0.2, "metadata": {"document_id": "doc1", "page_number": 1}},
        {"text": "Chunk 2", "distance": 0.8, "metadata": {"document_id": "doc1", "page_number": 2}}, # Should be filtered out
    ]
    return store

@pytest.fixture
def mock_metadata_store():
    store = AsyncMock()
    doc = Document(id="doc1", filename="test.pdf", original_name="test.pdf", file_size=100)
    store.get_document.return_value = doc
    return store

@pytest.fixture
def mock_embedder():
    embedder = AsyncMock()
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    return embedder

@pytest.mark.asyncio
async def test_retriever_filters_by_threshold(mock_vector_store, mock_metadata_store, mock_embedder):
    retriever = RetrieverService(
        vector_store=mock_vector_store,
        metadata_store=mock_metadata_store,
        embedder=mock_embedder,
        similarity_threshold=0.5
    )

    citations = await retriever.retrieve("test query")
    
    # Should only return Chunk 1 because Chunk 2's distance is 0.8 > 0.5
    assert len(citations) == 1
    assert citations[0].text == "Chunk 1"
    assert citations[0].document_id == "doc1"
    assert citations[0].document_name == "test.pdf"
    assert citations[0].score == 0.8  # 1.0 - 0.2
