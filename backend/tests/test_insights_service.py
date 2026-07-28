"""
Tests for InsightsService — insight generation and caching.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.core.models import DocumentInsight, InsightType, Document


class MockLLMProvider:
    async def generate_stream(self, messages, context):
        yield "This is a test insight."


class MockVectorStoreWithChunks:
    def get_chunks_by_document(self, doc_id, limit=30):
        return [
            {"text": "This document talks about revenue growth of 15%.", "metadata": {"chunk_index": 0}},
            {"text": "Key risks include regulatory changes.", "metadata": {"chunk_index": 1}},
            {"text": "Action item: Schedule Q3 review meeting.", "metadata": {"chunk_index": 2}},
        ]


@pytest.fixture
def mock_metadata_store():
    store = AsyncMock()
    store.get_insight.return_value = None  # No cached insight
    store.save_insight = AsyncMock()
    store.get_document.return_value = Document(
        id="doc1", filename="test.pdf", original_name="test.pdf", file_size=100
    )
    return store


@pytest.fixture
def insights_service(mock_metadata_store):
    from app.services.insights_service import InsightsService

    return InsightsService(
        metadata_store=mock_metadata_store,
        llm=MockLLMProvider(),
        vector_store=MockVectorStoreWithChunks(),
    )


@pytest.mark.asyncio
async def test_generate_insight_when_no_cache(insights_service, mock_metadata_store):
    """When no cached insight exists, it should generate via LLM and cache."""
    result = await insights_service.get_or_generate_insight("doc1", InsightType.EXECUTIVE_SUMMARY)

    assert result.content == "This is a test insight."
    assert result.insight_type == InsightType.EXECUTIVE_SUMMARY
    assert result.document_id == "doc1"
    mock_metadata_store.save_insight.assert_called_once()


@pytest.mark.asyncio
async def test_return_cached_insight(mock_metadata_store):
    """When a cached insight exists, it should return it without calling LLM."""
    from app.services.insights_service import InsightsService

    cached = DocumentInsight(
        document_id="doc1",
        insight_type=InsightType.KEY_INSIGHTS,
        content="Cached insight content",
    )
    mock_metadata_store.get_insight.return_value = cached

    service = InsightsService(
        metadata_store=mock_metadata_store,
        llm=MockLLMProvider(),
        vector_store=MockVectorStoreWithChunks(),
    )

    result = await service.get_or_generate_insight("doc1", InsightType.KEY_INSIGHTS)

    assert result.content == "Cached insight content"
    # save_insight should NOT be called because we returned cached
    mock_metadata_store.save_insight.assert_not_called()


@pytest.mark.asyncio
async def test_compare_documents(mock_metadata_store):
    """Compare documents should return comparison text."""
    from app.services.insights_service import InsightsService

    service = InsightsService(
        metadata_store=mock_metadata_store,
        llm=MockLLMProvider(),
        vector_store=MockVectorStoreWithChunks(),
    )

    result = await service.compare_documents(["doc1", "doc2"], "Compare the documents.")

    assert isinstance(result, str)
    assert len(result) > 0
