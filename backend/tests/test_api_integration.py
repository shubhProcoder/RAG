"""
Integration tests for API endpoints.
"""

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.fixture
async def client():
    """Async test client using httpx."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Health endpoint should return 200 with status."""
    response = await client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "components" in data


@pytest.mark.asyncio
async def test_list_documents_empty(client):
    """Document list should return empty list initially."""
    response = await client.get("/api/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)


@pytest.mark.asyncio
async def test_upload_validation_rejects_non_pdf(client):
    """Upload should reject non-PDF files."""
    response = await client.post(
        "/api/documents/upload",
        files={"files": ("test.txt", b"not a pdf", "text/plain")}
    )
    # Should return validation error (422)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_get_nonexistent_document(client):
    """Getting a non-existent document should return 404."""
    response = await client.get("/api/documents/nonexistent123")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_document(client):
    """Deleting a non-existent document should return 404."""
    response = await client.delete("/api/documents/nonexistent123")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_search_endpoint(client):
    """Search endpoint should return results array."""
    response = await client.get("/api/search", params={"query": "test"})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
