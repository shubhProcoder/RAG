"""
Insights Service for Document Intelligence Platform.

Generates insights using LLM and caches them in the metadata store.
"""

import logging
import json
from typing import Any

from app.core.interfaces import MetadataStore, LLMProvider, VectorStore
from app.core.models import ChatMessage, DocumentInsight, InsightType
from app.services.retriever_service import RetrieverService

logger = logging.getLogger(__name__)

class InsightsService:
    def __init__(
        self,
        metadata_store: MetadataStore,
        llm: LLMProvider,
        vector_store: VectorStore,
    ):
        self._metadata_store = metadata_store
        self._llm = llm
        self._vector_store = vector_store

    async def get_or_generate_insight(
        self, document_id: str, insight_type: InsightType
    ) -> DocumentInsight:
        """
        Retrieves a cached insight or generates a new one.
        """
        # 1. Check cache
        cached = await self._metadata_store.get_insight(document_id, insight_type)
        if cached:
            logger.info("Insight %s for %s found in cache.", insight_type.value, document_id)
            return cached

        logger.info("Generating insight %s for %s...", insight_type.value, document_id)

        # 2. Get document chunks context
        # To minimize LLM calls and avoid complex map-reduce, we fetch up to 30 chunks 
        # from the beginning of the document or relevant tables/chunks.
        chunks = []
        if hasattr(self._vector_store, "get_chunks_by_document"):
            raw_chunks = self._vector_store.get_chunks_by_document(document_id, limit=30)
            # Sort by chunk_index to keep order
            raw_chunks.sort(key=lambda c: c.get("metadata", {}).get("chunk_index", 0))
            for c in raw_chunks:
                chunks.append(c.get("text", ""))

        if not chunks:
            content = "No text available for this document to generate insights."
            return await self._save_and_return(document_id, insight_type, content)

        # Join chunks into context (truncate if too long, assuming 128k context)
        context = "\n\n".join(chunks)[:60000]

        # 3. Construct prompt based on type
        prompt = self._get_prompt_for_type(insight_type)
        
        messages = [
            ChatMessage(role="system", content="You are an expert AI Document Intelligence assistant. Provide concise, highly accurate, professional insights based ONLY on the provided document text."),
            ChatMessage(role="user", content=f"Document Text:\n{context}\n\nTask:\n{prompt}")
        ]

        # 4. Call LLM
        content_stream = self._llm.generate_stream(messages, "")
        
        full_content = ""
        try:
            async for chunk in content_stream:
                full_content += chunk
        except Exception as e:
            logger.exception("Error generating insight")
            full_content = f"Error generating insight: {e}"

        full_content = full_content.strip()

        # 5. Cache and return
        return await self._save_and_return(document_id, insight_type, full_content)

    async def _save_and_return(
        self, document_id: str, insight_type: InsightType, content: str
    ) -> DocumentInsight:
        insight = DocumentInsight(
            document_id=document_id,
            insight_type=insight_type,
            content=content,
        )
        await self._metadata_store.save_insight(insight)
        return insight

    def _get_prompt_for_type(self, insight_type: InsightType) -> str:
        prompts = {
            InsightType.EXECUTIVE_SUMMARY: "Provide a comprehensive but concise executive summary of the document. Write in professional paragraphs.",
            InsightType.KEY_INSIGHTS: "List the top 5-7 key insights, findings, or takeaways from this document. Use bullet points.",
            InsightType.ACTION_ITEMS: "Identify any action items, tasks, or next steps mentioned. If none are found, state 'No action items identified.' Use bullet points.",
            InsightType.RISKS: "Identify any potential risks, warnings, or compliance issues mentioned. If none, state 'No specific risks identified.' Use bullet points.",
            InsightType.RECOMMENDATIONS: "List any recommendations or proposals made in the document. If none, state 'No recommendations found.'",
            InsightType.KPI_METRICS: "Extract all key performance indicators (KPIs) and important metrics (e.g., percentages, growth rates). Format as a clear list.",
            InsightType.IMPORTANT_DATES: "Extract all significant dates and deadlines. Format as 'YYYY-MM-DD: Event description'.",
            InsightType.NAMED_ENTITIES: "Extract the most important People, Organizations, and Locations mentioned. Group them by category.",
            InsightType.FINANCIAL_METRICS: "Extract all financial figures, revenue, costs, or budget numbers. If none, state 'No financial metrics present.'",
            InsightType.TABLE_INSIGHTS: "Analyze any data tables present in the text and summarize the key trends or data points they convey.",
        }
        return prompts.get(insight_type, "Summarize the document.")

    async def compare_documents(self, document_ids: list[str], query: str) -> str:
        """
        Compare multiple documents based on a user query.
        """
        contexts = []
        for doc_id in document_ids:
            doc = await self._metadata_store.get_document(doc_id)
            doc_name = doc.original_name if doc else doc_id
            
            raw_chunks = []
            if hasattr(self._vector_store, "get_chunks_by_document"):
                raw_chunks = self._vector_store.get_chunks_by_document(doc_id, limit=15)
            
            text = "\n\n".join([c.get("text", "") for c in raw_chunks])[:20000]
            contexts.append(f"--- Document: {doc_name} ---\n{text}")

        full_context = "\n\n".join(contexts)

        messages = [
            ChatMessage(role="system", content="You are an expert AI Data Analyst. Compare the provided documents based on the user's query. Be objective, highlight similarities and differences, and use clear headings."),
            ChatMessage(role="user", content=f"Documents Context:\n{full_context}\n\nComparison Query:\n{query}")
        ]

        content_stream = self._llm.generate_stream(messages, "")
        full_content = ""
        try:
            async for chunk in content_stream:
                full_content += chunk
        except Exception as e:
            logger.exception("Error generating comparison")
            full_content = f"Error generating comparison: {e}"

        return full_content.strip()
