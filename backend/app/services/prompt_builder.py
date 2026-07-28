"""
Prompt Builder — constructs prompts for the LLM using context and instructions.
"""

from __future__ import annotations

from app.core.models import ChatMessage, Citation


class PromptBuilder:
    """Builds LLM prompts strictly adhering to RAG principles."""

    SYSTEM_PROMPT_TEMPLATE = """You are a helpful and precise assistant. 
You will be provided with context from uploaded documents to answer the user's question.

Guidelines:
1. Base your answer ONLY on the provided context.
2. If the context does not contain the information needed to answer the question, you must reply EXACTLY with: "Information not found in uploaded documents."
3. Do not use outside knowledge.
4. If you use information from the context, you can cite the document name and page number.

Context Information:
--------------------
{context}
--------------------
"""

    def build_messages(
        self, history: list[ChatMessage], query: str, citations: list[Citation]
    ) -> list[ChatMessage]:
        """Combine system instructions, context, history, and the new query."""
        
        # Build context block
        if citations:
            context_blocks = []
            for i, cite in enumerate(citations, 1):
                context_blocks.append(
                    f"[Source {i}: {cite.document_name}, Page {cite.page_number}]\n{cite.text}"
                )
            context_str = "\n\n".join(context_blocks)
        else:
            context_str = "No relevant documents found for this query."

        # Format system message
        system_msg = ChatMessage(
            role="system",
            content=self.SYSTEM_PROMPT_TEMPLATE.format(context=context_str),
        )

        # Build final message list
        final_messages = [system_msg]
        
        # Add conversation history
        for msg in history:
            final_messages.append(msg)
            
        # Add the new user query
        final_messages.append(ChatMessage(role="user", content=query))

        return final_messages
