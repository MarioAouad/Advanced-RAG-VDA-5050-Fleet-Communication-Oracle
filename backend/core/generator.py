from __future__ import annotations
import logging
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

from backend.core.config import (
    LLM_MAX_TOKENS,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    GROQ_API_KEY,
)
from backend.core.retriever import get_retriever

logger = logging.getLogger(__name__)

# System prompt
_SYSTEM_PROMPT = """\
You are the VDA-5050 Fleet Communication Oracle, an expert assistant on the \
VDA 5050 standard for communication between fleet control systems and \
autonomous mobile robots (AMRs).

Answer the user's question using ONLY the provided context below. \
If the context does not contain enough information to answer, say so clearly.

Rules:
- Be precise and cite specific section numbers, field names, or enum values \
  from the VDA 5050 specification when applicable.
- If the question relates to a JSON schema, refer to the exact required \
  fields and data types.
- Keep your answer concise yet thorough.

Context:
{context}
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM_PROMPT),
    ("human", "{question}")
])

def _get_llm() -> ChatGroq:
    return ChatGroq(
        model=LLM_MODEL_NAME,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        api_key=GROQ_API_KEY,
    )

def _format_docs(docs: list[Document]) -> str:
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# Public API
def query_rag(question: str) -> dict[str, str | list[str]]:

    # Retrieve (single call — hybrid + reranked)
    retriever = get_retriever()
    retrieved_docs: list[Document] = retriever.invoke(question)
    context_strings: list[str] = [doc.page_content for doc in retrieved_docs]
    source_strings: list[str] = [doc.metadata.get("source", "") for doc in retrieved_docs]

    logger.info("Retrieved %d chunks for question: %.60s...", len(retrieved_docs), question)

    # Build context and generate (reuse already-retrieved docs)
    llm = _get_llm()
    formatted_context = _format_docs(retrieved_docs)
    prompt_value = _PROMPT.invoke({"context": formatted_context, "question": question})
    answer: str = (llm | StrOutputParser()).invoke(prompt_value)

    logger.info("Generated answer (%d chars).", len(answer))

    return {
        "question": question,
        "answer": answer,
        "contexts": context_strings,
        "sources": source_strings,
    }
