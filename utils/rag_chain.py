"""LangChain RAG pipeline for LexAI."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from typing import List, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq


# ── Prompt Templates ─────────────────────────────────────────────────────────

LEGAL_EXPERT_SYSTEM = """You are LexAI, a senior legal analyst with expertise in contract law, 
corporate law, employment law, and compliance. You have 20+ years of experience reviewing 
legal documents across multiple jurisdictions.

Your role:
- Analyze legal documents with precision and objectivity
- Identify risks, obligations, and key provisions
- Explain complex legal language in clear, accessible terms
- Always ground your analysis in the provided document context
- If information is absent from the context, explicitly state that

Document Context:
{context}"""

CHAT_SYSTEM = """You are LexAI, an expert legal document assistant. Answer questions about 
the uploaded legal document based strictly on the provided context. 

Rules:
- Only use information from the document context below
- If asked about something not in the document, say "I cannot find that information in this document"
- Be precise and cite relevant sections when possible
- Keep answers clear and actionable

Document Context:
{context}"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_docs(docs: List[Document]) -> str:
    """Format a list of documents into a single context string."""
    return "\n\n---\n\n".join(
        f"[Excerpt {i + 1}]:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )


# ── RAG Chain ─────────────────────────────────────────────────────────────────

def build_rag_chain(llm: ChatGroq, vector_store: FAISS):
    """
    Build a streaming-capable RAG chain for document Q&A.

    Args:
        llm: ChatGroq LLM instance.
        vector_store: FAISS vector store with document chunks.

    Returns:
        LangChain LCEL chain.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages([
        ("system", CHAT_SYSTEM),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def query_with_sources(
    llm: ChatGroq,
    vector_store: FAISS,
    query: str,
    k: int = 5,
) -> Tuple[str, List[Document]]:
    """
    Run a RAG query and return both the answer and source documents.

    Args:
        llm: ChatGroq LLM instance.
        vector_store: FAISS vector store.
        query: User question.
        k: Number of context chunks to retrieve.

    Returns:
        Tuple of (answer_string, source_documents).
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(query)
    context = _format_docs(docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", LEGAL_EXPERT_SYSTEM),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": query})
    return answer, docs


def analyze_document(
    llm,
    text: str,
    analysis_prompt: str,
) -> str:

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are LexAI, an expert legal analyst. Analyze the provided legal document text and respond according to the specific instructions given.",
        ),
        (
            "human",
            """DOCUMENT TEXT:

{text}

---

INSTRUCTION:

{analysis_prompt}
""",
        ),
    ])

    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "text": text[:8000],
        "analysis_prompt": analysis_prompt,
    })
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({})
