"""Embedding model and FAISS vector store utilities for LexAI."""

from typing import List

import streamlit as st
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@st.cache_resource(show_spinner=False)
def get_embeddings_model() -> HuggingFaceEmbeddings:
    """
    Load and cache the HuggingFace sentence-transformer embeddings model.
    This runs locally — no API key required.

    Returns:
        Cached HuggingFaceEmbeddings instance.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_vector_store(chunks: List[str]) -> FAISS:
    """
    Create an in-memory FAISS vector store from text chunks.

    Args:
        chunks: List of text chunks to embed and index.

    Returns:
        FAISS vector store.
    """
    embeddings = get_embeddings_model()
    documents = [
        Document(
            page_content=chunk,
            metadata={"chunk_id": i, "source": f"chunk_{i}"},
        )
        for i, chunk in enumerate(chunks)
    ]
    return FAISS.from_documents(documents, embeddings)


def similarity_search(
    vector_store: FAISS,
    query: str,
    k: int = 5,
) -> List[Document]:
    """
    Retrieve top-k most relevant chunks for a query.

    Args:
        vector_store: FAISS vector store.
        query: Search query string.
        k: Number of results to return.

    Returns:
        List of relevant Document objects.
    """
    return vector_store.similarity_search(query, k=k)


def similarity_search_with_score(
    vector_store: FAISS,
    query: str,
    k: int = 5,
) -> List[tuple]:
    """
    Retrieve top-k chunks with relevance scores.

    Args:
        vector_store: FAISS vector store.
        query: Search query string.
        k: Number of results to return.

    Returns:
        List of (Document, score) tuples.
    """
    return vector_store.similarity_search_with_score(query, k=k)
