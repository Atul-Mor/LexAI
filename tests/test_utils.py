"""Tests for utility modules — Groq client, embeddings, RAG chain."""

import pytest
from unittest.mock import MagicMock, patch


# ── Test groq_client ─────────────────────────────────────────────────────────

class TestGroqClient:
    """Tests for Groq client utilities."""

    def test_get_groq_client_returns_instance(self):
        """Test that get_groq_client returns a ChatGroq instance."""
        from utils.groq_client import get_groq_client
        from langchain_groq import ChatGroq

        client = get_groq_client("fake_key_for_testing", model="llama-3.1-8b-instant")
        assert isinstance(client, ChatGroq)

    def test_get_groq_client_custom_model(self):
        """Test that model name is correctly set."""
        from utils.groq_client import get_groq_client

        model = "mixtral-8x7b-32768"
        client = get_groq_client("fake_key", model=model)
        assert client.model_name == model

    def test_available_models_not_empty(self):
        """Test that AVAILABLE_MODELS dictionary is populated."""
        from utils.groq_client import AVAILABLE_MODELS

        assert len(AVAILABLE_MODELS) > 0
        assert "llama-3.3-70b-versatile" in AVAILABLE_MODELS

    def test_simple_chat_calls_invoke(self):
        """Test simple_chat calls llm.invoke with correct messages."""
        from utils.groq_client import simple_chat
        from langchain.schema import AIMessage

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="Test response")

        result = simple_chat(mock_llm, "You are a lawyer.", "What is a contract?")

        assert mock_llm.invoke.called
        assert result == "Test response"


# ── Test rag_chain helpers ────────────────────────────────────────────────────

class TestRagChain:
    """Tests for RAG chain utilities."""

    def test_format_docs_empty(self):
        """Test _format_docs with empty list."""
        from utils.rag_chain import _format_docs

        result = _format_docs([])
        assert result == ""

    def test_format_docs_single(self):
        """Test _format_docs with one document."""
        from utils.rag_chain import _format_docs
        from langchain.schema import Document

        doc = Document(page_content="This is a clause.")
        result = _format_docs([doc])

        assert "This is a clause." in result
        assert "Excerpt 1" in result

    def test_format_docs_multiple(self):
        """Test _format_docs with multiple documents."""
        from utils.rag_chain import _format_docs
        from langchain.schema import Document

        docs = [
            Document(page_content="Clause 1"),
            Document(page_content="Clause 2"),
            Document(page_content="Clause 3"),
        ]
        result = _format_docs(docs)

        assert "Clause 1" in result
        assert "Clause 2" in result
        assert "Clause 3" in result
        assert "---" in result  # separator between docs


# ── Test embeddings helpers ───────────────────────────────────────────────────

class TestEmbeddings:
    """Tests for embeddings utilities."""

    def test_create_vector_store_returns_faiss(self):
        """Test that create_vector_store returns a FAISS object."""
        # This test requires the sentence-transformers model to be downloaded
        # Skip if running in CI without model
        pytest.importorskip("sentence_transformers")

        from utils.embeddings import create_vector_store
        from langchain_community.vectorstores import FAISS

        chunks = [
            "The tenant shall pay rent on the first of each month.",
            "The landlord may terminate with 30 days notice.",
            "This agreement shall be governed by the laws of New York.",
        ]

        # Use a mock to avoid downloading the model in CI
        with patch("utils.embeddings.get_embeddings_model") as mock_embed:
            mock_embed_instance = MagicMock()
            mock_embed.return_value = mock_embed_instance

            with patch("utils.embeddings.FAISS.from_documents") as mock_faiss:
                mock_faiss.return_value = MagicMock(spec=FAISS)
                store = create_vector_store(chunks)

                assert mock_faiss.called
                assert store is not None

    def test_similarity_search_calls_store(self):
        """Test similarity_search delegates to vector store."""
        from utils.embeddings import similarity_search
        from langchain.schema import Document

        mock_store = MagicMock()
        expected_docs = [Document(page_content="Relevant clause")]
        mock_store.similarity_search.return_value = expected_docs

        result = similarity_search(mock_store, "termination clause", k=3)

        mock_store.similarity_search.assert_called_once_with("termination clause", k=3)
        assert result == expected_docs
