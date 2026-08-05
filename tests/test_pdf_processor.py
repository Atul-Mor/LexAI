"""Tests for PDF processing utilities."""

import io
import pytest


# ── Test chunk_text ──────────────────────────────────────────────────────────

def test_chunk_text_basic():
    """Test that chunk_text splits text into chunks."""
    from utils.pdf_processor import chunk_text

    long_text = "This is a sentence. " * 200  # ~4000 chars
    chunks = chunk_text(long_text, chunk_size=500, chunk_overlap=50)

    assert len(chunks) > 1
    assert all(isinstance(c, str) for c in chunks)
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_empty():
    """Test chunk_text handles empty/whitespace-only input."""
    from utils.pdf_processor import chunk_text

    chunks = chunk_text("   ", chunk_size=500, chunk_overlap=50)
    assert chunks == []


def test_chunk_text_short_document():
    """Test chunk_text with a document shorter than chunk_size."""
    from utils.pdf_processor import chunk_text

    short_text = "This is a short contract."
    chunks = chunk_text(short_text, chunk_size=1000, chunk_overlap=200)

    assert len(chunks) == 1
    assert chunks[0] == short_text


def test_chunk_text_overlap():
    """Test that chunk overlap works correctly."""
    from utils.pdf_processor import chunk_text

    text = "word " * 500  # 2500 chars
    chunks_with_overlap = chunk_text(text, chunk_size=200, chunk_overlap=50)
    chunks_no_overlap = chunk_text(text, chunk_size=200, chunk_overlap=0)

    # Chunks with overlap should be more numerous
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)


# ── Test get_word_count ──────────────────────────────────────────────────────

def test_get_word_count():
    """Test word count calculation."""
    from utils.pdf_processor import get_word_count

    assert get_word_count("hello world") == 2
    assert get_word_count("one two three four five") == 5
    assert get_word_count("") == 1  # split on empty string gives ['']


def test_get_word_count_long():
    """Test word count on longer text."""
    from utils.pdf_processor import get_word_count

    text = "word " * 100
    assert get_word_count(text.strip()) == 100


# ── Test estimate_read_time ───────────────────────────────────────────────────

def test_estimate_read_time():
    """Test reading time estimation."""
    from utils.pdf_processor import estimate_read_time

    # 200 words at 200wpm = 1 min
    text_200 = "word " * 200
    assert estimate_read_time(text_200) == "1 min read"

    # 400 words at 200wpm = 2 min
    text_400 = "word " * 400
    assert estimate_read_time(text_400) == "2 min read"


def test_estimate_read_time_minimum():
    """Test that minimum read time is 1 min."""
    from utils.pdf_processor import estimate_read_time

    assert estimate_read_time("hello") == "1 min read"
    assert estimate_read_time("") == "1 min read"
