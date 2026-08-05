"""PDF processing utilities for LexAI."""

import io
from typing import List, Tuple

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_file) -> Tuple[str, int]:
    """
    Extract full text from an uploaded PDF file object.

    Args:
        pdf_file: Streamlit UploadedFile or file-like object.

    Returns:
        Tuple of (full_text, page_count)
    """
    pdf_bytes = pdf_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text_parts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            full_text_parts.append(f"[Page {page_num + 1}]\n{text.strip()}")

    page_count = len(doc)
    doc.close()

    return "\n\n".join(full_text_parts), page_count


def extract_text_from_bytes(pdf_bytes: bytes) -> Tuple[str, int]:
    """
    Extract text from raw PDF bytes.

    Args:
        pdf_bytes: Raw PDF file bytes.

    Returns:
        Tuple of (full_text, page_count)
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    full_text_parts = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            full_text_parts.append(f"[Page {page_num + 1}]\n{text.strip()}")

    page_count = len(doc)
    doc.close()

    return "\n\n".join(full_text_parts), page_count


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Split text into overlapping chunks for embedding.

    Args:
        text: Full document text.
        chunk_size: Maximum chunk size in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        List of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [c for c in chunks if c.strip()]


def get_word_count(text: str) -> int:
    """Return approximate word count of text."""
    return len(text.split())


def estimate_read_time(text: str, wpm: int = 200) -> str:
    """Estimate reading time based on word count."""
    words = get_word_count(text)
    minutes = max(1, round(words / wpm))
    return f"{minutes} min read"
