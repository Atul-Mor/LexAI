"""Tab 1 — Upload & Analyze legal document."""

import streamlit as st

from utils.embeddings import create_vector_store
from utils.pdf_processor import (
    chunk_text,
    estimate_read_time,
    extract_text_from_pdf,
    get_word_count,
)
from utils.rag_chain import analyze_document


def render():
    """Render the Upload & Analyze tab."""

    st.markdown("### 📤 Upload Your Legal Document")
    st.markdown(
        "Upload a PDF document for AI analysis."
    )

    # ── File Upload ──────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key="analyze_uploader",
        help="Supports contracts, NDAs, leases, employment agreements, and more.",
    )

    if uploaded_file is None:
        _render_placeholder()
        return

    # ── Process Document ─────────────────────────────────────────────────────
    if (
        "doc_name" not in st.session_state
        or st.session_state.get("doc_name") != uploaded_file.name
    ):
        with st.spinner("🔍 Processing document..."):
            _process_document(uploaded_file)

    # ── Document Stats ───────────────────────────────────────────────────────
    _render_doc_stats()

    st.divider()

    # ── Analysis Section ─────────────────────────────────────────────────────
    if "groq_client" not in st.session_state:
        st.warning("⚠️ Please enter your Groq API key in the sidebar to generate analysis.")
        return

    if st.button("🚀 Generate Full Analysis", type="primary", use_container_width=True):
        _generate_analysis()

    # Show cached analysis
    if "doc_analysis" in st.session_state:
        _render_analysis(st.session_state["doc_analysis"])


def _process_document(uploaded_file):
    """Extract, chunk, and embed the uploaded PDF."""
    text, page_count = extract_text_from_pdf(uploaded_file)

    if not text.strip():
        st.error("❌ Could not extract text from this PDF. It may be scanned or image-based.")
        return

    chunks = chunk_text(text)
    vector_store = create_vector_store(chunks)

    # Persist in session state for all tabs to use
    st.session_state["doc_text"] = text
    st.session_state["doc_chunks"] = chunks
    st.session_state["doc_name"] = uploaded_file.name
    st.session_state["doc_pages"] = page_count
    st.session_state["vector_store"] = vector_store
    st.session_state.pop("doc_analysis", None)
    st.session_state.pop("chat_history", None)

    st.success(f"✅ Document processed — {len(chunks)} chunks indexed!")


def _render_doc_stats():
    """Display document metadata stats."""
    text = st.session_state.get("doc_text", "")
    name = st.session_state.get("doc_name", "")
    pages = st.session_state.get("doc_pages", 0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 File", name[:20] + "..." if len(name) > 20 else name)
    with col2:
        st.metric("📃 Pages", pages)
    with col3:
        st.metric("📝 Words", f"{get_word_count(text):,}")
    with col4:
        st.metric("⏱️ Read Time", estimate_read_time(text))


def _generate_analysis():
    """Run AI analysis and cache the results."""
    llm = st.session_state["groq_client"]
    text = st.session_state["doc_text"]

    with st.spinner("⚖️ LexAI is analyzing your document..."):
        prompt = """
        ...
Formatting Rules:
- Do NOT use Markdown headings (#, ##, ###).
- Do NOT use **bold** or __bold__.
- Use plain text section titles followed by a colon.

Example:

DOCUMENT TYPE:
Research Paper

AUTHORS:
John Doe
Jane Doe

KEY FINDINGS:
• Finding 1
• Finding 2

EXECUTIVE SUMMARY:
...


You are LexAI, an expert legal and document analysis assistant.

First identify the document type.

Possible document types include:
- Contract
- Agreement
- NDA
- Lease
- Employment Agreement
- Court Judgment
- Legal Notice
- Government Policy
- Research Paper
- Report
- Memorandum
- Other

Return a clean, structured report.

Always include:
1. DOCUMENT TYPE
2. TITLE (if available)
3. PARTIES / AUTHORS / ORGANIZATIONS (whichever is applicable)
4. DATE / EFFECTIVE DATE / PUBLICATION DATE
5. PURPOSE OF THE DOCUMENT

Then provide ONLY the sections relevant to the identified document.

For Contracts or Agreements:
- Parties
- Obligations
- Payment Terms
- Confidentiality
- Intellectual Property
- Termination
- Governing Law
- Dispute Resolution

For Research Papers:
- Authors
- Institutions
- Research Objective
- Methodology
- Key Findings (bullet points)
- Recommendations / Policy Implications
- Funding (if mentioned)

For Court Judgments:
- Court
- Judge(s)
- Parties
- Facts
- Issues
- Decision
- Reasoning

Finish with:

## Executive Summary

Rules:
- Never invent information.
- Never create sections that are not applicable.
- If a section does not exist, omit it entirely.
- Do not write "Not mentioned", "None", or "No payment terms".
- Use concise bullet points.
- Keep the entire response under 700 words.
"""

        analysis = analyze_document(llm, text, prompt)
        st.session_state["doc_analysis"] = analysis


def _render_analysis(analysis: str):
    """Render the analysis result in a styled container."""
    st.markdown("---")
    st.markdown("### 📊 Document Analysis")
    st.markdown("### 📊 Document Analysis")

    with st.container(border=True):
        st.markdown(analysis)

    # Download button
    st.download_button(
        "⬇️ Download Analysis",
        data=analysis,
        file_name=f"lexai_analysis_{st.session_state.get('doc_name', 'document')}.txt",
        mime="text/plain",
    )


def _render_placeholder():
    """Render the empty state placeholder."""
    st.markdown(
        """
        <div style="
            border: 2px dashed #30363d;
            border-radius: 12px;
            padding: 3rem;
            text-align: center;
            margin-top: 1rem;
        ">
            <div style="font-size: 4rem; margin-bottom: 1rem;">⚖️</div>
            <h3 style="color: #d4a017; margin-bottom: 0.5rem;">No document uploaded yet</h3>
            <p style="color: #8b949e;">Upload a PDF contract, NDA, lease, or legal agreement above</p>
            <p style="color: #484f58; font-size: 0.85rem;">Supports: Contracts · NDAs · Leases · Employment Agreements · Terms of Service</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
