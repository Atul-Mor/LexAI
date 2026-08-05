"""Tab 3 — Compare two legal documents side-by-side."""

import streamlit as st

from utils.embeddings import create_vector_store
from utils.pdf_processor import chunk_text, extract_text_from_pdf
from utils.rag_chain import analyze_document


def render():
    """Render the Document Comparison tab."""
    st.markdown("### ⚖️ Compare Two Documents")
    st.markdown(
        "Upload two versions of the same contract to identify additions, removals, "
        "and significant changes."
    )

    if "groq_client" not in st.session_state:
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📄 Document A (Original)")
        file_a = st.file_uploader(
            "Upload original document",
            type=["pdf"],
            key="compare_file_a",
        )

    with col2:
        st.markdown("#### 📄 Document B (Revised)")
        file_b = st.file_uploader(
            "Upload revised document",
            type=["pdf"],
            key="compare_file_b",
        )

    # Process uploaded files
    text_a = text_b = None

    if file_a:
        with st.spinner("Processing Document A..."):
            text_a, pages_a = extract_text_from_pdf(file_a)
        st.success(f"✅ Document A loaded — {pages_a} pages")

    if file_b:
        with st.spinner("Processing Document B..."):
            text_b, pages_b = extract_text_from_pdf(file_b)
        st.success(f"✅ Document B loaded — {pages_b} pages")

    if text_a and text_b:
        st.divider()

        compare_btn = st.button(
            "🔍 Compare Documents",
            type="primary",
            use_container_width=True,
        )

        if compare_btn:
            _run_comparison(text_a, text_b, file_a.name, file_b.name)

        if "comparison_result" in st.session_state:
            _render_comparison(st.session_state["comparison_result"])


def _run_comparison(text_a: str, text_b: str, name_a: str, name_b: str):
    """Run AI-powered document comparison."""
    llm = st.session_state["groq_client"]

    with st.spinner("⚖️ Comparing documents..."):
        prompt = f"""Compare these two versions of a legal document and identify ALL differences.

DOCUMENT A ({name_a}) — First 4000 chars:
{text_a[:4000]}

DOCUMENT B ({name_b}) — First 4000 chars:
{text_b[:4000]}

Provide a structured comparison with:

### 📌 KEY CHANGES SUMMARY
(3-5 sentence overview of major differences)

### ➕ ADDITIONS IN DOCUMENT B
(clauses/sections added that weren't in Document A)

### ➖ REMOVALS FROM DOCUMENT A
(clauses/sections in A that are missing from B)

### ✏️ SIGNIFICANT MODIFICATIONS
(clauses that exist in both but changed meaningfully)

### ⚠️ RISK IMPLICATIONS
(what the changes mean for the parties — which version is more favorable and why)

### ✅ RECOMMENDATION
(which version to prefer and what to negotiate)

Be specific and cite actual text from the documents."""

        result = analyze_document(llm, "", prompt)
        st.session_state["comparison_result"] = {
            "text": result,
            "name_a": name_a,
            "name_b": name_b,
        }


def _render_comparison(result: dict):
    """Render the comparison results."""
    st.markdown("---")
    st.markdown(
        f"### 📊 Comparison: `{result['name_a']}` vs `{result['name_b']}`"
    )

    st.markdown(
        f"""
        <div style="
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 1.5rem;
            line-height: 1.8;
        ">
        {result['text'].replace(chr(10), '<br>')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.download_button(
        "⬇️ Download Comparison Report",
        data=result["text"],
        file_name=f"lexai_comparison_{result['name_a']}_vs_{result['name_b']}.txt",
        mime="text/plain",
    )
