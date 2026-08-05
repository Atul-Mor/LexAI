"""Tab 5 — Chat with your legal document using RAG."""

import streamlit as st

from utils.rag_chain import query_with_sources


# ── Suggested Questions ───────────────────────────────────────────────────────

SUGGESTED_QUESTIONS = [
    "What are the main obligations of each party?",
    "When does this agreement expire or terminate?",
    "What happens if either party breaches the contract?",
    "Are there any payment terms or financial obligations?",
    "What are the confidentiality requirements?",
    "Can I work for a competitor after this contract ends?",
    "Who owns the intellectual property created under this agreement?",
    "What are the conditions for early termination?",
    "Is there an automatic renewal clause?",
    "What dispute resolution process is specified?",
]


def render():
    """Render the Chat with Document tab."""
    st.markdown("### 💬 Chat with Your Document")
    st.markdown(
        "Ask any question about your legal document. LexAI uses RAG to search the "
        "document and provide grounded, accurate answers."
    )

    if "doc_text" not in st.session_state:
        st.info("📤 Please upload a document in the **Upload & Analyze** tab first.")
        return

    if "groq_client" not in st.session_state:
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        return

    if "vector_store" not in st.session_state:
        st.error("❌ Document not indexed. Please re-upload in the Upload & Analyze tab.")
        return

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # ── Document Context Banner ───────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        ">
            📄 <strong>Active Document:</strong> {st.session_state.get('doc_name', 'Unknown')}
            &nbsp;|&nbsp;
            🧩 <strong>Chunks Indexed:</strong> {len(st.session_state.get('doc_chunks', []))}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Suggested Questions ───────────────────────────────────────────────────
    with st.expander("💡 Suggested Questions", expanded=len(st.session_state["chat_history"]) == 0):
        cols = st.columns(2)
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 2]:
                if st.button(f"❓ {question}", key=f"suggest_{i}", use_container_width=True):
                    _ask_question(question)

    st.divider()

    # ── Chat History ─────────────────────────────────────────────────────────
    _render_chat_history()

    # ── Input Box ────────────────────────────────────────────────────────────
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input(
            "Ask a question about your document...",
            placeholder="e.g. What are my obligations under this contract?",
            key="chat_input",
            label_visibility="collapsed",
        )
    with col2:
        send_btn = st.button("Send ➤", type="primary", use_container_width=True)

    if send_btn and user_input.strip():
        _ask_question(user_input.strip())

    # Clear chat
    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()


def _ask_question(question: str):
    """Process a user question through the RAG pipeline."""
    llm = st.session_state["groq_client"]
    vector_store = st.session_state["vector_store"]

    # Add user message to history
    st.session_state["chat_history"].append({
        "role": "user",
        "content": question,
    })

    with st.spinner("🔍 Searching document..."):
        answer, sources = query_with_sources(llm, vector_store, question)

    # Format source excerpts
    source_texts = []
    for i, doc in enumerate(sources[:3]):  # Show top 3 sources
        preview = doc.page_content[:200].strip() + "..."
        source_texts.append(f"**Excerpt {i + 1}:** *{preview}*")

    st.session_state["chat_history"].append({
        "role": "assistant",
        "content": answer,
        "sources": source_texts,
    })

    st.rerun()


def _render_chat_history():
    """Render the full chat conversation history."""
    history = st.session_state.get("chat_history", [])

    if not history:
        st.markdown(
            """
            <div style="
                text-align: center;
                padding: 2rem;
                color: #484f58;
                border: 2px dashed #21262d;
                border-radius: 8px;
            ">
                <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">💬</div>
                <p>Start a conversation by asking a question below or selecting a suggested question above.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    for message in history:
        role = message["role"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])

        else:
            with st.chat_message("assistant", avatar="⚖️"):
                st.markdown(message["content"])

                # Show source citations
                sources = message.get("sources", [])
                if sources:
                    with st.expander("📚 Source Excerpts from Document"):
                        for src in sources:
                            st.markdown(
                                f"""
                                <div style="
                                    background: #21262d;
                                    border-left: 3px solid #d4a017;
                                    padding: 0.75rem;
                                    border-radius: 4px;
                                    margin-bottom: 0.5rem;
                                    font-size: 0.85rem;
                                    color: #8b949e;
                                ">
                                {src}
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
