"""LexAI — Legal Document Analyzer powered by Groq + LangChain RAG."""

import os

import streamlit as st
from dotenv import load_dotenv

from components import tab_analyze, tab_chat, tab_compare, tab_jurisdiction, tab_risk
from utils.groq_client import AVAILABLE_MODELS, get_groq_client

# ── Load env ──────────────────────────────────────────────────────────────────
load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LexAI — Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/yourusername/lexai",
        "Report a bug": "https://github.com/yourusername/lexai/issues",
        "About": "LexAI — AI-powered legal document analyzer using Groq + RAG",
    },
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Import Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .lexai-header {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .lexai-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: linear-gradient(135deg, rgba(212,160,23,0.05) 0%, transparent 60%);
        pointer-events: none;
    }
    .lexai-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #d4a017, #f5c842, #d4a017);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }
    .lexai-subtitle {
        color: #8b949e;
        font-size: 1rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    .lexai-badge {
        display: inline-block;
        background: rgba(212,160,23,0.15);
        border: 1px solid rgba(212,160,23,0.3);
        color: #d4a017;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 0.8rem;
        margin-right: 0.4rem;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #161b22;
        border-radius: 10px;
        padding: 4px;
        border: 1px solid #30363d;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        color: #8b949e !important;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: #d4a017 !important;
        color: #0d1117 !important;
        font-weight: 700;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #d4a017;
    }

    /* ── Buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #d4a017, #b8860b);
        color: #0d1117;
        border: none;
        font-weight: 700;
        border-radius: 8px;
        transition: all 0.2s ease;
        box-shadow: 0 4px 15px rgba(212,160,23,0.2);
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(212,160,23,0.35);
    }

    /* ── Metrics ── */
    [data-testid="metric-container"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }

    /* ── File uploader ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed #30363d;
        border-radius: 10px;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #d4a017;
    }

    /* ── Success/Error/Warning ── */
    .stSuccess {
        border-left: 4px solid #3fb950;
        background: rgba(63,185,80,0.08);
    }
    .stWarning {
        border-left: 4px solid #d29922;
        background: rgba(210,153,34,0.08);
    }
    .stError {
        border-left: 4px solid #f85149;
        background: rgba(248,81,73,0.08);
    }

    /* ── Chat messages ── */
    [data-testid="chat-message-container"] {
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        border: 1px solid #30363d;
        border-radius: 8px;
        background: #161b22;
    }

    /* ── Footer ── */
    .lexai-footer {
        text-align: center;
        color: #484f58;
        font-size: 0.75rem;
        padding: 1rem 0;
        border-top: 1px solid #21262d;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="lexai-header">
        <h1 class="lexai-title">⚖️ LexAI</h1>
        <p class="lexai-subtitle">AI-Powered Legal Document Analyzer · Built with Groq + LangChain RAG</p>
        <div>
            <span class="lexai-badge">🚀 Groq API</span>
            <span class="lexai-badge">🔗 LangChain RAG</span>
            <span class="lexai-badge">🤖 Llama 3.3 70B</span>
            <span class="lexai-badge">⚡ Real-time</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.divider()

    # API Key input
    api_key_env = os.getenv("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "🔑 Groq API Key",
        value=api_key_env,
        type="password",
        placeholder="gsk_...",
        help="Get your free API key at console.groq.com",
    )

    if api_key_input:
        if "groq_client" not in st.session_state or st.session_state.get("_api_key") != api_key_input:
            # Model selection
            model_key = st.selectbox(
                "🤖 Model",
                options=list(AVAILABLE_MODELS.keys()),
                format_func=lambda x: AVAILABLE_MODELS[x],
                key="model_selector",
            )
            with st.spinner("Connecting to Groq..."):
                try:
                    client = get_groq_client(api_key_input, model=model_key)
                    st.session_state["groq_client"] = client
                    st.session_state["_api_key"] = api_key_input
                    st.success("✅ Connected!")
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)[:80]}")
    else:
        st.markdown(
            """
            <div style="
                background: rgba(212,160,23,0.1);
                border: 1px solid rgba(212,160,23,0.3);
                border-radius: 8px;
                padding: 0.75rem;
                font-size: 0.85rem;
                color: #d4a017;
            ">
                🔑 Enter your <a href="https://console.groq.com" target="_blank" style="color: #d4a017;">Groq API key</a> to get started.<br><br>
                It's <strong>100% free</strong> — no credit card needed.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div style="margin-top: 0.5rem; font-size: 0.75rem; color: #484f58; text-align: center;">
            Also set in .env as GROQ_API_KEY
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # Document status
    st.markdown("### 📄 Document Status")
    if "doc_name" in st.session_state:
        st.markdown(
            f"""
            <div style="background: rgba(63,185,80,0.1); border: 1px solid rgba(63,185,80,0.3);
                        border-radius: 8px; padding: 0.75rem; font-size: 0.85rem;">
                ✅ <strong>{st.session_state['doc_name']}</strong><br>
                📃 {st.session_state.get('doc_pages', 0)} pages &nbsp;·&nbsp;
                🧩 {len(st.session_state.get('doc_chunks', []))} chunks
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("🗑️ Clear Document", use_container_width=True):
            for key in ["doc_text", "doc_chunks", "doc_name", "doc_pages",
                        "vector_store", "doc_analysis", "risk_results",
                        "comparison_result", "jurisdiction_result", "chat_history"]:
                st.session_state.pop(key, None)
            st.rerun()
    else:
        st.markdown(
            "<div style='color: #484f58; font-size: 0.85rem;'>No document loaded</div>",
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown(
        """
        <div style="font-size: 0.75rem; color: #484f58; text-align: center; line-height: 1.6;">
            ⚠️ LexAI is for informational purposes only.<br>
            Always consult a licensed attorney for legal advice.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📤 Upload & Analyze",
    "🚩 Red Flag Detector",
    "⚖️ Compare Docs",
    "🌍 Jurisdiction Check",
    "💬 Chat with Doc",
])

with tab1:
    tab_analyze.render()

with tab2:
    tab_risk.render()

with tab3:
    tab_compare.render()

with tab4:
    tab_jurisdiction.render()

with tab5:
    tab_chat.render()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="lexai-footer">
        ⚖️ LexAI · Built with Streamlit, Groq API & LangChain ·
        <a href="https://github.com/yourusername/lexai" target="_blank" style="color: #d4a017;">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
