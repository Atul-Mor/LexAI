"""Tab 4 — Jurisdiction-aware legal clause checker."""

import streamlit as st

from utils.rag_chain import analyze_document


# ── Jurisdiction Data ─────────────────────────────────────────────────────────

JURISDICTIONS = {
    "🇮🇳 India": {
        "regions": [
            "All India",
            "Maharashtra",
            "Karnataka",
            "Delhi",
            "Tamil Nadu",
            "Telangana",
            "Gujarat",
            "Rajasthan",
            "West Bengal",
            "Uttar Pradesh",
        ],
        "key_laws": "Indian Contract Act 1872, IT Act 2000, Labour Laws",
        "notes": "Non-compete enforceability varies by state; IP governed by Patents Act & Copyright Act",
    },
    "🇺🇸 United States": {
        "regions": [
            "Federal (All States)",
            "California",
            "New York",
            "Texas",
            "Florida",
            "Delaware",
            "Illinois",
            "Washington",
            "Massachusetts",
        ],
        "key_laws": "UCC, CCPA (CA), GDPR equivalents, State contract laws",
        "notes": "Non-competes void in California; Delaware favored for corporate contracts",
    },
    "🇬🇧 United Kingdom": {
        "regions": [
            "England & Wales",
            "Scotland",
            "Northern Ireland",
        ],
        "key_laws": "UK Contract Law, UK GDPR, Employment Rights Act 1996",
        "notes": "Post-Brexit data transfer rules apply; non-competes must be reasonable",
    },
    "🇪🇺 European Union": {
        "regions": [
            "EU-wide",
            "Germany",
            "France",
            "Netherlands",
            "Spain",
            "Italy",
        ],
        "key_laws": "GDPR, EU AI Act, EU Contract Law Directives",
        "notes": "GDPR strictly enforced; strong employee protections across EU",
    },
    "🇸🇬 Singapore": {
        "regions": ["Singapore"],
        "key_laws": "Contract Act, PDPA, Employment Act",
        "notes": "Balanced non-compete enforcement; strong IP protection",
    },
    "🇦🇺 Australia": {
        "regions": [
            "Federal",
            "New South Wales",
            "Victoria",
            "Queensland",
            "Western Australia",
        ],
        "key_laws": "Australian Consumer Law, Privacy Act, Fair Work Act",
        "notes": "Non-competes must be reasonable; Privacy Act applies to personal data",
    },
}


def render():
    """Render the Jurisdiction Check tab."""
    st.markdown("### 🌍 Jurisdiction Checker")
    st.markdown(
        "Select your jurisdiction to identify clauses that may be **illegal, unenforceable, "
        "or non-compliant** with local laws."
    )

    if "doc_text" not in st.session_state:
        st.info("📤 Please upload a document in the **Upload & Analyze** tab first.")
        return

    if "groq_client" not in st.session_state:
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        return

    # ── Jurisdiction Selection ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        country = st.selectbox(
            "🌐 Select Country / Region",
            options=list(JURISDICTIONS.keys()),
            key="jurisdiction_country",
        )

    with col2:
        regions = JURISDICTIONS[country]["regions"]
        region = st.selectbox(
            "📍 Select State / Province",
            options=regions,
            key="jurisdiction_region",
        )

    # Show jurisdiction info
    jdata = JURISDICTIONS[country]
    st.markdown(
        f"""
        <div style="
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
        ">
            <strong style="color: #d4a017;">📚 Applicable Laws:</strong> {jdata['key_laws']}<br>
            <strong style="color: #d4a017;">📝 Key Notes:</strong> {jdata['notes']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    check_btn = st.button(
        f"🔍 Check Compliance for {region}, {country.split(' ', 1)[-1]}",
        type="primary",
        use_container_width=True,
    )

    if check_btn:
        _run_jurisdiction_check(country, region, jdata)

    if "jurisdiction_result" in st.session_state:
        cached = st.session_state["jurisdiction_result"]
        if cached.get("jurisdiction") == f"{region}, {country}":
            _render_jurisdiction_result(cached)


def _run_jurisdiction_check(country: str, region: str, jdata: dict):
    """Run jurisdiction-specific compliance check."""
    llm = st.session_state["groq_client"]
    text = st.session_state["doc_text"]
    jurisdiction_label = f"{region}, {country.split(' ', 1)[-1]}"

    with st.spinner(f"🌍 Checking compliance for {jurisdiction_label}..."):
        prompt = f"""Review this legal document for compliance with the laws of: {jurisdiction_label}

Key applicable laws: {jdata['key_laws']}
Important local notes: {jdata['notes']}

Analyze and report on:

### ✅ COMPLIANT CLAUSES
(Clauses that align well with {jurisdiction_label} law)

### ❌ POTENTIALLY ILLEGAL / UNENFORCEABLE
(Clauses that may violate or be void under {jurisdiction_label} law — with specific law references)

### ⚠️ GREY AREAS
(Clauses that are uncertain and may need legal review for this jurisdiction)

### 📋 MISSING REQUIRED PROVISIONS
(Mandatory clauses under {jurisdiction_label} law that are absent from this document)

### 🔧 RECOMMENDED MODIFICATIONS
(Specific changes needed to make this document fully compliant with {jurisdiction_label} law)

### ⚖️ OVERALL COMPLIANCE RATING
(COMPLIANT / PARTIALLY COMPLIANT / NON-COMPLIANT — with brief justification)

Be specific and mention relevant laws, acts, or regulations by name where possible."""

        result = analyze_document(llm, text, prompt)

        st.session_state["jurisdiction_result"] = {
            "text": result,
            "jurisdiction": f"{region}, {country}",
        }


def _render_jurisdiction_result(result: dict):
    """Render jurisdiction check results."""
    st.markdown("---")
    st.markdown(f"### 📋 Compliance Report — {result['jurisdiction']}")

    st.markdown(
        f"""
        <div style="
            background: #161b22;
            border: 1px solid #30363d;
            border-left: 4px solid #d4a017;
            border-radius: 8px;
            padding: 1.5rem;
            line-height: 1.8;
        ">
        {result['text'].replace(chr(10), '<br>')}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "⚠️ This is AI-generated analysis for informational purposes only. "
        "Always consult a licensed attorney for legal advice."
    )

    st.download_button(
        "⬇️ Download Compliance Report",
        data=result["text"],
        file_name=f"lexai_compliance_{result['jurisdiction'].replace(', ', '_')}.txt",
        mime="text/plain",
    )
