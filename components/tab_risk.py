"""Tab 2 — Red Flag Detector."""

import json

import streamlit as st

from utils.rag_chain import analyze_document


# ── Risk Categories & Weights ─────────────────────────────────────────────────

RED_FLAG_CATEGORIES = {
    "Liability & Indemnification": {
        "keywords": ["unlimited liability", "indemnify", "hold harmless", "indemnification"],
        "description": "Clauses that could expose you to unlimited financial risk",
        "severity": "HIGH",
    },
    "Non-Compete & Restrictions": {
        "keywords": ["non-compete", "non-solicitation", "restraint of trade", "competing business"],
        "description": "Restrictions on your professional activities after contract ends",
        "severity": "HIGH",
    },
    "Intellectual Property Transfer": {
        "keywords": ["assign", "all rights", "work for hire", "intellectual property", "IP ownership"],
        "description": "Clauses that transfer ownership of your work or ideas",
        "severity": "HIGH",
    },
    "Automatic Renewal": {
        "keywords": ["automatic renewal", "auto-renew", "evergreen", "unless terminated"],
        "description": "Contract auto-renews without explicit notice — easy to miss",
        "severity": "MEDIUM",
    },
    "Unilateral Modification": {
        "keywords": ["may amend", "reserves the right", "at its sole discretion", "without notice"],
        "description": "One party can change terms without your consent",
        "severity": "HIGH",
    },
    "Governing Law & Jurisdiction": {
        "keywords": ["governing law", "jurisdiction", "venue", "courts of"],
        "description": "Disputes must be resolved in a specific location or under specific law",
        "severity": "MEDIUM",
    },
    "Liquidated Damages": {
        "keywords": ["liquidated damages", "penalty", "forfeiture", "damages clause"],
        "description": "Fixed penalties for breach that may be disproportionate",
        "severity": "MEDIUM",
    },
    "Broad Confidentiality": {
        "keywords": ["confidential", "proprietary", "trade secret", "non-disclosure"],
        "description": "Overly broad confidentiality that may restrict you unreasonably",
        "severity": "LOW",
    },
    "Termination at Will": {
        "keywords": ["at will", "terminate at any time", "without cause", "immediately terminate"],
        "description": "Other party can terminate the agreement at any time for any reason",
        "severity": "MEDIUM",
    },
    "Waiver of Rights": {
        "keywords": ["waive", "waiver", "waives all rights", "relinquish"],
        "description": "Clauses where you surrender legal rights",
        "severity": "HIGH",
    },
}

SEVERITY_COLORS = {
    "HIGH": "#f85149",
    "MEDIUM": "#d29922",
    "LOW": "#3fb950",
}

SEVERITY_ICONS = {
    "HIGH": "🔴",
    "MEDIUM": "🟡",
    "LOW": "🟢",
}


def render():
    """Render the Red Flag Detector tab."""
    st.markdown("### 🚩 Red Flag Detector")
    st.markdown(
        "AI scans your document for potentially dangerous clauses and assigns a risk score."
    )

    if "doc_text" not in st.session_state:
        st.info("📤 Please upload a document in the **Upload & Analyze** tab first.")
        return

    if "groq_client" not in st.session_state:
        st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"**Document:** `{st.session_state.get('doc_name', 'Unknown')}`")
    with col2:
        scan_btn = st.button("🔍 Scan for Red Flags", type="primary", use_container_width=True)

    if scan_btn:
        _run_risk_scan()

    if "risk_results" in st.session_state:
        _render_risk_dashboard(st.session_state["risk_results"])


def _run_risk_scan():
    """Run AI-powered red flag detection."""
    llm = st.session_state["groq_client"]
    text = st.session_state["doc_text"]

    with st.spinner("🔍 Scanning for red flags..."):
        prompt = f"""Analyze this legal document for potentially risky or dangerous clauses.

For each risk found, provide a JSON response in this EXACT format:
{{
  "overall_risk_score": <number 0-100>,
  "risk_summary": "<one paragraph summary>",
  "flags": [
    {{
      "title": "<risk title>",
      "severity": "<HIGH|MEDIUM|LOW>",
      "clause": "<exact or paraphrased clause from document>",
      "explanation": "<why this is risky>",
      "recommendation": "<what to do about it>"
    }}
  ]
}}

Check for: unlimited liability, non-compete clauses, IP ownership transfer, auto-renewal, 
unilateral changes, unfair termination rights, waiver of rights, excessive penalties, 
jurisdiction issues, and overly broad confidentiality.

Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""

        raw = analyze_document(llm, text, prompt)

        # Parse JSON from response
        try:
            # Strip markdown code blocks if present
            clean = raw.strip()
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            results = json.loads(clean.strip())
        except json.JSONDecodeError:
            # Fallback: show raw response
            results = {
                "overall_risk_score": 50,
                "risk_summary": raw,
                "flags": [],
            }

        st.session_state["risk_results"] = results


def _render_risk_dashboard(results: dict):
    """Render the risk dashboard UI."""
    score = results.get("overall_risk_score", 0)
    summary = results.get("risk_summary", "")
    flags = results.get("flags", [])

    # ── Risk Score Gauge ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Overall Risk Score")

    if score >= 70:
        score_color = "#f85149"
        risk_label = "HIGH RISK"
        risk_emoji = "🔴"
    elif score >= 40:
        score_color = "#d29922"
        risk_label = "MEDIUM RISK"
        risk_emoji = "🟡"
    else:
        score_color = "#3fb950"
        risk_label = "LOW RISK"
        risk_emoji = "🟢"

    st.markdown(
        f"""
        <div style="
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            margin-bottom: 1rem;
        ">
            <div style="font-size: 4rem; font-weight: 800; color: {score_color};">{score}/100</div>
            <div style="font-size: 1.2rem; color: {score_color}; font-weight: 600;">{risk_emoji} {risk_label}</div>
            <div style="margin-top: 0.8rem; background: #21262d; border-radius: 6px; height: 12px; overflow: hidden;">
                <div style="width: {score}%; height: 100%; background: {score_color}; border-radius: 6px; transition: width 0.5s;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Summary ──────────────────────────────────────────────────────────────
    if summary:
        st.markdown(f"> {summary}")

    # ── Flag Counts ──────────────────────────────────────────────────────────
    high = sum(1 for f in flags if f.get("severity") == "HIGH")
    med = sum(1 for f in flags if f.get("severity") == "MEDIUM")
    low = sum(1 for f in flags if f.get("severity") == "LOW")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🚩 Total Flags", len(flags))
    c2.metric("🔴 High Risk", high)
    c3.metric("🟡 Medium Risk", med)
    c4.metric("🟢 Low Risk", low)

    # ── Individual Flags ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Detected Risk Flags")

    if not flags:
        st.success("✅ No significant red flags detected in this document!")
        return

    # Sort by severity
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    flags_sorted = sorted(flags, key=lambda x: severity_order.get(x.get("severity", "LOW"), 2))

    for flag in flags_sorted:
        severity = flag.get("severity", "LOW")
        icon = SEVERITY_ICONS.get(severity, "⚪")
        color = SEVERITY_COLORS.get(severity, "#8b949e")

        with st.expander(f"{icon} {flag.get('title', 'Unknown Risk')} — **{severity}**"):
            st.markdown(
                f"""
                <div style="border-left: 3px solid {color}; padding-left: 1rem; margin-bottom: 0.8rem;">
                    <strong>📌 Clause Found:</strong><br>
                    <em style="color: #8b949e;">"{flag.get('clause', 'N/A')}"</em>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(f"**⚠️ Why It's Risky:** {flag.get('explanation', 'N/A')}")
            st.markdown(f"**✅ Recommendation:** {flag.get('recommendation', 'Consult a lawyer.')}")

    # Download report
    report_text = _build_risk_report(results)
    st.download_button(
        "⬇️ Download Risk Report",
        data=report_text,
        file_name=f"lexai_risk_report_{st.session_state.get('doc_name', 'document')}.txt",
        mime="text/plain",
    )


def _build_risk_report(results: dict) -> str:
    """Build a plain text risk report for download."""
    lines = [
        "LEXAI — RISK ANALYSIS REPORT",
        "=" * 40,
        f"Overall Risk Score: {results.get('overall_risk_score', 0)}/100",
        "",
        "SUMMARY:",
        results.get("risk_summary", ""),
        "",
        "DETECTED FLAGS:",
        "-" * 40,
    ]
    for i, flag in enumerate(results.get("flags", []), 1):
        lines += [
            f"\n{i}. {flag.get('title')} [{flag.get('severity')}]",
            f"   Clause: {flag.get('clause')}",
            f"   Risk: {flag.get('explanation')}",
            f"   Action: {flag.get('recommendation')}",
        ]
    return "\n".join(lines)
