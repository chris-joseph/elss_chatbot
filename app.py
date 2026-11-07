"""
app.py
Streamlit web application for Groww Mutual Fund Facts-Only FAQ Assistant.
"""

import streamlit as st
import os
import sys
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.engine import RAGEngine

# Page configuration
st.set_page_config(
    page_title="Groww Mutual Fund FAQ Assistant",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom Styling for Groww Aesthetic
st.markdown("""
<style>
    /* Groww Brand Colors */
    :root {
        --groww-primary: #00D09C;
        --groww-dark: #121212;
        --groww-card: #1F2228;
        --groww-border: #2D3139;
    }
    
    .stApp {
        background-color: #0D1117;
        color: #E6EDF3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    .header-box {
        text-align: center;
        padding: 24px 16px 16px 16px;
        background: linear-gradient(180deg, rgba(0, 208, 156, 0.08) 0%, rgba(13, 17, 23, 0) 100%);
        border-radius: 12px;
        margin-bottom: 20px;
        border-bottom: 1px solid #21262D;
    }
    
    .header-title {
        font-size: 28px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 6px;
    }
    
    .header-title span {
        color: #00D09C;
    }
    
    .welcome-line {
        font-size: 16px;
        color: #8B949E;
        margin-bottom: 12px;
    }
    
    .disclaimer-badge {
        display: inline-block;
        background-color: rgba(0, 208, 156, 0.12);
        color: #00D09C;
        border: 1px solid rgba(0, 208, 156, 0.3);
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    
    .example-btn {
        margin: 4px 0px;
        width: 100%;
        text-align: left;
    }
    
    .response-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 18px 20px;
        margin-top: 16px;
        line-height: 1.6;
    }
    
    .source-tag {
        display: inline-flex;
        align-items: center;
        background-color: rgba(56, 139, 253, 0.15);
        color: #58A6FF;
        border: 1px solid rgba(56, 139, 253, 0.3);
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin-top: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Engine in session state
@st.cache_resource
def get_engine():
    return RAGEngine()

engine = get_engine()

# Header & Welcome
st.markdown("""
<div class="header-box">
    <div class="header-title">Groww Mutual Fund <span>FAQ Assistant</span></div>
    <div class="welcome-line">Welcome! I am your facts-only guide for mutual fund schemes, charges, exit loads, and statement guides.</div>
    <div class="disclaimer-badge">Facts-only. No investment advice.</div>
</div>
""", unsafe_allow_html=True)

# Example Questions Configuration
EXAMPLE_QUESTIONS = [
    "What is the expense ratio and exit load of Groww ELSS Tax Saver Fund?",
    "What is the mandatory lock-in period for ELSS mutual funds?",
    "How do I download my Mutual Fund capital gains statement on Groww?"
]

st.markdown("##### 💡 Try an Example Question:")
col1, col2, col3 = st.columns(3)

if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "trigger_submit" not in st.session_state:
    st.session_state.trigger_submit = False

with col1:
    if st.button("📊 Groww ELSS Details", use_container_width=True, help="Expense ratio & exit load"):
        st.session_state.user_query = EXAMPLE_QUESTIONS[0]
        st.session_state.trigger_submit = True
with col2:
    if st.button("🔒 ELSS Lock-in Rules", use_container_width=True, help="Statutory 3-year lock-in"):
        st.session_state.user_query = EXAMPLE_QUESTIONS[1]
        st.session_state.trigger_submit = True
with col3:
    if st.button("📄 Capital Gains Guide", use_container_width=True, help="Download tax statement"):
        st.session_state.user_query = EXAMPLE_QUESTIONS[2]
        st.session_state.trigger_submit = True

# Search / Question Input
query_input = st.text_input(
    "Ask a factual mutual fund question:",
    value=st.session_state.user_query,
    placeholder="e.g., What is the minimum SIP for Groww Multicap Fund?",
    key="search_input"
)

# Handle submission
active_query = query_input if query_input else (st.session_state.user_query if st.session_state.trigger_submit else "")

if active_query:
    with st.spinner("Retrieving verified facts from official sources..."):
        result = engine.answer(active_query)

    # Display Response
    status = result.get("status", "factual")
    ans_text = result.get("answer", "")

    if status == "refusal_pii":
        st.error(f"🛡️ **Privacy Guardrail Triggered:**\n\n{ans_text}")
    elif status == "refusal_advice":
        st.warning(f"⚖️ **Facts-Only Policy:**\n\n{ans_text}")
    elif status == "refusal_performance":
        st.info(f"📈 **No Performance Claims:**\n\n{ans_text}")
    else:
        st.success("✅ **Official Factsheet & Guide Answer:**")
        st.markdown(f"""
        <div class="response-card">
            {ans_text}
        </div>
        """, unsafe_allow_html=True)

    # Show latency and details in expander
    with st.expander("ℹ️ Verification & Retrieval Details", expanded=False):
        st.write(f"**Query Status:** `{status}`")
        st.write(f"**Retrieval Latency:** `{result.get('latency_ms', 0)} ms`")
        if result.get("source_url"):
            st.write(f"**Primary Source Citation:** [{result.get('source_title')}]({result.get('source_url')})")
        st.caption("All answers are derived strictly from official Groww AMC, AMFI, and SEBI regulatory pages.")

# Sidebar Information & Sources
with st.sidebar:
    st.markdown("### 📚 Scoped Schemes (Groww AMC)")
    st.markdown("""
    * **Groww ELSS Tax Saver Fund** (Equity / ELSS)
    * **Groww Nifty Total Market Index Fund** (Index)
    * **Groww Multicap Fund** (Equity / Multicap)
    * **Groww Liquid Fund** (Debt / Liquid)
    * **Groww Banking & Financial Services Fund** (Sectoral)
    """)
    
    st.markdown("---")
    st.markdown("### 🛡️ Guardrails & Policy")
    st.markdown("""
    * **Facts-Only**: Strictly no buy/sell advice or portfolio recommendations.
    * **Zero PII**: Automatic detection & refusal of PAN, Aadhaar, account numbers, and phone numbers.
    * **Official Sources**: Only public AMC, AMFI, and SEBI links cited.
    * **Brevity**: Answers are concise (≤3 sentences).
    """)
    
    st.markdown("---")
    st.markdown("### 📖 Official Corpus")
    catalog_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sources", "sources_catalog.json")
    if os.path.exists(catalog_path):
        with open(catalog_path, "r") as f:
            cat_data = json.load(f)
        st.caption(f"Indexed **{len(cat_data)} verified public sources**.")
        with st.expander("View Indexed Sources"):
            for item in cat_data[:10]:
                st.markdown(f"- [{item['title']}]({item['url']})")
            if len(cat_data) > 10:
                st.caption(f"... and {len(cat_data) - 10} more.")
