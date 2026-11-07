"""
app.py
Conversational Streamlit web application for Groww Mutual Fund Facts-Only FAQ Assistant.
Features modern chat interface (st.chat_message & st.chat_input), state-isolated example pills,
and real-time verification badges.
"""

import streamlit as st
import os
import sys
import json

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.engine import RAGEngine

# Page Configuration
st.set_page_config(
    page_title="Groww Mutual Fund FAQ Assistant",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Styling for Groww Brand Aesthetic
st.markdown("""
<style>
    :root {
        --groww-primary: #00D09C;
        --groww-dark: #0D1117;
        --groww-card: #161B22;
        --groww-border: #30363D;
    }
    
    .stApp {
        background-color: var(--groww-dark);
        color: #E6EDF3;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    .header-box {
        text-align: center;
        padding: 24px 16px 14px 16px;
        background: linear-gradient(180deg, rgba(0, 208, 156, 0.12) 0%, rgba(13, 17, 23, 0) 100%);
        border-radius: 12px;
        margin-bottom: 20px;
        border-bottom: 1px solid var(--groww-border);
    }
    
    .header-title {
        font-size: 26px;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: 6px;
    }
    
    .header-title span {
        color: var(--groww-primary);
    }
    
    .welcome-line {
        font-size: 15px;
        color: #8B949E;
        margin-bottom: 10px;
    }
    
    .disclaimer-badge {
        display: inline-block;
        background-color: rgba(0, 208, 156, 0.15);
        color: var(--groww-primary);
        border: 1px solid rgba(0, 208, 156, 0.35);
        padding: 3px 12px;
        border-radius: 14px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }

    .stButton>button {
        border-radius: 8px;
        border: 1px solid var(--groww-border);
        background-color: var(--groww-card);
        color: #E6EDF3;
        font-size: 13px;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        border-color: var(--groww-primary);
        color: var(--groww-primary);
    }
</style>
""", unsafe_allow_html=True)

# Cache RAG Engine instance
@st.cache_resource
def get_engine():
    return RAGEngine()

engine = get_engine()

# Initialize Chat Message History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header & Welcome line
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

st.markdown("##### 💡 Example Questions (Click to Ask):")
col1, col2, col3 = st.columns(3)

def handle_query_submission(query_text: str):
    """Processes a user query cleanly through the RAG pipeline."""
    # Append user turn
    st.session_state.messages.append({"role": "user", "content": query_text})
    
    # Run RAG Engine
    res = engine.answer(query_text)
    
    # Append assistant turn
    st.session_state.messages.append({
        "role": "assistant",
        "content": res["answer"],
        "status": res.get("status", "factual"),
        "source_url": res.get("source_url"),
        "source_title": res.get("source_title"),
        "latency_ms": res.get("latency_ms", 0.0)
    })

# Example Button Handlers
with col1:
    if st.button("📊 Groww ELSS Details", use_container_width=True, help="Expense ratio & exit load"):
        handle_query_submission(EXAMPLE_QUESTIONS[0])
        st.rerun()

with col2:
    if st.button("🔒 ELSS Lock-in Rules", use_container_width=True, help="Statutory 3-year lock-in"):
        handle_query_submission(EXAMPLE_QUESTIONS[1])
        st.rerun()

with col3:
    if st.button("📄 Capital Gains Guide", use_container_width=True, help="Download tax statement"):
        handle_query_submission(EXAMPLE_QUESTIONS[2])
        st.rerun()

st.markdown("---")

# Render Conversation Stream
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    else:
        with st.chat_message("assistant", avatar="📈"):
            status = msg.get("status", "factual")
            content = msg["content"]
            
            if status == "refusal_pii":
                st.error(f"🛡️ **Privacy Guardrail Triggered:**\n\n{content}")
            elif status == "refusal_advice":
                st.warning(f"⚖️ **Facts-Only Policy:**\n\n{content}")
            elif status == "refusal_performance":
                st.info(f"📈 **No Performance Claims:**\n\n{content}")
            else:
                st.markdown(content)
            
            # Metadata drawer
            if msg.get("source_url") or msg.get("latency_ms"):
                with st.expander("ℹ️ Source & Retrieval Metadata", expanded=False):
                    if msg.get("source_url"):
                        st.markdown(f"**Primary Citation:** [{msg.get('source_title', 'Official Source')}]({msg['source_url']})")
                    st.caption(f"Status: `{status}` | Latency: `{msg.get('latency_ms', 0)} ms`")

# Chat Input Box
user_prompt = st.chat_input("Ask a factual question about Groww mutual funds, charges, or statements...")
if user_prompt:
    handle_query_submission(user_prompt)
    st.rerun()

# Sidebar: Controls, Scope, and Sources Explorer
with st.sidebar:
    st.markdown("### ⚙️ Session Controls")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
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
