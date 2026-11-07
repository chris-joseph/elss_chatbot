"""
app.py
Adaptive, Mobile-Friendly Streamlit UI for Groww Mutual Fund Facts-Only FAQ Assistant.
Theme: Orange, Cream, and Navy Blue.
Features conversational chat (st.chat_message & st.chat_input), touch-friendly targets,
and fluid responsive typography.
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
    initial_sidebar_state="collapsed"
)

# Custom Adaptive CSS: Orange, Cream, and Navy Blue Theme
st.markdown("""
<style>
    /* Theme Palette Variables */
    :root {
        --navy-bg: #0A192F;         /* Deep Midnight Navy */
        --navy-surface: #112240;    /* Elevated Navy Card */
        --navy-border: #233554;     /* Navy Border / Divider */
        --cream-light: #FFFDF9;     /* Pure Warm Cream */
        --cream-text: #F8F3E6;      /* High-contrast Cream Typography */
        --cream-muted: #D3C5B4;     /* Secondary Muted Cream */
        --orange-primary: #FF6B35;  /* Vibrant Sunset Orange */
        --orange-hover: #FF8552;    /* Warm Hover Orange */
        --orange-glow: rgba(255, 107, 53, 0.2);
        --orange-badge-bg: rgba(255, 107, 53, 0.12);
    }
    
    /* Main Background & Text */
    .stApp {
        background-color: var(--navy-bg);
        color: var(--cream-text);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Responsive Header Container */
    .header-box {
        text-align: center;
        padding: clamp(16px, 4vw, 28px) clamp(12px, 3vw, 20px);
        background: linear-gradient(180deg, rgba(255, 107, 53, 0.14) 0%, rgba(17, 34, 64, 0.5) 100%);
        border: 1px solid var(--navy-border);
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    
    .header-title {
        font-size: clamp(22px, 5.5vw, 30px);
        font-weight: 800;
        color: var(--cream-light);
        margin-bottom: 8px;
        line-height: 1.25;
        letter-spacing: -0.3px;
    }
    
    .header-title span {
        color: var(--orange-primary);
        text-shadow: 0 0 20px rgba(255, 107, 53, 0.4);
    }
    
    .welcome-line {
        font-size: clamp(13px, 3.5vw, 15px);
        color: var(--cream-muted);
        margin-bottom: 12px;
        line-height: 1.5;
    }
    
    .disclaimer-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background-color: var(--orange-badge-bg);
        color: var(--orange-primary);
        border: 1px solid rgba(255, 107, 53, 0.4);
        padding: 5px 14px;
        border-radius: 20px;
        font-size: clamp(11px, 3vw, 13px);
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    
    /* Mobile-Friendly Adaptive Example Buttons */
    .stButton > button {
        border-radius: 10px;
        border: 1px solid var(--navy-border);
        background: linear-gradient(180deg, #172A45 0%, #112240 100%);
        color: var(--cream-text);
        font-size: clamp(12px, 3vw, 13.5px);
        font-weight: 500;
        padding: 10px 14px;
        min-height: 46px;
        width: 100%;
        transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    .stButton > button:hover {
        border-color: var(--orange-primary);
        color: var(--orange-primary);
        background: linear-gradient(180deg, #1E3A5F 0%, #172A45 100%);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px var(--orange-glow);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    /* Links in Orange */
    a {
        color: var(--orange-primary) !important;
        text-decoration: underline;
        font-weight: 600;
        transition: color 0.2s ease;
    }
    a:hover {
        color: var(--orange-hover) !important;
        text-shadow: 0 0 10px var(--orange-glow);
    }

    /* Chat Messages - Navy & Cream Surfaces */
    [data-testid="stChatMessage"] {
        background-color: var(--navy-surface);
        border: 1px solid var(--navy-border);
        border-radius: 12px;
        padding: clamp(12px, 3vw, 16px);
        margin-bottom: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }

    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(90deg, rgba(23, 42, 69, 0.7) 0%, rgba(17, 34, 64, 0.9) 100%);
        border-left: 3px solid var(--orange-primary);
    }

    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: var(--navy-surface) !important;
        color: var(--cream-muted) !important;
        border-radius: 8px !important;
        font-size: 13px !important;
    }

    /* Sidebar Navigation in Navy & Cream */
    [data-testid="stSidebar"] {
        background-color: #081426;
        border-right: 1px solid var(--navy-border);
    }
    [data-testid="stSidebar"] h3 {
        color: var(--orange-primary);
        font-size: 16px;
        font-weight: 700;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
        color: var(--cream-muted);
        font-size: 13.5px;
    }

    /* Mobile Responsive Media Queries */
    @media (max-width: 640px) {
        .header-box {
            padding: 16px 10px;
            margin-bottom: 14px;
        }
        .header-title {
            font-size: 22px;
        }
        .stButton > button {
            min-height: 48px; /* Touch target accessibility */
            margin-bottom: 6px;
        }
        [data-testid="stChatMessage"] {
            padding: 10px 12px;
            font-size: 14px;
        }
    }
</style>
""", unsafe_allow_html=True)

# Cache RAG Engine instance
@st.cache_resource
def get_engine():
    return RAGEngine()

engine = get_engine()

# Initialize Chat & Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar: Controls, LLM Status, Scope, and Sources Explorer
with st.sidebar:
    st.markdown("### ⚡ LLM Engine Status")
    llm_status = engine.get_llm_status()
    st.markdown(f"<p style='font-size: 13.5px; font-weight: 600; color: #FF6B35; margin-bottom: 2px;'>{llm_status['status_label']}</p>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🧹 Session Controls")
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
    st.markdown("### 🎓 Academic Disclaimer")
    st.markdown("""
    * **Academic Project**: Developed strictly as an educational and research prototype.
    * **No Affiliation**: Not associated, affiliated, authorized, or endorsed by **Groww**, **SEBI**, **AMFI**, or any AMC.
    * **Zero Liability**: Not responsible or liable for any misuse of this platform or reliance on its outputs.
    * **Verification**: Always verify facts against primary official regulatory disclosures.
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

# Header & Welcome line with Orange, Cream, and Navy styling
st.markdown("""
<div class="header-box">
    <div class="header-title">Groww Mutual Fund <span>FAQ Assistant</span></div>
    <div class="welcome-line">Welcome! I am your facts-only guide for mutual fund schemes, charges, exit loads, and statement guides.</div>
    <div class="disclaimer-badge">🔒 Facts-only. No investment advice. Academic Project.</div>
</div>
""", unsafe_allow_html=True)

# Example Questions Configuration
EXAMPLE_QUESTIONS = [
    "What is the expense ratio and exit load of Groww ELSS Tax Saver Fund?",
    "What is the mandatory lock-in period for ELSS mutual funds?",
    "How do I download my Mutual Fund capital gains statement on Groww?"
]

st.markdown("<p style='font-size: 14px; font-weight: 600; color: #D3C5B4; margin-bottom: 8px;'>💡 Quick Example Questions:</p>", unsafe_allow_html=True)

# Adaptive 3-Column / Mobile Stacked Button Grid
col1, col2, col3 = st.columns([1, 1, 1])

def handle_query_submission(query_text: str):
    """Processes user query cleanly and adds to chat history."""
    clean_q = query_text.strip()
    if not clean_q:
        return
        
    # Append user question
    st.session_state.messages.append({"role": "user", "content": clean_q})
    
    # Run RAG Engine with active LLM configuration
    res = engine.answer(clean_q)
    
    # Append assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": res["answer"],
        "status": res.get("status", "factual"),
        "source_url": res.get("source_url"),
        "source_title": res.get("source_title"),
        "latency_ms": res.get("latency_ms", 0.0)
    })

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

st.markdown("<hr style='border: none; border-top: 1px solid #233554; margin: 16px 0;'>", unsafe_allow_html=True)

# Render Chat Stream
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"<span style='color: #F8F3E6;'>{msg['content']}</span>", unsafe_allow_html=True)
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

# Chat Input Box (Fixed at bottom, mobile-friendly)
user_prompt = st.chat_input("Ask a factual question about Groww mutual funds, charges, or statements...")
if user_prompt:
    handle_query_submission(user_prompt)
    st.rerun()

# Persistent Academic & Legal Disclaimer Footer
st.markdown("""
<div style='text-align: center; margin-top: 40px; padding: 18px 12px; border-top: 1px solid #233554; font-size: 11.5px; color: #8892B0; line-height: 1.6;'>
    🎓 <b>Academic Research Prototype</b> &bull; Facts-only. No investment advice.<br>
    This project is not affiliated with, endorsed by, or associated with Groww, SEBI, AMFI, or any mutual fund house.<br>
    The authors and maintainers are not responsible for any misuse of this platform or reliance on its outputs. Always verify facts with official regulatory filings.
</div>
""", unsafe_allow_html=True)
