# Groww Mutual Fund Facts-Only FAQ Assistant 📈

A lightweight, production-ready, facts-only RAG (Retrieval-Augmented Generation) FAQ Assistant for **Groww Mutual Fund**. It answers queries regarding mutual fund scheme facts (expense ratio, exit load, minimum SIP, lock-in, riskometer, benchmark, cut-off timings, and statement downloads) exclusively using official public pages from Groww AMC, Groww Platform, AMFI, and SEBI.

---

## 🚀 Key Features & Constraints

* **Strictly Facts-Only (Zero Investment Advice):**
  Refuses opinionated, portfolio, or speculative questions (e.g., *"Should I buy/sell?"*, *"Which fund is best?"*) with a polite educational refusal linking to official AMFI investor education.
* **No PII Policy:**
  Input guardrails actively inspect and reject any sensitive PII (PAN, Aadhaar, account numbers, OTPs, phone numbers, emails) before RAG processing.
* **No Performance Claims:**
  Never calculates, projects, or compares returns/CAGR. Automatically directs users to official scheme factsheets.
* **Verifiable Source Citations:**
  Every answer includes **exactly one official markdown link** to the primary public document supporting the fact.
* **Brevity & Transparency:**
  Explanations are strictly **≤ 3 sentences** and conclude with an auditable timestamp: `Last updated from sources: YYYY-MM-DD`.
* **Configurable LLM Inference (Groq, DeepSeek, OpenAI, Ollama):**
  Supports any OpenAI-compatible provider. A **Groq API Key** (`GROQ_API_KEY`) is highly recommended: running `llama-3.3-70b-versatile` or `llama-3.1-8b-instant` at 300–500 tokens/sec on LPUs with a generous free tier.
* **Zero-Failure Fallback:**
  Includes an offline deterministic grounded extraction engine so it functions 100% reliably even without an internet connection or API key.
* **Adaptive Responsive UI (Orange, Cream, Navy Blue):**
  Polished, mobile-friendly Streamlit web app with dynamic typography, touch-friendly touch targets, live LLM switching in the sidebar, and chat history controls.

---

## 🎯 Scope of Corpus

* **Chosen Product:** **Groww** (`groww.in`)
* **Chosen Asset Management Company:** **Groww Mutual Fund** (`growwmf.in` - Groww Asset Management Limited)
* **5 Scoped Schemes:**
  1. **Groww ELSS Tax Saver Fund** (Equity / ELSS — 3-year statutory lock-in, 0% exit load, 1.47% expense ratio)
  2. **Groww Nifty Total Market Index Fund** (Index / Flexi Cap — 0.25% exit load if redeemed ≤7 days, 0.66% expense ratio)
  3. **Groww Multicap Fund** (Equity / Multi Cap — 1% exit load if redeemed ≤1 year, 0.98% expense ratio)
  4. **Groww Liquid Fund** (Debt / Liquid — 7-day graded exit load, 0.11% expense ratio)
  5. **Groww Banking & Financial Services Fund** (Equity / Sectoral — 1% exit load if redeemed ≤30 days, 1.42% expense ratio)
* **Corpus Sources (33 Public Pages):**
  * Groww AMC Statutory KIM/SID and Factsheets (`growwmf.in`)
  * Groww Platform Help & Statements Guides (`groww.in/blog`, `groww.in/help`, `groww.in/pricing`)
  * SEBI Regulations on Risk-o-meter, Categorization & TRI Benchmarking (`sebi.gov.in`, `investor.sebi.gov.in`)
  * AMFI Master Guides & CAS Retrieval Portals (`amfiindia.com`, `portal.amfiindia.com`)

---

## 🛠️ Project Structure

```
elss_chatbot/
├── sources/                          # 33 Downloaded official sources
│   ├── groww_scheme_elss_tax_saver.md
│   ├── groww_scheme_nifty_total_market.md
│   ├── groww_guide_capital_gains_statement.md
│   ├── sebi_circular_riskometer_labeling.md
│   ├── amfi_faq_cas_pdf.md
│   ├── sources_catalog.json          # Machine-readable catalog
│   ├── sources_catalog.csv           # Required CSV deliverable
│   └── sources_catalog.md            # Markdown catalog with hyperlinks
├── rag/
│   ├── __init__.py
│   ├── guardrails.py                 # PII and advice/performance filter
│   ├── retriever.py                  # Hybrid BM25/TF-IDF source retriever
│   ├── generator.py                  # Facts-only generator with DeepSeek & offline fallback
│   └── engine.py                     # Unified RAG coordinator
├── tests/
│   ├── test_guardrails.py            # Unit tests for guardrails
│   ├── test_retriever.py             # Unit tests for retrieval accuracy
│   └── test_rag_pipeline.py          # End-to-end integration tests
├── app.py                            # Streamlit tiny interactive web UI
├── cli.py                            # Interactive CLI and headless tester
├── sample_qa.md                      # Evaluated Q&A pairs (markdown)
├── sample_qa.json                    # Evaluated Q&A pairs (json)
├── disclaimer.md                     # Official disclaimer snippet
├── Dockerfile                        # Self-hosting Docker container
├── docker-compose.yml                # Self-hosting Docker Compose file
├── run.sh                            # One-click startup script
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## ⚙️ Configurable LLM Providers

The assistant supports plug-and-play LLM providers configured either via environment variables (`.env`) or directly within the Streamlit UI sidebar:

| Provider | Model | API Key Environment Variable | Why Use It? |
| :--- | :--- | :--- | :--- |
| **Groq (Recommended)** | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | Ultra-fast (300-500 tokens/sec), generous free tier, state-of-the-art Llama 3.3 70B accuracy. |
| **DeepSeek** | `deepseek-chat` | `DEEPSEEK_API_KEY` | Cost-effective, high reasoning capability. |
| **OpenAI** | `gpt-4o-mini` | `OPENAI_API_KEY` | Industry standard JSON & instruction-following accuracy. |
| **Ollama (Local)** | `llama3.1` | None required | 100% private, local inference without external network calls. |
| **Offline Fallback** | Deterministic Engine | None required | Zero-dependency, regex & keyword grounded extractor guaranteeing 100% uptime. |

### Does a Groq API Key help?
**Yes, significantly!**
1. **Speed:** Groq's LPU inference engines process at 300–500 tokens per second, making RAG generation near-instantaneous (<1 second round-trip).
2. **Cost:** Groq provides a generous free tier with zero setup cost.
3. **Accuracy:** You can run `llama-3.3-70b-versatile` which excels at following the strict 3-sentence, single-citation, and facts-only guardrails.
4. **Ease of Setup:** Paste your `gsk_...` key in the Streamlit sidebar or set `GROQ_API_KEY=gsk_...` in `.env`.

---

## 💻 Self-Hosting Instructions

### Option 1: Native Run with Python Virtual Environment

```bash
# 1. Clone repository
git clone <repo-url> elss_chatbot
cd elss_chatbot

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Configure environment variables
cp .env.example .env
# Edit .env to add your DEEPSEEK_API_KEY if desired

# 5. Run tests
pytest tests/ -v

# 6. Launch the Streamlit Web Application
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```
Open `http://localhost:8501` in your browser.

---

### Option 2: Self-Hosting with Docker & Docker Compose

```bash
# Start container using Docker Compose
docker compose up -d --build

# View container logs
docker compose logs -f

# Stop container
docker compose down
```

---

### Option 3: Command-Line Interface (CLI Mode)

```bash
# Interactive Chat in Terminal
./venv/bin/python cli.py

# Run Automated Evaluation Suite
./venv/bin/python cli.py --eval
```

---

## 🧪 Verification & Testing

Run the full automated test suite:
```bash
./venv/bin/pytest tests/ -v
```
All 44 test cases validate:
* PAN, Aadhaar, Phone, Email, OTP, and Bank Account privacy blocking.
* Buy/sell opinion & portfolio advice refusals with AMFI education links.
* Return calculations and CAGR prediction refusals with Factsheet links.
* Exact citation link formatting and ≤3 sentences length constraint.
* Keyword & semantic retrieval accuracy on all 5 scoped Groww schemes.
* Groq, DeepSeek, OpenAI, Ollama, and offline generator configurations.

---

## ⚠️ Known Limits

1. **Facts-Only Boundary:** The assistant will not evaluate personal asset allocation, risk profile suitability, or recommend portfolio changes.
2. **Factsheet Frequency:** Source scheme metrics (TER, AUM, exit load) reflect official monthly AMC and platform disclosures; intra-day NAV fluctuations are not computed locally.
3. **Scope Constraint:** Coverage is focused on Groww AMC schemes and official Groww / AMFI / SEBI procedures. External AMCs outside Groww's ecosystem are directed to AMFI CAS portals.

---

## 📜 Disclaimer Snippet

```text
Facts-only. No investment advice.
```

> **Legal Disclaimer:** This assistant provides factual information regarding mutual fund schemes, charges, exit loads, and regulatory procedures derived exclusively from official public disclosures. It does not provide investment advice or recommendations. Read all scheme-related documents carefully before investing. No PII is accepted or stored.
