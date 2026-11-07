# Groww Mutual Fund Facts-Only FAQ Assistant 📈

An academic research prototype for a facts-only RAG (Retrieval-Augmented Generation) FAQ Assistant for **Groww Mutual Fund**. It answers factual questions about mutual fund schemes—such as expense ratios, exit loads, minimum SIP, statutory lock-in (ELSS), riskometers, benchmarks, and statement downloads—using exclusively verified public disclosures.

---

## 🎯 Scope of Corpus

* **Target Product:** **Groww** (`groww.in`)
* **Asset Management Company:** **Groww Mutual Fund** (`growwmf.in` - Groww Asset Management Limited)
* **Comprehensive Scheme Catalog & Core Scoped Offerings:**
  * **Complete Scheme Directory:** [Groww Mutual Fund Schemes Directory](sources/groww_amc_schemes_directory.md) covering all active equity, passive index, ETF FoF, and debt/liquid funds managed by Groww AMC.
  * **1. Groww ELSS Tax Saver Fund** (Equity / ELSS — 3-year statutory lock-in, Nil exit load, 1.47% expense ratio)
  * **2. Groww Nifty Total Market Index Fund** (Index / Flexi Cap — 0.25% exit load if redeemed ≤7 days, 0.66% expense ratio)
  * **3. Groww Multicap Fund** (Equity / Multi Cap — 1% exit load if redeemed ≤1 year, 0.98% expense ratio)
  * **4. Groww Liquid Fund** (Debt / Liquid — 7-day graded exit load, 0.11% expense ratio)
  * **5. Groww Banking & Financial Services Fund** (Equity / Sectoral — 1% exit load if redeemed ≤30 days, 1.42% expense ratio)
* **Corpus Sources (47 Public Regulatory & Platform Documents):**
  * Groww AMC Complete Schemes Directory, KIM/SID, and Factsheets (`growwmf.in`)
  * Groww Platform Help, Concepts, Taxation, and Statements Guides (`groww.in/blog`, `groww.in/p`, `groww.in/help`)
  * Comprehensive Mutual Fund Basics & Glossary taxonomy (AUM, NAV, TER, CAGR, XIRR, CAS, Cut-off times, Direct vs Regular, Growth vs IDCW, SIP vs Lump sum, KYC & Nomination)
  * SEBI Regulations & Investor Education Guides (`sebi.gov.in`, `investor.sebi.gov.in`)
  * AMFI Master Guides, Definitions & CAS Retrieval Portals (`amfiindia.com`, `portal.amfiindia.com`)

---

## 🏗️ Tri-Modal Hybrid GraphRAG Architecture

The system employs a multi-tiered retrieval pipeline:
1. **Dense Semantic Vector Stream (ChromaDB ONNX):** Embedded local vector store using `all-MiniLM-L6-v2` (384 dimensions) for semantic paraphrasing and natural question intent matching.
2. **Lexical BM25F Stream:** Exact numerical weighting and field-aligned scoring for precise financial parameters (expense ratios, lock-ins, cutoff times).
3. **Knowledge Graph Stream (NetworkX):** Directed graph modeling entities (Schemes, AMCs, Benchmarks, Lock-in periods, Tax Sections) and relational edges (`MANAGES`, `BENCHMARKED_TO`, `HAS_LOCK_IN`).
4. **Calibrated Reciprocal Rank Fusion (RRF):** Fuses dense, lexical, and graph streams with candidate injection and scheme entity prioritization.

---

## 🚀 Key Guardrails & Constraints

* **Strictly Facts-Only (Zero Investment Advice):**
  Politely refuses opinionated, portfolio, or speculative questions (e.g., *"Should I buy/sell?"*, *"Which fund is best?"*) and directs users to official AMFI investor education.
* **Zero PII Policy:**
  Input guardrails actively inspect and reject any sensitive personal information (PAN, Aadhaar, account numbers, OTPs, phone numbers, emails) before processing.
* **No Performance Claims:**
  Never calculates, projects, or compares returns or CAGR. Automatically redirects users to official scheme factsheets.
* **Verifiable Citations:**
  Every answer includes **exactly one official markdown link** to the primary public document supporting the fact.
* **Brevity & Transparency:**
  Explanations are strictly **≤ 10 sentences** and conclude with an auditable source timestamp: `Last updated from sources: YYYY-MM-DD`.

---

## ⚡ Multi-Provider LLM Engine & Dynamic Auto-Reconciliation

The generation engine supports multi-provider LLM inference with automatic cross-provider model reconciliation:
* **Production Mode (Groq):** Powered by `llama-3.3-70b-versatile` with high-throughput inference and full 2,500-character context chunks.
* **Google Gemini:** `gemini-2.0-flash` via OpenAI-compatible endpoint.
* **DeepSeek:** `deepseek-chat` with dynamic parameter correction.
* **OpenAI:** `gpt-4o-mini`.
* **Grounded Offline Fallback:** Fully deterministic extractive engine using regex attribute parsers and BM25 sentence density ranking, operating with zero external API calls or network dependencies.

---

## 🚀 Quickstart Setup

### Prerequisites
* Python 3.10+
* Docker & Docker Compose (optional for container deployment)

### Local Setup
```bash
# 1. Clone repository
git clone <repo-url> elss_chatbot
cd elss_chatbot

# 2. Set up virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Launch the Assistant UI
streamlit run app.py
```
The application will be live at `http://localhost:8501`.

### Docker Compose Deployment
```bash
# 1. Build and launch container in background
docker compose up -d --build

# 2. View container logs
docker compose logs -f
```

*(Optional: Set `GROQ_API_KEY`, `DEEPSEEK_API_KEY`, or `GEMINI_API_KEY` in `.env` to enable live LLM synthesis. By default, the app runs completely offline using its built-in grounded factual extractor).*

---

## 🔄 Automated Overnight Sync & Re-indexing Pipeline

The repository includes a self-healing automated overnight synchronization and regression testing pipeline:
* **Master Orchestrator:** [`scripts/overnight_update.py`](scripts/overnight_update.py)
* **Shell Wrapper:** [`scripts/cron_sync_sources.sh`](scripts/cron_sync_sources.sh)
* **Automated Cycle:**
  1. Creates an atomic snapshot backup (`.backups/snapshot_YYYYMMDD_HHMMSS`) of `sources/` and `vector_store/`.
  2. Cleans source markdown and strips navigation artifacts (`scripts/clean_sources.py`).
  3. Rebuilds metadata catalogs (`scripts/rebuild_catalog.py`).
  4. Re-indexes the Hybrid GraphRAG pipeline (Knowledge Graph and ChromaDB embeddings).
  5. Executes the full 121-test regression test suite (`pytest tests/`).
  6. Signals running applications via `.reload_trigger` to hot-reload state.
  7. **Atomic Rollback:** If any step fails, the system automatically restores the previous snapshot.

### Scheduling via Linux Crontab
Add to crontab (`crontab -e`):
```bash
0 2 * * * /home/chris/Project/elss_chatbot/scripts/cron_sync_sources.sh
```

---

## 📁 Repository Structure

```
elss_chatbot/
├── docs/
│   └── architecture.md               # Detailed RAG pipeline architecture & system design
├── sources/                          # 47 Verified official sources
│   ├── groww_amc_schemes_directory.md# Complete directory of schemes under Groww AMC
│   ├── groww_scheme_elss_tax_saver.md
│   ├── groww_scheme_nifty_total_market.md
│   ├── groww_guide_capital_gains_statement.md
│   ├── sources_catalog.json          # Machine-readable catalog (47 items)
│   ├── sources_catalog.csv           # Required CSV deliverable (47 items)
│   └── sources_catalog.md            # Markdown catalog with hyperlinks (47 items)
├── rag/
│   ├── guardrails.py                 # PII and advice/performance filter
│   ├── retriever.py                  # Field-weighted BM25F lexical retriever
│   ├── chroma_store.py               # Local ChromaDB persistent dense vector store
│   ├── knowledge_graph.py            # NetworkX in-memory knowledge graph engine
│   ├── hybrid_retriever.py           # Tri-Modal hybrid RRF retriever
│   ├── generator.py                  # Facts-only generator with multi-provider LLM & offline fallback
│   └── engine.py                     # Unified RAG coordinator
├── scripts/
│   ├── overnight_update.py           # Master automated sync orchestrator with rollback
│   ├── cron_sync_sources.sh          # Linux crontab execution wrapper
│   ├── clean_sources.py              # Source sanitizer and banner stripper
│   └── rebuild_catalog.py            # Catalog synchronization utility
├── vector_store/
│   └── chroma/                       # Persistent local ChromaDB collection (180 passages)
├── tests/                            # Automated test suite (121 tests, 100% passing)
├── app.py                            # Streamlit interactive web UI
├── cli.py                            # Command-line interface and evaluator
├── Dockerfile                        # Self-hosting container definition
├── docker-compose.yml                # Docker Compose multi-environment configuration
├── sample_qa.md                      # Evaluated Q&A pairs (markdown)
├── sample_qa.json                    # Evaluated Q&A pairs (json)
├── disclaimer.md                     # Official disclaimer snippet & disclosures
└── requirements.txt                  # Python dependencies
```

---

## 🧪 Testing & Verification

The suite includes **121 automated unit, integration, and offline evaluation tests** passing with 100% compliance:
```bash
# Run complete test suite
./venv/bin/pytest tests/ -v

# Run headless CLI evaluation
./venv/bin/python cli.py --eval

# Query specific question
./venv/bin/python cli.py --query "what are schemes under groww amc"
```

---

## ⚠️ Known Limits

1. **Facts-Only Boundary:** The assistant will not evaluate personal asset allocation, risk profile suitability, or recommend portfolio changes.
2. **Factsheet Frequency:** Source scheme metrics (TER, AUM, exit load) reflect official monthly AMC and platform disclosures; intra-day NAV fluctuations are not computed locally.
3. **Scope Constraint:** Coverage is focused on Groww AMC schemes and official Groww / AMFI / SEBI procedures. External AMCs outside Groww's ecosystem are directed to AMFI CAS portals.

---

## 📜 Disclaimer & Academic Disclosure

### Mandatory UI Snippet
```text
Facts-only. No investment advice. Academic project.
```

### Full Legal & Academic Disclosure
> **Academic Project Notice:** This application is strictly an independent **academic and educational research project** developed solely for prototype study and research evaluation of facts-only Retrieval-Augmented Generation (RAG) guardrails.
>
> **No Affiliation or Association:** This project, its authors, and contributors are **not associated, affiliated, authorized, endorsed, sponsored, or in any way officially connected** with **Groww** (Nextbillion Technology Pvt. Ltd.), **Groww Mutual Fund** (Groww Asset Management Limited), the **Securities and Exchange Board of India (SEBI)**, the **Association of Mutual Funds in India (AMFI)**, or any of their subsidiaries or affiliates. All product names, scheme titles, and trademarks are the property of their respective owners.
>
> **No Investment Advice (Facts-Only):** This assistant provides factual information derived exclusively from public regulatory disclosures. It does not provide investment advice, financial planning, or buy/sell recommendations. Mutual fund investments are subject to market risks; read all scheme-related documents carefully.
>
> **Disclaimer of Liability & Misuse:** Neither the authors nor contributors shall be held liable for any misuse, financial decisions, losses, or damages resulting from the use of or reliance on this academic prototype. Users are solely responsible for independently verifying all scheme facts against official regulatory sources. No personally identifiable information (PII) is accepted or stored.
