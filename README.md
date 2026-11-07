# Groww Mutual Fund Facts-Only FAQ Assistant 📈

An academic research prototype for a facts-only RAG (Retrieval-Augmented Generation) FAQ Assistant for **Groww Mutual Fund**. It answers factual questions about mutual fund schemes—such as expense ratios, exit loads, minimum SIP, statutory lock-in (ELSS), riskometers, benchmarks, and statement downloads—using exclusively verified public disclosures.

---

## 🎯 Scope of Corpus

* **Target Product:** **Groww** (`groww.in`)
* **Asset Management Company:** **Groww Mutual Fund** (`growwmf.in` - Groww Asset Management Limited)
* **5 Scoped Schemes:**
  1. **Groww ELSS Tax Saver Fund** (Equity / ELSS — 3-year statutory lock-in, 0% exit load, 1.47% expense ratio)
  2. **Groww Nifty Total Market Index Fund** (Index / Flexi Cap — 0.25% exit load if redeemed ≤7 days, 0.66% expense ratio)
  3. **Groww Multicap Fund** (Equity / Multi Cap — 1% exit load if redeemed ≤1 year, 0.98% expense ratio)
  4. **Groww Liquid Fund** (Debt / Liquid — 7-day graded exit load, 0.11% expense ratio)
  5. **Groww Banking & Financial Services Fund** (Equity / Sectoral — 1% exit load if redeemed ≤30 days, 1.42% expense ratio)
* **Corpus Sources (33 Public Regulatory & Platform Pages):**
  * Groww AMC Statutory KIM/SID and Factsheets (`growwmf.in`)
  * Groww Platform Help & Statements Guides (`groww.in/blog`, `groww.in/help`, `groww.in/pricing`)
  * SEBI Regulations on Risk-o-meter, Categorization & TRI Benchmarking (`sebi.gov.in`, `investor.sebi.gov.in`)
  * AMFI Master Guides & CAS Retrieval Portals (`amfiindia.com`, `portal.amfiindia.com`)

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
  Explanations are strictly **≤ 3 sentences** and conclude with an auditable source timestamp: `Last updated from sources: YYYY-MM-DD`.

---

## 🚀 Quickstart Setup

### Prerequisites
* Python 3.10+

### Setup Steps
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

*(Optional: Copy `.env.example` to `.env` if you wish to configure an external inference provider like Groq, DeepSeek, or OpenAI. By default, the app runs completely offline using its built-in grounded factual extractor).*

---

## 📁 Repository Structure

```
elss_chatbot/
├── sources/                          # 33 Downloaded official sources
│   ├── groww_scheme_elss_tax_saver.md
│   ├── groww_scheme_nifty_total_market.md
│   ├── groww_guide_capital_gains_statement.md
│   ├── sources_catalog.json          # Machine-readable catalog
│   ├── sources_catalog.csv           # Required CSV deliverable
│   └── sources_catalog.md            # Markdown catalog with hyperlinks
├── rag/
│   ├── guardrails.py                 # PII and advice/performance filter
│   ├── retriever.py                  # Hybrid BM25/TF-IDF source retriever
│   ├── generator.py                  # Facts-only generator with offline fallback
│   └── engine.py                     # Unified RAG coordinator
├── tests/                            # Automated test suite (49 tests)
├── app.py                            # Streamlit interactive web UI
├── cli.py                            # Command-line interface and evaluator
├── sample_qa.md                      # Evaluated Q&A pairs (markdown)
├── sample_qa.json                    # Evaluated Q&A pairs (json)
├── disclaimer.md                     # Official disclaimer snippet & disclosures
└── requirements.txt                  # Python dependencies
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
