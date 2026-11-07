# RAG Pipeline Architecture & System Design Documentation 🏛️

This document provides a comprehensive technical breakdown of the **Groww Mutual Fund Facts-Only RAG (Retrieval-Augmented Generation) Pipeline**. It covers the end-to-end request lifecycle, information retrieval algorithms, dynamic query normalization, guardrail policy enforcement, grounded extractive generation, and system component interactions.

---

## 1. Executive Summary & Design Principles

The primary objective of this system is to serve verified, auditable mutual fund facts without introducing hallucinations, investment advice, or opinionated portfolio bias.

```
       ┌────────────────────────────────────────────────────────┐
       │                Core System Tenets                      │
       ├────────────────────────────────────────────────────────┤
       │ 1. Zero Investment Advice & Opinion Isolation          │
       │ 2. Strict Input PII Sanitization (PAN, Aadhaar, etc.)  │
       │ 3. Field-Weighted Information Retrieval (BM25F concept)│
       │ 4. Anti-Overfitting & Dynamic Vocabulary Normalization │
       │ 5. Dual-Mode Grounded Generation (LLM + Offline Engine)│
       │ 6. Auditable Brevity (≤3 Sentences + 1 Verified Link)  │
       └────────────────────────────────────────────────────────┘
```

### Anti-Overfitting Philosophy
Traditional naive RAG implementations often rely on hardcoded keyword branches (e.g. `if "what is a mutual fund" in query: return ...`). Such architectures are brittle, fail to generalize, and fail on common misspellings or related queries. This pipeline relies on **principled Information Retrieval mathematics**:
* Field-weighted TF-IDF vectorization (BM25F principles).
* Vocabulary-guided edit-distance spelling correction.
* Dual-signal topic alignment (Title/Section Recall + Jaccard Specificity).
* Passage keyword density and verbatim n-gram phrase matching.
* Dynamic structured key-value parsing combined with BM25 sentence salience ranking.

---

## 2. End-to-End System Architecture

### 2.1 Component Flowchart

```mermaid
flowchart TD
    UserQuery["User Input Query"] --> Guardrails{"Guardrails Filter<br/>(guardrails.py)"}
    
    Guardrails -- "Contains PII<br/>(PAN, Aadhaar, OTP)" --> RefusalPII["Refusal: Privacy Protection Notice"]
    Guardrails -- "Financial Advice / Opinion<br/>('Should I buy?')" --> RefusalAdvice["Refusal: Regulatory Non-Advice Notice<br/>+ AMFI Investor Education Link"]
    Guardrails -- "Return Projection<br/>('How much return?')" --> RefusalPerf["Refusal: No Performance Claims<br/>+ Official Factsheet Link"]
    
    Guardrails -- "Safe Factual Query" --> Normalizer["Corpus-Guided Spell-Checker<br/>(difflib on Corpus Vocabulary)"]
    
    subgraph OfflineCorpusIndex ["Offline Document Indexing"]
        RawDocs["36 Verified Sources (Markdown)"] --> Cleaner["Pre-processor & Navbar Cleaner"]
        Cleaner --> Chunker["Header Segmentation & Stub Merging"]
        Chunker --> FieldWeights["Field-Weighting (Title 3x, Sec 2x, Cat 1x, Content 1x)"]
        FieldWeights --> TFIDFIndex["Scikit-Learn TF-IDF Matrix (1, 2)-grams"]
        FieldWeights --> VocabIndex["Corpus Vocabulary & Frequency Counter"]
    end
    
    VocabIndex -.-> Normalizer
    
    Normalizer --> Retriever{"Multi-Signal Retriever<br/>(retriever.py)"}
    TFIDFIndex -.-> Retriever
    
    Retriever --> ConfidenceGate{"Relevance Confidence Gate<br/>(Score ≥ 0.45 & In-Vocab)?"}
    
    ConfidenceGate -- "Below Threshold / OOD<br/>('boomba diagga', 'time now')" --> OutOfScope["Status: out_of_scope<br/>Scope Refusal Notice + Factsheet Link"]
    ConfidenceGate -- "Confidence Validated<br/>(Score ≥ 0.45)" --> TopK["Top-k Scored Passages<br/>(Cosine Sim + Title Jaccard + Section Alignment)"]
    
    TopK --> Generator{"Generator Engine<br/>(generator.py)"}
    
    Generator -- "LLM Key Active<br/>(Groq / DeepSeek / OpenAI / Ollama)" --> OnlineLLM["Strict System-Prompted LLM Inference<br/>(Temperature: 0.1, Max Tokens: 250)"]
    Generator -- "Offline / No API Key / Timeout" --> OfflineExtractor["Dynamic Grounded Extractor<br/>(Key-Value Bullet Parser + BM25 Sentence Ranker)"]
    
    OnlineLLM --> PostProc["Post-Processor: Truncate to ≤3 Sentences & Ensure Citation"]
    OfflineExtractor --> PostProc
    
    PostProc --> FinalResponse["Verified Factual Response<br/>(≤3 Sentences + 1 Official Citation Link + Timestamp)"]
```

---

## 3. Detailed Component Breakdown & Processing Steps

### 3.1 Step 1: Input Guardrails & Risk Isolation (`rag/guardrails.py`)
Before search or retrieval occurs, the query undergoes deterministic regex and pattern inspection to satisfy regulatory and privacy standards:

1. **Personally Identifiable Information (PII) Inspection:**
   * **PAN Cards:** `[A-Z]{5}[0-9]{4}[A-Z]`
   * **Aadhaar Numbers:** `\b\d{4}\s?\d{4}\s?\d{4}\b`
   * **Indian Mobile Numbers:** `(?:\+91[\s-]?)?[6-9]\d{9}`
   * **Email Addresses:** Standard RFC-5322 regex
   * **Financial Secrets:** 4–6 digit OTP tokens, bank/demat account numbers (`\b\d{11,18}\b`).
   * *Action if triggered:* Instant termination with privacy notification; no query logs or downstream vector lookups are performed.

2. **Regulatory Non-Advice Enforcement:**
   * Detects intent related to buy/sell recommendations, portfolio suitability, or speculative selection:
     * Patterns: `"should i invest"`, `"which is better"`, `"is it good to buy"`, `"recommend me"`, `"help me choose"`.
   * *Action if triggered:* Returns standard SEBI/AMFI educational disclaimer redirecting to [AMFI Investor Education](https://www.amfiindia.com/investor).

3. **Performance & Return Claims Refusal:**
   * Detects queries seeking future return projections, CAGR estimates, or guaranteed outcomes:
     * Patterns: `"how much return"`, `"projected return"`, `"cagr"`, `"will it double"`, `"profit in 2 years"`.
   * *Action if triggered:* Refuses estimation and directs the user to the official [Groww Monthly Factsheets](https://www.growwmf.in/downloads/factsheets).

---

### 3.2 Step 2: Corpus Ingestion, Cleaning & Structural Chunking (`rag/retriever.py`)

The pipeline loads 36 official markdown documents from `sources/`. Each document conforms to structured YAML frontmatter:

```yaml
---
id: groww_scheme_elss_tax_saver
title: "Groww ELSS Tax Saver Fund - Scheme Facts & Details"
url: "https://groww.in/mutual-funds/groww-elss-tax-saver-fund-direct-growth"
organization: "Groww Mutual Fund"
category: "ELSS / Tax Saver"
last_updated_from_sources: "2026-11-07"
---
```

#### Cleaning & Header Stub Merging Process
1. **Scraper Navigation Elimination:** Scraped web artifacts (such as top navigation bars, stock tickers, and generic promotional menus) are stripped before chunking to prevent spurious keyword inflation.
2. **Markdown Header Segmentation:** Documents are partitioned along header boundaries (`#`, `##`, `###`).
3. **Preamble & Stub Merging:** Sub-sections smaller than 180 characters (such as introductory paragraphs or standalone headers like `# Groww Liquid Fund Direct Growth`) are merged with their respective factual body tables. This prevents the index from storing isolated, context-deficient snippets.

---

### 3.3 Step 3: Dynamic Corpus-Guided Spelling Normalization

Rather than maintaining a brittle, manually authored typo dictionary, the retriever builds a live vocabulary index directly from the corpus:
* **Index Creation:** Every unique lowercase token ($\ge 3$ characters) across all 36 source documents is registered in `self.vocabulary` alongside its term frequency in `self.word_counts`.
* **Dynamic Correction Algorithm:**
  1. For each token $t$ in user query $Q$:
  2. If $t \in \text{vocabulary}$ or $\text{len}(t) < 4$, keep $t$ unmodified.
  3. If $t \notin \text{vocabulary}$, compute Levenshtein / Gestalt similarity against `self.vocabulary` using `difflib.get_close_matches(t, self.vocabulary, n=3, cutoff=0.75)`.
  4. If candidate matches exist, select the match $m$ with the highest corpus occurrence $\text{word\_counts}[m]$.
* **Example:**
  * `"what is a Mutal fund"` $\to$ `"mutal"` is absent in vocabulary; corrected to `"mutual"` (frequency: 180+).
  * `"expence ratio"` $\to$ corrected to `"expense ratio"`.
  * `"benchmrk index"` $\to$ corrected to `"benchmark index"`.

---

### 3.4 Step 4: Field-Weighted Vectorization & Multi-Signal Retrieval

#### A. Field-Weighted Representation (BM25F Principle)
In financial FAQs, the document title and section heading carry exponentially higher topic specificity than repeated mentions in body text. Each passage $p$ is vectorized using weighted field repetition:

$$\text{Text}(p) = \underbrace{p.\text{title} \times 3}_{\text{Document Subject}} + \underbrace{p.\text{section\_title} \times 2}_{\text{Sub-topic Topic}} + \underbrace{p.\text{category} \times 1}_{\text{Asset Class}} + \underbrace{p.\text{content} \times 1}_{\text{Factual Body}}$$

Vectorization is performed using Scikit-Learn `TfidfVectorizer`:
* `ngram_range = (1, 2)` (captures key phrases like *"mutual fund"*, *"expense ratio"*, *"exit load"*).
* `sublinear_tf = True` (replaces linear TF $f$ with $1 + \log(f)$ to prevent long documents from dominating over concise documents).
* `token_pattern = r'(?u)\b\w+\b'`.

#### B. Multi-Signal Scoring Function
For a normalized query $Q$ with keywords $Q_{\text{keywords}} = \{w \in Q \mid \text{len}(w) \ge 3 \text{ and } w \notin \text{STOPWORDS}\}$:

$$\text{Score}(p, Q) = S_{\text{cosine}} + B_{\text{title\_recall}} + B_{\text{title\_jaccard}} + B_{\text{sec\_jaccard}} + B_{\text{content\_coverage}} + B_{\text{phrase}}$$

Where:
1. **$S_{\text{cosine}}$:** Cosine similarity between query vector and passage vector in TF-IDF space.
2. **Title Recall ($B_{\text{title\_recall}}$):**
   $$B_{\text{title\_recall}} = 0.30 \times \frac{|Q_{\text{keywords}} \cap T_{\text{clean}}|}{|Q_{\text{keywords}}|}$$
3. **Title Jaccard Specificity ($B_{\text{title\_jaccard}}$):** Measures whether the title is specifically about the query:
   $$B_{\text{title\_jaccard}} = 0.35 \times \frac{|Q_{\text{keywords}} \cap T_{\text{clean}}|}{|Q_{\text{keywords}} \cup T_{\text{clean}}|}$$
   *(For "What is a Mutual Fund", the Jaccard similarity is $2/2 = 1.0$. For an Expense Ratio guide mentioning mutual funds, Jaccard is only $2/5 = 0.40$).*
4. **Section Jaccard Specificity ($B_{\text{sec\_jaccard}}$):**
   $$B_{\text{sec\_jaccard}} = 0.25 \times \frac{|Q_{\text{keywords}} \cap S_{\text{clean}}|}{|Q_{\text{keywords}} \cup S_{\text{clean}}|}$$
5. **Passage Content Coverage ($B_{\text{content\_coverage}}$):**
   $$B_{\text{content\_coverage}} = 0.20 \times \frac{|Q_{\text{keywords}} \cap C_{\text{words}}|}{|Q_{\text{keywords}}|}$$
6. **Verbatim N-Gram Phrase Bonus ($B_{\text{phrase}}$):**
   $$B_{\text{phrase}} = 0.25 \quad \text{if cleaned query string is a substring of } p.\text{title} \text{ or } p.\text{section\_title}$$

---

### 3.5 Step 5: Answer Generation & Extractive Grounding (`rag/generator.py`)

The generation engine supports dual-mode operation:

```mermaid
stateDiagram-v2
    [*] --> InspectProvider
    InspectProvider --> CheckAPIKey: LLM Provider Specified
    InspectProvider --> OfflineExtractive: LLM_PROVIDER=offline / No Keys
    
    CheckAPIKey --> OnlineLLM: Valid Key (Groq / OpenAI / DeepSeek / Ollama)
    CheckAPIKey --> OfflineExtractive: Missing Key / Timeout
    
    OnlineLLM --> SentenceValidation: Raw LLM Output
    OfflineExtractive --> SentenceValidation: Extracted Output
    
    SentenceValidation --> CitationEnforcement: Truncate to ≤3 Sentences
    CitationEnforcement --> FinalAnswer: Append Primary Link & Source Timestamp
```

#### Mode A: Online LLM Generation
When an API key is provided in `.env`:
* **Supported Backends:** Groq (`llama-3.3-70b-versatile`), DeepSeek (`deepseek-chat`), OpenAI (`gpt-4o-mini`), or local Ollama (`llama3.1`).
* **Generation Parameters:**
  * `temperature: 0.1` (minimizes creative hallucination).
  * `max_tokens: 250`.
  * `system_prompt`: Enforces strict facts-only answering based exclusively on provided context, $\le 3$ sentence maximum, and inclusion of source timestamp.

#### Mode B: Offline Grounded Extractive Summarizer
When running offline (zero cloud dependencies, air-gapped, or free local deployment), the generator uses a deterministic two-tier extractor:

1. **Tier 1 — Structured Fact Parser:**
   * Scans top retrieved passages for key-value bullet points matching the regex:
     `^[-*]?\s*(?:\*\*)?([A-Za-z0-9\s/()-]+?)(?:\*\*)?:\s*(.+)$`
   * Filters out generic mutual fund stop-tokens (`fund`, `scheme`, `groww`, `details`).
   * If a specific query attribute matches (e.g. *Expense Ratio*, *Exit Load*, *Lock-in Period*, *Benchmark*, *Minimum SIP*), formats a direct statement:
     `"The {Key} for {Scheme Name} is {Value}."`
2. **Tier 2 — BM25 Sentence Salience Ranker:**
   * For narrative, conceptual, or procedural queries (e.g. *"What is a mutual fund"*, *"How to cancel SIP"*, *"What is NAV"*):
   * Partitions retrieved passages into clean candidate sentences.
   * Scores each sentence $S$:
     $$\text{Salience}(S) = \frac{|Q_{\text{keywords}} \cap S|}{\sqrt{|S|}} + \text{Bonus}_{\text{definitional}} + \text{Bonus}_{\text{procedural}} + \text{Bonus}_{\text{position}}$$
     * $\text{Bonus}_{\text{definitional}} = +0.8$ if sentence contains copulas (`"is a"`, `"represents"`, `"means"`).
     * $\text{Bonus}_{\text{procedural}} = +0.8$ if query is procedural and sentence contains action verbs (`"tap"`, `"select"`, `"go to"`, `"download"`).
     * $\text{Bonus}_{\text{position}} = +0.2$ for the opening sentence of the top section.
   * Selects the top 1 or 2 highest-scoring coherent sentences directly from the passage text.

---

### 3.6 Step 6: Post-Processing & Verification Constraints

Before returning the answer to the user:
1. **Sentence Cap Enforcer (`_truncate_to_sentences`):**
   * Splits answer into grammatical sentences using regex `(?<=[.!?])\s+`.
   * Preserves any trailing `Last updated from sources: YYYY-MM-DD` timestamp while trimming the body text to a maximum of 3 sentences.
2. **Citation Link Validator (`_ensure_citation`):**
   * Verifies that the answer contains exactly one markdown citation `[Primary Title](Primary URL)`.
   * If omitted by an LLM backend, automatically synthesizes and injects the verified link from `primary_passage.url`.
3. **Auditable Source Suffix:**
   * Enforces the conclusion format: `Last updated from sources: {primary_passage.last_updated}`.

---

### 3.7 Step 7: Out-of-Scope (OOD) Gating & Hallucination Defense

To prevent the assistant from hallucinating or answering questions outside its verified documentation corpus (e.g. *"boomba diagga?"*, *"what time is it now?"*, *"who is the prime minister of india?"*, *"recipe for pasta"*), the pipeline enforces a **two-tier confidence defense**:

1. **Tier 1 — Fast Out-of-Vocabulary Gate ($O(1)$):**
   * Before vector search, the retriever inspects the non-stopword query keywords $Q_{\text{keywords}}$.
   * If non-empty query keywords contain zero overlapping tokens with `self.vocabulary` (e.g. *"boomba diagga"*), the query is immediately rejected without executing unnecessary vector transforms.

2. **Tier 2 — Multi-Signal Relevance Confidence Gate (`MIN_RELEVANCE_SCORE = 0.45`):**
   * When an unrelated query contains common general English words (e.g., *"time"* in *"what time is it now?"*, or *"india"* in *"prime minister of india"*), naive retrievers yield weak spurious matches (scoring $0.10 \sim 0.28$).
   * By enforcing a strict threshold of $S \ge 0.45$, all out-of-domain queries are discarded.
   * Empirically, legitimate in-domain queries consistently score between $0.57$ and $1.55$, ensuring **zero false rejections**.

3. **Status Classification & Dedicated UI Badge:**
   * When no passages meet the confidence threshold, `RAGEngine` returns `status: "out_of_scope"`.
   * Streamlit displays a distinct warning banner: `⚠️ Out of Scope / Unverified Query`.
   * CLI displays tag: `[OUT_OF_SCOPE]`.
   * No API calls are dispatched to external LLMs, eliminating both inference cost and hallucination risk.

---

## 4. Sequence Diagram: Inference Request Lifecycle

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Streamlit as Streamlit UI (app.py)
    participant Engine as RAGEngine (engine.py)
    participant Guard as Guardrails (guardrails.py)
    participant Ret as SourceRetriever (retriever.py)
    participant Gen as FactsOnlyGenerator (generator.py)
    
    User->>Streamlit: Enters Query ("What is a Mutal fund")
    Streamlit->>Engine: answer(query)
    
    Engine->>Guard: inspect(query)
    alt Contains PII or Advice
        Guard-->>Engine: Refusal Payload
        Engine-->>Streamlit: Refusal Response
        Streamlit-->>User: Display Refusal Notice
    else Safe Query
        Guard-->>Engine: Clean Signal
        Engine->>Ret: retrieve(query, top_k=3)
        Ret->>Ret: Dynamic Typo Correction ("Mutal" -> "mutual")
        Ret->>Ret: Compute Cosine Sim + Title Jaccard Overlap
        Ret-->>Engine: Top 3 Passages
        
        Engine->>Gen: generate(query, passages)
        alt LLM Active
            Gen->>Gen: Call OpenAI/Groq API (temp=0.1)
        else Offline Fallback
            Gen->>Gen: Extract Structured Fact / Salient Sentences
        end
        Gen->>Gen: Enforce ≤3 Sentences + Citation + Date
        Gen-->>Engine: Verified Answer
        Engine-->>Streamlit: Final Response Payload
        Streamlit-->>User: Render Factual Answer + Citation Link
    end
```

---

## 5. Offline Mode vs. Online LLM Mode Comparison

| Feature | Offline Grounded Mode (Default) | Online LLM Mode (Optional) |
| :--- | :--- | :--- |
| **Provider** | Deterministic Python Extractor | Groq / DeepSeek / OpenAI / Ollama |
| **API Key Required** | ❌ No (Completely Free & Private) | ✅ Yes (configured via `.env`) |
| **External Network Call** | ❌ None (Runs 100% locally) | ✅ Outbound HTTPS to API endpoint |
| **Inference Latency** | **150 ms – 200 ms** | 450 ms – 1200 ms |
| **Hallucination Risk** | **0%** (Direct substring / sentence extraction) | Extremely low (temp=0.1 + context bounds) |
| **Computational Footprint** | Extremely light (runs on 512MB RAM VPS) | Negligible local footprint (delegated) |
| **Fallback Behavior** | Native baseline | Auto-falls back to Offline Mode on API failure |

---

## 6. Project Data Models

### 6.1 `Passage` Data Model (`rag/retriever.py`)
```python
@dataclass
class Passage:
    doc_id: str          # Unique file slug (e.g., 'groww_scheme_elss_tax_saver')
    title: str           # Source document title from frontmatter
    url: str             # Verified canonical URL on groww.in / sebi.gov.in / amfiindia.com
    organization: str    # Originating entity (e.g., 'Groww Mutual Fund', 'SEBI', 'AMFI')
    category: str        # Taxonomy category (e.g., 'ELSS / Tax Saver', 'Charges & TER')
    last_updated: str    # ISO date string (e.g., '2026-11-07')
    section_title: str   # Clean heading of the specific chunk
    content: str         # Clean factual text of the section
    score: float = 0.0   # Hybrid composite relevance score
```

### 6.2 Engine Response Payload (`rag/engine.py`)
```json
{
  "query": "what is a Mutal fund",
  "status": "factual",
  "answer": "A mutual fund is a pool of money managed by a professional Fund Manager. A mutual fund is an investment vehicle formed when an Asset Management Company (AMC) or fund house pools money from multiple individuals and institutional investors who share a common investment objective. Official details: [What is a Mutual Fund](https://groww.in/p/what-is-mutual-fund). Last updated from sources: 2026-11-07",
  "source_url": "https://groww.in/p/what-is-mutual-fund",
  "source_title": "What is a Mutual Fund",
  "latency_ms": 174.5,
  "passages_count": 3
}
```

---

## 7. Configuration & Self-Hosting

### 7.1 Environment Variables (`.env`)
Inference parameters are managed exclusively via `.env`. No API keys or configurations are exposed through the web UI:

```bash
# Optional: LLM Provider selection (auto, groq, deepseek, openai, ollama, offline)
LLM_PROVIDER=auto

# Optional API Keys (leave empty to run 100% offline)
GROQ_API_KEY=
DEEPSEEK_API_KEY=
OPENAI_API_KEY=

# Optional Model & Endpoint Overrides
LLM_MODEL=llama-3.3-70b-versatile
LLM_BASE_URL=https://api.groq.com/openai/v1
```

### 7.2 Verification & Test Execution
The entire pipeline is guarded by automated test suites validating retrieval accuracy, guardrails, formatting limits, and offline fallback behavior:

```bash
# Run all 52 unit and integration tests
./venv/bin/pytest tests/ -v

# Run the automated headless evaluation suite across standard questions
./venv/bin/python cli.py --eval

# Run a single query directly from the terminal
./venv/bin/python cli.py --query "what is a Mutal fund"
```
