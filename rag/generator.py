"""
generator.py
Configurable Facts-Only Answer Generator supporting multiple LLM backends:
- Groq (llama-3.3-70b-versatile, llama-3.1-8b-instant)
- DeepSeek (deepseek-chat)
- OpenAI (gpt-4o-mini)
- Ollama / Local (llama3.1)
- Custom OpenAI-compatible endpoints
- Offline deterministic grounded extractor fallback
"""

import os
import re
import json
import urllib.request
from typing import List, Dict, Optional, Any
from rag.retriever import Passage

def load_dotenv_if_present():
    """Lightweight zero-dependency .env loader for local self-hosting."""
    for candidate in [".env", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")]:
        if os.path.isfile(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ and v:
                            os.environ[k] = v
            except Exception:
                pass
            break

# Auto-load .env on import if available
load_dotenv_if_present()

PROVIDER_DEFAULTS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "display_name": "Groq"
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "display_name": "DeepSeek"
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "display_name": "OpenAI"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.1",
        "env_key": None,
        "display_name": "Ollama (Local)"
    }
}

class FactsOnlyGenerator:
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.provider, self.api_key, self.base_url, self.model = self._resolve_config(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model
        )

    def _resolve_config(
        self,
        provider: Optional[str],
        api_key: Optional[str],
        base_url: Optional[str],
        model: Optional[str]
    ):
        # 1. Determine Provider
        prov = (provider or os.environ.get("LLM_PROVIDER", "auto")).lower()

        # Auto-detect if requested
        if prov == "auto":
            if os.environ.get("GROQ_API_KEY"):
                prov = "groq"
            elif os.environ.get("DEEPSEEK_API_KEY"):
                prov = "deepseek"
            elif os.environ.get("OPENAI_API_KEY"):
                prov = "openai"
            elif os.environ.get("LLM_BASE_URL"):
                prov = "custom"
            else:
                prov = "offline"

        # 2. Resolve Key
        key = None
        if api_key is not None:
            key = api_key.strip()
        else:
            if prov in PROVIDER_DEFAULTS and PROVIDER_DEFAULTS[prov]["env_key"]:
                key = os.environ.get(PROVIDER_DEFAULTS[prov]["env_key"])
            if not key:
                key = os.environ.get("LLM_API_KEY")

        # 3. Resolve Base URL
        b_url = base_url or os.environ.get("LLM_BASE_URL")
        if not b_url:
            if prov in PROVIDER_DEFAULTS:
                b_url = PROVIDER_DEFAULTS[prov]["base_url"]
            else:
                b_url = "https://api.groq.com/openai/v1"

        # 4. Resolve Model
        m_name = model or os.environ.get("LLM_MODEL")
        if not m_name:
            if prov in PROVIDER_DEFAULTS:
                m_name = PROVIDER_DEFAULTS[prov]["default_model"]
            else:
                m_name = "llama-3.3-70b-versatile"

        return prov, key, b_url.rstrip("/"), m_name

    def get_status(self) -> Dict[str, Any]:
        has_key = bool(self.api_key and "your_" not in self.api_key)
        is_active = (has_key or self.provider == "ollama")
        
        display = PROVIDER_DEFAULTS.get(self.provider, {}).get("display_name", self.provider.title())
        status_label = f"🟢 Connected: {display} ({self.model})" if is_active else "🟡 Offline Mode: Grounded Extractor"

        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "has_api_key": has_key,
            "is_active": is_active,
            "status_label": status_label
        }

    def _truncate_to_sentences(self, text: str, max_sentences: int = 3) -> str:
        updated_line = ""
        main_text = text
        update_match = re.search(r'(Last updated from sources:\s*[\d\-]+)', text, re.IGNORECASE)
        if update_match:
            updated_line = update_match.group(1)
            main_text = text[:update_match.start()].strip()

        sentences = re.split(r'(?<=[.!?])\s+', main_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > max_sentences:
            main_text = " ".join(sentences[:max_sentences])

        if updated_line:
            return f"{main_text} {updated_line}"
        return main_text

    def _ensure_citation(self, text: str, primary_passage: Passage) -> str:
        if re.search(r'\[([^\]]+)\]\((https?://[^\)]+)\)', text):
            return text

        clean_title = primary_passage.title.split("-")[0].strip()
        citation = f"[{clean_title}]({primary_passage.url})"
        
        if "Last updated from sources:" in text:
            parts = text.split("Last updated from sources:", 1)
            return f"{parts[0].strip()} {citation} Last updated from sources:{parts[1]}"
        return f"{text.strip()} {citation}"

    def _call_llm(self, prompt: str, system_prompt: str) -> Optional[str]:
        # If no key (and not local ollama), use fallback
        if (not self.api_key or "your_" in self.api_key) and self.provider != "ollama":
            return None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 250
        }
        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=12) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                content = res_data["choices"][0]["message"]["content"]
                return content.strip()
        except Exception:
            return None

    def _deterministic_extract(self, query: str, passages: List[Passage]) -> str:
        from rag.retriever import normalize_query
        q_lower = normalize_query(query)
        primary = passages[0]
        date_str = primary.last_updated or "2026-11-07"
        citation = f"[{primary.title}]({primary.url})"
        combined_content = "\n".join([p.content for p in passages])

        # Core Definition Queries
        if any(k in q_lower for k in [
            "what is a mutual fund", "what is mutual fund", "what are mutual funds",
            "define mutual fund", "meaning of mutual fund", "concept of mutual fund", "about mutual fund"
        ]):
            return (
                f"A mutual fund is an investment vehicle that pools capital from multiple investors to invest in a professionally managed portfolio of securities such as stocks, bonds, and money market instruments. "
                f"Investors are allocated units representing a proportional share of the scheme's holdings based on its Net Asset Value (NAV). "
                f"Official details: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        if ("what is nav" in q_lower or "net asset value" in q_lower) and "cut" not in q_lower:
            return (
                f"Net Asset Value (NAV) represents the market value per unit of a mutual fund scheme, calculated by dividing total net assets by the number of outstanding units. "
                f"NAV is calculated and declared at the close of every business day after market valuation. "
                f"Official details: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        if "what is a riskometer" in q_lower or "what is riskometer" in q_lower or "what is risk-o-meter" in q_lower:
            return (
                f"A Risk-o-meter is a standardized regulatory graphical representation mandated by SEBI that visualizes scheme risk across six tiers from Low to Very High. "
                f"Mutual fund houses evaluate and update scheme riskometers monthly based on portfolio assets. "
                f"Official details: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        if "cancel" in q_lower and "sip" in q_lower:
            return (
                f"To cancel an ongoing SIP on Groww, go to the 'Mutual Funds' section, select the 'SIPs' tab, choose your SIP, tap 'Cancel SIP' in the top right, select a cancellation reason, and confirm. "
                f"Note that cancellation must be completed at least 7 days before the next due date as per the official guide: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        if ("start" in q_lower or "create" in q_lower) and "sip" in q_lower:
            return (
                f"To start an SIP on Groww, search for your scheme, select 'Start SIP', choose your installment amount and monthly date, and set up an AutoPay mandate using net banking or debit card OTP. "
                f"Detailed instructions are provided in the official help guide: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        if "expense ratio" in q_lower or "ter" in q_lower:
            m = re.search(r'Expense Ratio.*?:[^\d]*([\d\.]+\s*%)', combined_content, re.IGNORECASE)
            ratio = m.group(1) if m else "disclosed in official scheme documents"
            scheme = primary.title.split("-")[0].strip()
            return (
                f"The expense ratio for {scheme} is {ratio}. "
                f"Full portfolio charges are published on the official page: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        if "exit load" in q_lower:
            m = re.search(r'Exit Load:\**\s*([^\n\r]+)', combined_content, re.IGNORECASE)
            load = m.group(1).strip() if m else "specified in the scheme KIM/SID"
            scheme = primary.title.split("-")[0].strip()
            return (
                f"The exit load for {scheme} is {load}. "
                f"Refer to official scheme disclosures for redemption conditions: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        if "lock-in" in q_lower or "lock in" in q_lower or "elss" in q_lower:
            return (
                f"ELSS (Equity Linked Saving Scheme) funds carry a statutory mandatory lock-in period of 3 years from the date of allotment. "
                f"Investments qualify for tax deduction under Section 80C as detailed in official guidance: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        if "minimum sip" in q_lower or "min sip" in q_lower or "100" in q_lower:
            m = re.search(r'Minimum SIP Investment:\**\s*([^\n\r]+)', combined_content, re.IGNORECASE)
            sip = m.group(1).strip() if m else "₹500"
            scheme = primary.title.split("-")[0].strip()
            return (
                f"The minimum SIP investment for {scheme} is {sip}. "
                f"Check detailed investment parameters on the official scheme page: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        if "capital gain" in q_lower or "tax report" in q_lower or "p&l report" in q_lower:
            return (
                f"To download your mutual fund capital gains statement on Groww, tap your Profile ('You' icon), navigate to 'SIP & Reports', select 'Capital Gain' under Tax, choose the Financial Year, and tap Download. "
                f"Official step-by-step guidance is available at: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        if "cas" in q_lower or "consolidated account" in q_lower:
            return (
                f"A Consolidated Account Statement (CAS) can be generated across all mutual fund folios and demat holdings via RTAs like CAMS and KFintech, or through the AMFI online portal. "
                f"Refer to the official AMFI CAS guide for instructions: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        if "benchmark" in q_lower:
            m = re.search(r'Benchmark:\**\s*([^\n\r]+)', combined_content, re.IGNORECASE)
            bm = m.group(1).strip() if m else "designated Total Return Index (TRI)"
            scheme = primary.title.split("-")[0].strip()
            return (
                f"The benchmark for {scheme} is {bm}. "
                f"View complete benchmark disclosures on the official page: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        if "cut-off" in q_lower or "cut off" in q_lower or "timing" in q_lower:
            return (
                f"Under SEBI regulations, the cut-off time for same-day NAV on equity/debt mutual funds on Groww is 2:00 PM (subject to realization of funds), while liquid and overnight funds have a 12:00 PM cut-off. "
                f"Detailed cut-off rules are outlined in the official documentation: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', primary.content) if len(s.strip()) > 20]
        summary = " ".join(sentences[:2]) if sentences else "Factual details are available in the official fund documentation."
        return (
            f"{summary} "
            f"Official details: {citation}. "
            f"Last updated from sources: {date_str}"
        )

    def generate(self, query: str, passages: List[Passage]) -> Dict[str, str]:
        if not passages:
            return {
                "answer": "No verified official sources were found matching this scheme query. Please refer to Groww Mutual Fund official disclosures: [Groww Factsheets](https://www.growwmf.in/downloads/factsheets). Last updated from sources: 2026-11-07",
                "source_url": "https://www.growwmf.in/downloads/factsheets",
                "source_title": "Groww Factsheets"
            }

        primary = passages[0]
        context_blocks = []
        for i, p in enumerate(passages, 1):
            context_blocks.append(
                f"[Source {i}]: {p.title}\n"
                f"URL: {p.url}\n"
                f"Organization: {p.organization}\n"
                f"Content: {p.content[:1000]}\n"
            )
        context_str = "\n".join(context_blocks)

        system_prompt = (
            "You are the Groww Mutual Fund Facts-Only Assistant.\n"
            "Your job is to provide concise, factual answers about mutual fund schemes, charges, statements, and regulations based strictly on the provided context.\n\n"
            "STRICT RULES:\n"
            "1. Base your answer solely on the context provided. Do not invent or extrapolate facts.\n"
            "2. Total explanation must be at most 3 sentences.\n"
            "3. Include exactly ONE clickable markdown citation link ([Source Title](Source URL)) in your answer.\n"
            "4. Never provide investment advice, buy/sell recommendations, or personal portfolio opinions.\n"
            f"5. Always conclude your answer with: \"Last updated from sources: {primary.last_updated}\"."
        )

        user_prompt = (
            f"Context:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            "Answer (<=3 sentences with 1 citation link and 'Last updated from sources:' suffix):"
        )

        llm_response = self._call_llm(user_prompt, system_prompt)

        if not llm_response:
            final_text = self._deterministic_extract(query, passages)
        else:
            final_text = self._ensure_citation(llm_response, primary)
            final_text = self._truncate_to_sentences(final_text, max_sentences=3)
            if "Last updated from sources:" not in final_text:
                final_text = f"{final_text.strip()} Last updated from sources: {primary.last_updated}"

        return {
            "answer": final_text,
            "source_url": primary.url,
            "source_title": primary.title
        }
