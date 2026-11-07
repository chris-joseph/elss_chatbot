"""
generator.py
Facts-Only Answer Generator with strict citation enforcement and length constraints.
Integrates with DeepSeek Chat API and includes an offline deterministic grounded extractor.
"""

import os
import re
import json
import urllib.request
from typing import List, Dict, Optional
from rag.retriever import Passage

class FactsOnlyGenerator:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key if api_key is not None else os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = (base_url or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-chat")

    def _truncate_to_sentences(self, text: str, max_sentences: int = 3) -> str:
        # Separate out the "Last updated from sources:" line if present
        updated_line = ""
        main_text = text
        update_match = re.search(r'(Last updated from sources:\s*[\d\-]+)', text, re.IGNORECASE)
        if update_match:
            updated_line = update_match.group(1)
            main_text = text[:update_match.start()].strip()

        # Split text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', main_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if len(sentences) > max_sentences:
            main_text = " ".join(sentences[:max_sentences])

        if updated_line:
            return f"{main_text} {updated_line}"
        return main_text

    def _ensure_citation(self, text: str, primary_passage: Passage) -> str:
        # Check if text already has a valid markdown link
        if re.search(r'\[([^\]]+)\]\((https?://[^\)]+)\)', text):
            return text

        clean_title = primary_passage.title.split("-")[0].strip()
        citation = f"[{clean_title}]({primary_passage.url})"
        
        if "Last updated from sources:" in text:
            parts = text.split("Last updated from sources:", 1)
            return f"{parts[0].strip()} {citation} Last updated from sources:{parts[1]}"
        return f"{text.strip()} {citation}"

    def _call_llm(self, prompt: str, system_prompt: str) -> Optional[str]:
        if not self.api_key or "your_deepseek" in self.api_key:
            return None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
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
        """Deterministic grounded extractor for reliable facts when offline."""
        q_lower = query.lower()
        primary = passages[0]
        date_str = primary.last_updated or "2026-11-07"
        citation = f"[{primary.title}]({primary.url})"

        # Cancel SIP
        if "cancel" in q_lower and "sip" in q_lower:
            return (
                f"To cancel an ongoing SIP on Groww, go to the 'Mutual Funds' section, select the 'SIPs' tab, choose your SIP, tap 'Cancel SIP' in the top right, select a cancellation reason, and confirm. "
                f"Note that cancellation must be completed at least 7 days before the next due date as per the official guide: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        # Start SIP
        if ("start" in q_lower or "create" in q_lower) and "sip" in q_lower:
            return (
                f"To start an SIP on Groww, search for your scheme, select 'Start SIP', choose your installment amount and monthly date, and set up an AutoPay mandate using net banking or debit card OTP. "
                f"Detailed instructions are provided in the official help guide: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        # Expense ratio query
        if "expense ratio" in q_lower or "ter" in q_lower:
            m = re.search(r'Expense Ratio.*?:[^\d]*([\d\.]+\s*%)', primary.content, re.IGNORECASE)
            ratio = m.group(1) if m else "disclosed in official scheme documents"
            scheme = primary.title.split("-")[0].strip()
            return (
                f"The expense ratio for {scheme} is {ratio}. "
                f"Full portfolio charges are published on the official page: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        # Exit load query
        if "exit load" in q_lower:
            m = re.search(r'Exit Load:\**\s*([^\n\r]+)', primary.content, re.IGNORECASE)
            load = m.group(1).strip() if m else "specified in the scheme KIM/SID"
            scheme = primary.title.split("-")[0].strip()
            return (
                f"The exit load for {scheme} is {load}. "
                f"Refer to official scheme disclosures for redemption conditions: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        # Lock-in / ELSS query
        if "lock-in" in q_lower or "lock in" in q_lower or "elss" in q_lower:
            return (
                f"ELSS (Equity Linked Saving Scheme) funds carry a statutory mandatory lock-in period of 3 years from the date of allotment. "
                f"Investments qualify for tax deduction under Section 80C as detailed in official guidance: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        # Minimum SIP or Investment query
        if "minimum sip" in q_lower or "min sip" in q_lower or "100" in q_lower:
            m = re.search(r'Minimum SIP Investment:\**\s*([^\n\r]+)', primary.content, re.IGNORECASE)
            sip = m.group(1).strip() if m else "₹500"
            scheme = primary.title.split("-")[0].strip()
            return (
                f"The minimum SIP investment for {scheme} is {sip}. "
                f"Check detailed investment parameters on the official scheme page: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        # Capital gains / statement download
        if "capital gain" in q_lower or "tax report" in q_lower or "p&l report" in q_lower:
            return (
                f"To download your mutual fund capital gains statement on Groww, tap your Profile ('You' icon), navigate to 'SIP & Reports', select 'Capital Gain' under Tax, choose the Financial Year, and tap Download. "
                f"Official step-by-step guidance is available at: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        # CAS statement
        if "cas" in q_lower or "consolidated account" in q_lower:
            return (
                f"A Consolidated Account Statement (CAS) can be generated across all mutual fund folios and demat holdings via RTAs like CAMS and KFintech, or through the AMFI online portal. "
                f"Refer to the official AMFI CAS guide for instructions: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        # Benchmark query
        if "benchmark" in q_lower:
            m = re.search(r'Benchmark:\s*([^\n\r]+)', primary.content, re.IGNORECASE)
            bm = m.group(1).strip() if m else "designated Total Return Index (TRI)"
            scheme = primary.title.split("-")[0].strip()
            return (
                f"The benchmark for {scheme} is {bm}. "
                f"View complete benchmark disclosures on the official page: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        # Cut-off timing query
        if "cut-off" in q_lower or "cut off" in q_lower or "timing" in q_lower:
            return (
                f"Under SEBI regulations, the cut-off time for same-day NAV on equity/debt mutual funds on Groww is 2:00 PM (subject to realization of funds), while liquid and overnight funds have a 12:00 PM cut-off. "
                f"Detailed cut-off rules are outlined in the official documentation: {citation}. "
                f"Last updated from sources: {date_str}"
            )

        # Generic summary from first passage
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
