"""
generator.py
Configurable Facts-Only Answer Generator supporting multiple LLM backends:
- Groq (llama-3.3-70b-versatile, llama-3.1-8b-instant)
- DeepSeek (deepseek-chat)
- OpenAI (gpt-4o-mini)
- Ollama / Local (llama3.1)
- Custom OpenAI-compatible endpoints
- Offline principled grounded extractive summarizer fallback
"""

import os
import re
import json
import urllib.request
from typing import List, Dict, Optional, Any, Set, Tuple
from rag.retriever import Passage, STOPWORDS

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

GENERIC_TOPIC_WORDS: Set[str] = {
    "fund", "funds", "mutual", "scheme", "groww", "details", "facts",
    "what", "is", "the", "for", "official", "tell", "about", "which"
}

def clean_sentence(s: str) -> str:
    """Removes links, formatting noise, and dates from extracted sentences."""
    s = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', s)
    s = re.sub(r'[*_#`]', '', s)
    s = re.sub(r'^\d{1,2}\s+[A-Za-z]+\s+\d{4}\s+\d+\s+min\s+read\s*', '', s)
    s = re.sub(r'^\d+\.\s*', '', s)
    s = re.sub(r'^[-*]\s*', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

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
        prov = (provider or os.environ.get("LLM_PROVIDER", "auto")).lower()

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

        key = None
        if api_key is not None:
            key = api_key.strip()
        else:
            if prov in PROVIDER_DEFAULTS and PROVIDER_DEFAULTS[prov]["env_key"]:
                key = os.environ.get(PROVIDER_DEFAULTS[prov]["env_key"])
            if not key:
                key = os.environ.get("LLM_API_KEY")

        b_url = base_url or os.environ.get("LLM_BASE_URL")
        if not b_url:
            if prov in PROVIDER_DEFAULTS:
                b_url = PROVIDER_DEFAULTS[prov]["base_url"]
            else:
                b_url = "https://api.groq.com/openai/v1"

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

    def _extract_structured_facts(
        self,
        passages: List[Passage],
        q_tokens: List[str]
    ) -> List[Tuple[float, str, str, Passage]]:
        """
        Dynamically extracts key-value attribute lines (e.g. Expense Ratio, Exit Load, Benchmark)
        matching query-specific tokens.
        """
        q_specific = set(q_tokens) - GENERIC_TOPIC_WORDS
        if not q_specific:
            return []

        facts = []
        seen_keys = set()
        for p in passages[:2]:
            for line in p.content.splitlines():
                line = line.strip()
                m = re.match(r"^[-*]?\s*(?:\*\*)?([A-Za-z0-9\s/()-]+?)(?:\*\*)?:\s*(.+)$", line)
                if m:
                    k, v = m.group(1).strip(), m.group(2).strip()
                    if len(k.split()) > 5 or len(v) < 2:
                        continue
                    if k.lower().startswith(("please", "note", "http", "click", "watch", "disclaimer", "was")):
                        continue

                    k_clean = re.sub(r"[^a-zA-Z0-9\s]", "", k).lower()
                    k_tokens = set(k_clean.split())
                    overlap = k_tokens & q_specific
                    if overlap:
                        base_k = k_clean.replace("direct growth", "").strip()
                        if base_k in seen_keys:
                            continue
                        seen_keys.add(base_k)

                        v_clean = clean_sentence(v)
                        if not v_clean or len(v_clean) < 2:
                            continue
                        score = len(overlap) / max(1, len(k_tokens - GENERIC_TOPIC_WORDS))
                        facts.append((score, k, v_clean, p))

        facts.sort(key=lambda x: x[0], reverse=True)
        return facts

    def _score_sentences(
        self,
        passages: List[Passage],
        query: str,
        q_tokens: List[str]
    ) -> List[Tuple[float, str, Passage]]:
        """
        Ranks candidate sentences from retrieved passages based on query keyword density,
        definitional copulas, and procedural actions.
        """
        q_set = set(q_tokens)
        q_lower = query.lower()
        is_def = any(w in q_lower for w in ["what is", "what are", "define", "meaning"])
        is_proc = any(w in q_lower for w in ["how to", "how do", "how can", "steps", "cancel", "start", "download"])

        scored_sentences = []
        seen = set()

        for p_idx, p in enumerate(passages[:2]):
            raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", p.content) if s.strip()]
            for s_idx, raw_s in enumerate(raw_sentences):
                if raw_s.startswith("#") or raw_s.endswith("?"):
                    continue
                clean_s = clean_sentence(raw_s)
                if len(clean_s) < 25 or len(clean_s) > 300:
                    continue
                if clean_s.endswith(":") or clean_s in seen:
                    continue
                seen.add(clean_s)

                s_words = [w for w in re.findall(r"\b\w+\b", clean_s.lower()) if len(w) >= 3]
                s_set = set(s_words)
                if not s_words:
                    continue

                overlap = q_set & s_set
                density = len(overlap) / (len(s_words) ** 0.5)
                score = density

                # Definitional query boost
                if is_def and re.search(r"\b(is a|is an|refers to|means|is defined as|represents|is a mechanism|is a pool)\b", clean_s.lower()):
                    score += 0.8

                # Procedural query boost
                if is_proc and re.search(r"\b(step\s*\d+|steps|tap|select|click|navigate|go to|choose|confirm|download|autopay|mandate|days before)\b", clean_s.lower()):
                    score += 0.8

                if s_idx == 0:
                    score += 0.2
                if p_idx == 0:
                    score += 0.1

                scored_sentences.append((score, clean_s, p))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return scored_sentences

    def _deterministic_extract(self, query: str, passages: List[Passage]) -> str:
        """
        Fully grounded, un-overfitted extractive engine that answers user queries
        directly from retrieved source documents without hardcoded query mappings.
        """
        if not passages:
            return (
                "No verified official sources were found matching this query. "
                "Official details: [Groww Factsheets](https://www.growwmf.in/downloads/factsheets). "
                "Last updated from sources: 2026-11-07"
            )

        primary = passages[0]
        q_clean = query.lower()
        q_tokens = [w for w in re.findall(r"\b\w+\b", q_clean) if len(w) >= 3 and w not in STOPWORDS]

        # 1. Check for structured key-value attributes
        struct_facts = self._extract_structured_facts(passages, q_tokens)
        final_sentences = []

        if struct_facts:
            scheme_name = primary.title.split("-")[0].strip()
            for _, k, v, _ in struct_facts[:2]:
                final_sentences.append(f"The {k} for {scheme_name} is {v}.")
        else:
            # 2. Extract salient sentences based on BM25 query term overlap density
            narrative_sents = self._score_sentences(passages, query, q_tokens)
            for _, sent, _ in narrative_sents[:2]:
                final_sentences.append(sent if sent.endswith((".", "!", "?")) else f"{sent}.")

        if not final_sentences:
            final_sentences = ["Official scheme facts and regulatory disclosures are available in the verified documentation."]

        main_text = " ".join(final_sentences[:2])
        citation = f"[{primary.title}]({primary.url})"
        return f"{main_text} Official details: {citation}. Last updated from sources: {primary.last_updated}"

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
