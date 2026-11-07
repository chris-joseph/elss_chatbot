"""
guardrails.py
Validates user queries against strict compliance constraints:
1. No PII (PAN, Aadhaar, Phone, Email, OTP, Bank Account)
2. No Investment Advice / Portfolio Opinions (Buy/Sell/Switch recommendations)
3. No Performance Claims (Return predictions, CAGR comparisons)
"""

import re
from dataclasses import dataclass
from typing import Optional

@dataclass
class GuardrailResult:
    is_safe: bool
    status: str  # "ok", "refusal_pii", "refusal_advice", "refusal_performance"
    message: Optional[str] = None
    citation_url: Optional[str] = None
    citation_title: Optional[str] = None

class GuardrailsChecker:
    # Regex Patterns for PII
    PAN_PATTERN = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]\b', re.IGNORECASE)
    AADHAAR_PATTERN = re.compile(r'\b\d{4}\s?\d{4}\s?\d{4}\b')
    PHONE_PATTERN = re.compile(r'(?:\+?91[\s-]?)?[6-9]\d{9}\b')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b')
    OTP_PATTERN = re.compile(r'\b(?:otp|one\s*time\s*password)\b.*?\b\d{4,6}\b|\b\d{4,6}\b.*?\b(?:otp|code)\b', re.IGNORECASE)
    ACCOUNT_PATTERN = re.compile(r'\b(?:account|acct|acc)\s*(?:no|number)?\s*[:#-]?\s*\d{9,18}\b', re.IGNORECASE)

    # Patterns indicating personal advice / opinion requests
    ADVICE_PATTERNS = [
        r'\bshould\s+i\s+(?:buy|sell|invest|switch|choose|hold|redeem|exit|pick)\b',
        r'\bwhich\b.*?\b(?:better|best)\b',
        r'\bwhich\s+one\s+should\s+i\b',
        r'\b(?:recommend|suggest)\b.*?\b(?:fund|scheme|portfolio|investment)\b',
        r'\bis\s+(?:it\s+)?(?:a\s+)?(?:good|safe|bad|right)\s+(?:time\s+)?to\s+invest\b',
        r'\bwhere\s+should\s+i\s+(?:invest|put)\b',
        r'\bhow\s+should\s+i\s+(?:allocate|distribute)\b',
        r'\bhelp\s+me\s+(?:choose|pick|select|build\s+a\s+portfolio)\b',
        r'\bwhat\s+is\s+your\s+(?:opinion|recommendation|advice)\b',
        r'\bwill\s+i\s+become\s+rich\b',
        r'\bis\s+groww\s+(?:elss|multicap|nifty|liquid)\s+(?:better|worse)\s+than\b'
    ]

    # Patterns for performance/return claims
    PERFORMANCE_PATTERNS = [
        r'\b(?:how\s+much|what)\s+(?:return|returns|profit|gain|cagr|yield)\b',
        r'\b(?:return|returns|profit|cagr)\b.*?\b(?:give|yield|generate|expect)\b',
        r'\bgive\s+\d+%\s*(?:return|profit|cagr)?\b',
        r'\b\d+%\s*returns?\b',
        r'\bcompare\s+(?:the\s+)?returns\b',
        r'\bwhich\s+fund\s+gives\s+(?:higher|highest|more|best)\s+return\b',
        r'\bcalculate\s+(?:my\s+)?(?:future\s+)?returns?\b',
        r'\bexpected\s+returns?\b',
        r'\bprojected\s+returns?\b'
    ]

    def check(self, query: str) -> GuardrailResult:
        normalized = query.strip()

        # 1. PII Check
        if self.PAN_PATTERN.search(normalized):
            return GuardrailResult(
                is_safe=False,
                status="refusal_pii",
                message="For your security and privacy, please do not share personal identification details such as PAN, Aadhaar, account numbers, or OTPs. I only answer public factual questions about mutual fund schemes."
            )
        if self.AADHAAR_PATTERN.search(normalized):
            return GuardrailResult(
                is_safe=False,
                status="refusal_pii",
                message="For your security and privacy, please do not share personal identification details such as PAN, Aadhaar, account numbers, or OTPs. I only answer public factual questions about mutual fund schemes."
            )
        if self.PHONE_PATTERN.search(normalized):
            return GuardrailResult(
                is_safe=False,
                status="refusal_pii",
                message="For your security and privacy, please do not share phone numbers or personal contact info. I only answer public factual questions about mutual fund schemes."
            )
        if self.EMAIL_PATTERN.search(normalized):
            return GuardrailResult(
                is_safe=False,
                status="refusal_pii",
                message="For your security and privacy, please do not share email addresses. I only answer public factual questions about mutual fund schemes."
            )
        if self.OTP_PATTERN.search(normalized) or self.ACCOUNT_PATTERN.search(normalized):
            return GuardrailResult(
                is_safe=False,
                status="refusal_pii",
                message="For your security and privacy, sensitive financial credentials (OTPs, bank account numbers) are not accepted. I only answer public factual questions about mutual fund schemes."
            )

        # 2. Performance / Returns Claim Check (Run before general advice so specific return queries trigger performance refusal)
        for pattern in self.PERFORMANCE_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return GuardrailResult(
                    is_safe=False,
                    status="refusal_performance",
                    message="I cannot compute, project, or compare fund returns. Please consult the official Groww Mutual Fund monthly factsheet for historical scheme disclosures: [Groww Factsheets](https://www.growwmf.in/downloads/factsheets).",
                    citation_url="https://www.growwmf.in/downloads/factsheets",
                    citation_title="Groww Factsheets"
                )

        # 3. Advice / Opinion Check
        for pattern in self.ADVICE_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE):
                return GuardrailResult(
                    is_safe=False,
                    status="refusal_advice",
                    message="As a facts-only assistant, I do not provide investment advice, buy/sell recommendations, or personal portfolio opinions. You can review official investor education resources on AMFI: [AMFI Investor Education](https://www.amfiindia.com/investor).",
                    citation_url="https://www.amfiindia.com/investor",
                    citation_title="AMFI Investor Education"
                )

        return GuardrailResult(is_safe=True, status="ok")
