"""
Input Guardrails

Screens user input (queries, pasted contract text) before it reaches the LLM:
- PII detection (credit cards, SSNs, emails, phone numbers)
- Prompt injection / jailbreak pattern detection

PII in a legal contract's body is expected and is NOT blocked (that's the
document being analyzed); only PII in the *query* field is flagged, since
a user typing their own card number into a chat box is the classic accidental
leak this guardrail exists to catch.
"""

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class InputCheckResult:
    allowed: bool
    blocked_reasons: List[str] = field(default_factory=list)
    pii_findings: List[str] = field(default_factory=list)
    redacted_text: str = ""


def _luhn_checksum(digits: str) -> bool:
    """Validate a digit string against the Luhn algorithm (reduces credit-card false positives)."""
    total = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        n = int(d)
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


class InputGuardrails:
    """Detects PII and prompt-injection/jailbreak attempts in user-supplied text."""

    _CREDIT_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
    _SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    _EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    _PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")

    # Common jailbreak / prompt-injection phrasing seen in the wild. Not exhaustive,
    # but catches the frequent patterns without needing a hosted classifier.
    _INJECTION_PATTERNS = [
        re.compile(r"ignore (?:all |any )?(?:previous|prior|above) instructions", re.I),
        re.compile(r"disregard (?:all |any )?(?:previous|prior|above) (?:instructions|rules)", re.I),
        re.compile(r"you are now (?:in )?(?:dan|developer mode|jailbreak)", re.I),
        re.compile(r"act as (?:if you (?:are|have) no|an unfiltered|a jailbroken)", re.I),
        re.compile(r"reveal your (?:system prompt|instructions|prompt)", re.I),
        re.compile(r"pretend (?:you have |to have )?no (?:restrictions|guidelines|filters)", re.I),
        re.compile(r"do anything now", re.I),
        re.compile(r"\bsystem\s*:\s*", re.I),
        re.compile(r"<\s*/?system\s*>", re.I),
    ]

    def __init__(self, block_pii: bool = True, block_injection: bool = True):
        self.block_pii = block_pii
        self.block_injection = block_injection

    def _find_pii(self, text: str) -> List[str]:
        findings = []
        if self._SSN_RE.search(text):
            findings.append("ssn")
        if self._EMAIL_RE.search(text):
            findings.append("email")
        if self._PHONE_RE.search(text):
            findings.append("phone")
        for match in self._CREDIT_CARD_RE.finditer(text):
            digits = re.sub(r"[ -]", "", match.group())
            if len(digits) in (13, 14, 15, 16, 17, 18, 19) and _luhn_checksum(digits):
                findings.append("credit_card")
                break
        return findings

    def _redact(self, text: str) -> str:
        redacted = self._SSN_RE.sub("[REDACTED-SSN]", text)
        redacted = self._EMAIL_RE.sub("[REDACTED-EMAIL]", redacted)
        redacted = self._PHONE_RE.sub("[REDACTED-PHONE]", redacted)

        def _cc_sub(match: "re.Match") -> str:
            digits = re.sub(r"[ -]", "", match.group())
            if len(digits) in range(13, 20) and _luhn_checksum(digits):
                return "[REDACTED-CARD]"
            return match.group()

        redacted = self._CREDIT_CARD_RE.sub(_cc_sub, redacted)
        return redacted

    def _find_injection(self, text: str) -> List[str]:
        return [p.pattern for p in self._INJECTION_PATTERNS if p.search(text)]

    def check_query(self, query: str) -> InputCheckResult:
        """Check a user query for PII leakage and prompt-injection attempts."""
        pii_findings = self._find_pii(query)
        injection_hits = self._find_injection(query)

        blocked_reasons = []
        if self.block_injection and injection_hits:
            blocked_reasons.append("prompt_injection_detected")
        if self.block_pii and pii_findings:
            blocked_reasons.append("pii_detected:" + ",".join(sorted(set(pii_findings))))

        return InputCheckResult(
            allowed=len(blocked_reasons) == 0,
            blocked_reasons=blocked_reasons,
            pii_findings=pii_findings,
            redacted_text=self._redact(query),
        )
