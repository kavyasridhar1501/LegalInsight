"""
Output Guardrails

Checks LLM output before it is returned to the user:
- Schema conformance for structured extraction (parties, dates, payment terms, ...)
- Toxic / abusive language
- On-topic check (response should stay within legal-contract analysis)
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OutputCheckResult:
    allowed: bool
    blocked_reasons: List[str] = field(default_factory=list)
    schema_errors: List[str] = field(default_factory=list)


# Required top-level fields for the structured key-term extraction output.
EXTRACTION_SCHEMA_REQUIRED_FIELDS = {
    "parties": list,
    "dates": list,
    "payment_terms": (str, list),
    "liability_cap": (str, type(None)),
    "risks": list,
}

# Small denylist of clearly toxic/abusive terms. Intentionally conservative
# (few, high-precision entries) to avoid false-positives on legal text, which
# routinely contains words like "liability", "breach", "damages", etc.
_TOXIC_TERMS = [
    "kill yourself", "i hate you", "go die", "you are worthless",
]

_LEGAL_DOMAIN_HINTS = [
    "contract", "agreement", "clause", "party", "parties", "term", "terms",
    "liability", "obligation", "termination", "indemnif", "warranty",
    "confidential", "governing law", "dispute", "payment", "breach",
    "damages", "provision", "section", "article", "insufficient information",
]


class OutputGuardrails:
    """Validates LLM responses for schema conformance, toxicity, and topicality."""

    def __init__(self, enforce_on_topic: bool = True):
        self.enforce_on_topic = enforce_on_topic

    def check_toxicity(self, text: str) -> List[str]:
        lowered = text.lower()
        return [term for term in _TOXIC_TERMS if term in lowered]

    def check_on_topic(self, text: str) -> bool:
        """Heuristic: response should reference at least one legal/contract term."""
        if not text.strip():
            return False
        lowered = text.lower()
        return any(hint in lowered for hint in _LEGAL_DOMAIN_HINTS)

    def check_answer(self, text: str) -> OutputCheckResult:
        """Guardrail pass for a free-text analysis answer (not structured JSON)."""
        blocked_reasons = []

        toxic_hits = self.check_toxicity(text)
        if toxic_hits:
            blocked_reasons.append("toxic_content_detected")

        if self.enforce_on_topic and not self.check_on_topic(text):
            blocked_reasons.append("off_topic")

        return OutputCheckResult(
            allowed=len(blocked_reasons) == 0,
            blocked_reasons=blocked_reasons,
        )

    def check_extraction_schema(self, raw_output: str) -> OutputCheckResult:
        """Validate the structured key-term extraction JSON against the expected schema."""
        schema_errors = []

        parsed: Optional[Dict[str, Any]] = None
        try:
            parsed = json.loads(raw_output)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        if parsed is None or not isinstance(parsed, dict):
            return OutputCheckResult(
                allowed=False,
                blocked_reasons=["invalid_json"],
                schema_errors=["Response is not valid JSON"],
            )

        for field_name, expected_type in EXTRACTION_SCHEMA_REQUIRED_FIELDS.items():
            if field_name not in parsed:
                schema_errors.append(f"missing field: {field_name}")
                continue
            if not isinstance(parsed[field_name], expected_type):
                schema_errors.append(f"field '{field_name}' has wrong type")

        toxic_hits = self.check_toxicity(raw_output)
        blocked_reasons = []
        if schema_errors:
            blocked_reasons.append("schema_violation")
        if toxic_hits:
            blocked_reasons.append("toxic_content_detected")

        return OutputCheckResult(
            allowed=len(blocked_reasons) == 0,
            blocked_reasons=blocked_reasons,
            schema_errors=schema_errors,
        )
