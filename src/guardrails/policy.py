"""
Policy Engine

Loads a YAML rules file so non-engineers can define what's allowed without
touching code, e.g. "never discuss competitors", "block medical advice".

Two rule types are supported:
- keyword_block: violation if any keyword appears in the text
- keyword_require: violation if NONE of the keywords appear in the text
  (used for "always cite sources"-style rules)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class PolicyViolation:
    rule_name: str
    reason: str


@dataclass
class PolicyCheckResult:
    allowed: bool
    violations: List[PolicyViolation] = field(default_factory=list)


class PolicyEngine:
    """Evaluates text against a configurable set of YAML-defined policy rules."""

    def __init__(self, rules: List[Dict[str, Any]]):
        self.rules = rules

    @classmethod
    def from_yaml(cls, config_path: str) -> "PolicyEngine":
        path = Path(config_path)
        if not path.exists():
            return cls(rules=[])
        with open(path, "r") as f:
            config = yaml.safe_load(f) or {}
        return cls(rules=config.get("rules", []))

    def _applies(self, rule: Dict[str, Any], applies_to: str) -> bool:
        target = rule.get("applies_to", "both")
        return target == "both" or target == applies_to

    def check(self, text: str, applies_to: str = "output") -> PolicyCheckResult:
        """
        Evaluate `text` against all configured rules.

        Args:
            text: The text to check (a user query or an LLM response).
            applies_to: "input" or "output" -- filters which rules apply.
        """
        lowered = text.lower()
        violations: List[PolicyViolation] = []

        for rule in self.rules:
            if not rule.get("enabled", True):
                continue
            if not self._applies(rule, applies_to):
                continue

            rule_type = rule.get("type")
            keywords = [k.lower() for k in rule.get("keywords", [])]
            name = rule.get("name", "unnamed_rule")

            if rule_type == "keyword_block":
                hit = next((k for k in keywords if k in lowered), None)
                if hit:
                    violations.append(PolicyViolation(
                        rule_name=name,
                        reason=f"blocked keyword matched: '{hit}'",
                    ))

            elif rule_type == "keyword_require":
                if keywords and not any(k in lowered for k in keywords):
                    violations.append(PolicyViolation(
                        rule_name=name,
                        reason="required keyword/citation not found",
                    ))

        return PolicyCheckResult(allowed=len(violations) == 0, violations=violations)
