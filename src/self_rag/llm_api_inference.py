"""
Hosted-LLM-backed Self-RAG inference.

Same retrieve -> generate -> critique architecture as gguf_inference.py, but
generation and self-critique run against a hosted LLM API (OpenAI or
Anthropic) instead of a local GGUF model. Trades a locally-hosted 4GB+
model (slow on CPU, needs a beefy host) for a fast API call, and trades the
fine-tuned model's native reflection tokens (found to be an unreliable
single-sample signal -- see the resample fix in self_healing_graph.py) for
an explicit structured critique prompt against a more capable model.

Exposes the same .generate(question, passage, temperature) shape as
SelfRAGGGUFInference, so it plugs into build_self_healing_pipeline()
unchanged -- the retrieve/critique/retry graph doesn't care which engine
is answering.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import requests

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
}

CRITIQUE_SYSTEM_PROMPT = (
    "You are a legal analysis assistant answering questions about a contract "
    "using ONLY the provided passage. Respond with a single JSON object with "
    "exactly these fields, and nothing else:\n"
    '  "answer": your answer to the question, grounded only in the passage. '
    "If the passage doesn't address the question, say so plainly instead of guessing.\n"
    '  "isrel": "Relevant" or "Irrelevant" -- does the passage address the question at all?\n'
    '  "issup": "Fully supported", "Partially supported", or "No support" -- '
    "is your answer actually backed by the passage's text, or did you add anything not there?"
)


@dataclass
class SelfRAGOutput:
    """Same shape gguf_inference.SelfRAGOutput exposes, so callers (including
    build_self_healing_pipeline) don't need to know which engine produced it."""
    answer: str
    isrel: Optional[str] = None
    issup: Optional[str] = None
    isuse: Optional[str] = None
    raw_output: str = ""


class LLMAPIInference:
    """Self-RAG-style inference backed by a hosted LLM API instead of a local GGUF model."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.provider = provider
        self.api_key = api_key or os.getenv(f"{provider.upper()}_API_KEY")
        if not self.api_key:
            raise ValueError(
                f"No API key for provider '{provider}'. Set {provider.upper()}_API_KEY "
                f"or pass api_key= explicitly."
            )
        self.model = model or DEFAULT_MODELS.get(provider)
        if not self.model:
            raise ValueError(f"Unknown provider: {provider}")
        self.timeout = timeout

    def generate(
        self,
        question: str,
        passage: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 400,
        **kwargs,
    ) -> SelfRAGOutput:
        if passage:
            user_prompt = f"Passage:\n{passage}\n\nQuestion: {question}"
        else:
            user_prompt = f"No passage was retrieved for this question.\n\nQuestion: {question}"

        raw = self._call_api(user_prompt, temperature, max_tokens, CRITIQUE_SYSTEM_PROMPT)
        return self._parse(raw)

    def generate_json(
        self,
        instruction: str,
        context: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> Optional[dict]:
        """
        General-purpose structured JSON call -- NOT the Self-RAG critique
        schema .generate() uses. For tasks like key-term extraction where
        the caller defines its own JSON shape in `instruction`.

        Returns the parsed dict, or None if the response wasn't valid JSON.
        """
        system_prompt = (
            "You are a legal data-extraction specialist. Respond with ONLY a "
            "single valid JSON object matching the requested shape -- no other text."
        )
        user_prompt = f"{instruction}\n\nContract text:\n{context}" if context else instruction
        raw = self._call_api(user_prompt, temperature, max_tokens, system_prompt)
        return self._parse_json_only(raw)

    def _call_api(self, user_prompt: str, temperature: float, max_tokens: int, system_prompt: str) -> str:
        if self.provider == "openai":
            res = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                },
                timeout=self.timeout,
            )
            res.raise_for_status()
            return res.json()["choices"][0]["message"]["content"]

        if self.provider == "anthropic":
            res = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
                timeout=self.timeout,
            )
            res.raise_for_status()
            return res.json()["content"][0]["text"]

        raise ValueError(f"Unknown provider: {self.provider}")

    @staticmethod
    def _parse_json_only(raw: str) -> Optional[dict]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
            text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    @classmethod
    def _parse(cls, raw: str) -> SelfRAGOutput:
        data = cls._parse_json_only(raw)

        try:
            isrel = data.get("isrel", "Relevant")
            issup = data.get("issup", "Partially supported")
            return SelfRAGOutput(
                answer=data.get("answer", "").strip(),
                isrel=f"[{isrel}]",
                issup=f"[{issup}]",
                raw_output=raw,
            )
        except AttributeError:
            # Model didn't return valid JSON -- treat the raw text as the answer
            # but mark the critique as unreliable rather than pretending it passed.
            return SelfRAGOutput(
                answer=raw.strip(),
                isrel="[Relevant]",
                issup="[Partially supported]",
                raw_output=raw,
            )
