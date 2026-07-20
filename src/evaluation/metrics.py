"""
Evaluation metrics for the legal RAG pipeline.

Computes the metrics an "LLM eval CI/CD" gate needs to make a pass/fail
decision on every change to a prompt, model, or knowledge base:
- Retrieval quality: precision@k, recall@k, MRR
- Hallucination rate (EigenScore + ISSUP-based)
- Faithfulness to retrieved sources (ISSUP-based)
- Latency percentiles (p50/p95)
- Estimated cost per query
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _char_iou(a: str, b: str) -> float:
    """Character n-gram Jaccard-style overlap used as a cheap proxy for span IoU."""
    a, b = a.strip().lower(), b.strip().lower()
    if not a or not b:
        return 0.0
    if a in b or b in a:
        shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
        return len(shorter) / len(longer)

    def ngrams(s: str, n: int = 5) -> set:
        return {s[i:i + n] for i in range(max(len(s) - n + 1, 1))}

    set_a, set_b = ngrams(a), ngrams(b)
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


@dataclass
class EvalRecord:
    """One golden-dataset query run through the pipeline."""
    query: str
    gold_passage: Optional[str] = None
    retrieved_passages: List[str] = field(default_factory=list)
    predicted_answer: Optional[str] = None
    issup: Optional[str] = None
    isrel: Optional[str] = None
    eigenscore: Optional[float] = None
    used_fallback: bool = False
    latency_seconds: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


# ---- Retrieval metrics ----

def _hit_rank(record: EvalRecord, min_iou: float) -> Optional[int]:
    """1-indexed rank of the first retrieved passage that matches the gold passage, or None."""
    if not record.gold_passage:
        return None
    for rank, passage in enumerate(record.retrieved_passages, start=1):
        if _char_iou(passage, record.gold_passage) >= min_iou:
            return rank
    return None


def precision_at_k(records: List[EvalRecord], k: int, min_iou: float = 0.5) -> float:
    scored = [r for r in records if r.gold_passage]
    if not scored:
        return 0.0
    hits = 0
    for r in scored:
        rank = _hit_rank(r, min_iou)
        if rank is not None and rank <= k:
            hits += 1
    return hits / len(scored) / k if k > 0 else 0.0


def recall_at_k(records: List[EvalRecord], k: int, min_iou: float = 0.5) -> float:
    """With one relevant passage per query, recall@k reduces to hit-rate@k."""
    scored = [r for r in records if r.gold_passage]
    if not scored:
        return 0.0
    hits = sum(1 for r in scored if (rank := _hit_rank(r, min_iou)) is not None and rank <= k)
    return hits / len(scored)


def mean_reciprocal_rank(records: List[EvalRecord], min_iou: float = 0.5) -> float:
    scored = [r for r in records if r.gold_passage]
    if not scored:
        return 0.0
    total = 0.0
    for r in scored:
        rank = _hit_rank(r, min_iou)
        if rank is not None:
            total += 1.0 / rank
    return total / len(scored)


# ---- Generation quality metrics ----

def hallucination_rate(records: List[EvalRecord], eigenscore_threshold: float = -2.0) -> float:
    """
    Fraction of records flagged as a hallucination: high EigenScore divergence,
    an ISSUP token indicating no/contradictory support, or the pipeline fell
    back to "insufficient information" (counted separately, not a hallucination,
    but excluded from the denominator's non-fallback answers is intentionally
    NOT done here -- an honest fallback is the desired behavior, not a failure).
    """
    scored = [r for r in records if not r.used_fallback]
    if not scored:
        return 0.0
    flagged = 0
    for r in scored:
        is_bad = False
        if r.eigenscore is not None and r.eigenscore > eigenscore_threshold:
            is_bad = True
        if r.issup is not None and ("No support" in r.issup or "Contradictory" in r.issup):
            is_bad = True
        if is_bad:
            flagged += 1
    return flagged / len(scored)


def faithfulness_score(records: List[EvalRecord]) -> float:
    """Average ISSUP-derived faithfulness in [0, 1]; records without an ISSUP token are skipped."""
    scores = []
    for r in records:
        if r.issup is None:
            continue
        if "Fully" in r.issup:
            scores.append(1.0)
        elif "Partially" in r.issup:
            scores.append(0.5)
        else:
            scores.append(0.0)
    return sum(scores) / len(scores) if scores else 0.0


def fallback_rate(records: List[EvalRecord]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.used_fallback) / len(records)


# ---- Operational metrics ----

def latency_percentiles(records: List[EvalRecord], percentiles: List[int] = [50, 95]) -> Dict[str, float]:
    values = sorted(r.latency_seconds for r in records if r.latency_seconds is not None)
    if not values:
        return {f"p{p}": 0.0 for p in percentiles}
    result = {}
    for p in percentiles:
        idx = min(math.ceil(p / 100 * len(values)) - 1, len(values) - 1)
        idx = max(idx, 0)
        result[f"p{p}"] = values[idx]
    return result


# Rough per-1K-token pricing (USD) for the AI providers this app supports.
# Used only for a cost estimate; keep in sync loosely with provider pricing pages.
DEFAULT_PRICE_PER_1K_TOKENS = {
    "prompt": 0.005,
    "completion": 0.015,
}


def cost_per_query(records: List[EvalRecord], price_table: Dict[str, float] = None) -> float:
    price_table = price_table or DEFAULT_PRICE_PER_1K_TOKENS
    if not records:
        return 0.0
    total_cost = 0.0
    for r in records:
        total_cost += (r.prompt_tokens / 1000) * price_table["prompt"]
        total_cost += (r.completion_tokens / 1000) * price_table["completion"]
    return total_cost / len(records)


@dataclass
class EvalReport:
    num_records: int
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    mrr: float
    hallucination_rate: float
    faithfulness_score: float
    fallback_rate: float
    latency: Dict[str, float]
    avg_cost_per_query: float
    passed: bool
    failures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_records": self.num_records,
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": round(self.mrr, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "faithfulness_score": round(self.faithfulness_score, 4),
            "fallback_rate": round(self.fallback_rate, 4),
            "latency": {k: round(v, 4) for k, v in self.latency.items()},
            "avg_cost_per_query": round(self.avg_cost_per_query, 6),
            "passed": self.passed,
            "failures": self.failures,
        }


def compute_report(
    records: List[EvalRecord],
    k_values: List[int] = [1, 3, 5],
    min_iou: float = 0.5,
    eigenscore_threshold: float = -2.0,
    max_hallucination_rate: float = 0.05,
    max_latency_p95_seconds: Optional[float] = None,
    min_faithfulness: Optional[float] = None,
) -> EvalReport:
    """Compute the full metric suite and gate on the given thresholds."""
    precision = {k: precision_at_k(records, k, min_iou) for k in k_values}
    recall = {k: recall_at_k(records, k, min_iou) for k in k_values}
    mrr = mean_reciprocal_rank(records, min_iou)
    hall_rate = hallucination_rate(records, eigenscore_threshold)
    faithfulness = faithfulness_score(records)
    fb_rate = fallback_rate(records)
    latency = latency_percentiles(records)
    cost = cost_per_query(records)

    failures = []
    if hall_rate > max_hallucination_rate:
        failures.append(
            f"hallucination_rate {hall_rate:.2%} exceeds threshold {max_hallucination_rate:.2%}"
        )
    if max_latency_p95_seconds is not None and latency.get("p95", 0.0) > max_latency_p95_seconds:
        failures.append(
            f"p95 latency {latency['p95']:.2f}s exceeds SLA {max_latency_p95_seconds:.2f}s"
        )
    if min_faithfulness is not None and faithfulness < min_faithfulness:
        failures.append(
            f"faithfulness {faithfulness:.2%} below minimum {min_faithfulness:.2%}"
        )

    return EvalReport(
        num_records=len(records),
        precision_at_k=precision,
        recall_at_k=recall,
        mrr=mrr,
        hallucination_rate=hall_rate,
        faithfulness_score=faithfulness,
        fallback_rate=fb_rate,
        latency=latency,
        avg_cost_per_query=cost,
        passed=len(failures) == 0,
        failures=failures,
    )
