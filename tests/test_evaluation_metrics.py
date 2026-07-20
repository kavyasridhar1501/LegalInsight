from src.evaluation.metrics import (
    EvalRecord,
    compute_report,
    faithfulness_score,
    fallback_rate,
    hallucination_rate,
    latency_percentiles,
    mean_reciprocal_rank,
    precision_at_k,
    recall_at_k,
)


def make_record(**kwargs) -> EvalRecord:
    defaults = dict(query="q")
    defaults.update(kwargs)
    return EvalRecord(**defaults)


class TestRetrievalMetrics:
    def test_recall_at_k_hit_in_top_result(self):
        records = [make_record(gold_passage="Termination requires 30 days notice.",
                                retrieved_passages=["Termination requires 30 days notice.", "other"])]
        assert recall_at_k(records, k=1) == 1.0

    def test_recall_at_k_miss(self):
        records = [make_record(gold_passage="Termination requires 30 days notice.",
                                retrieved_passages=["completely unrelated passage text"])]
        assert recall_at_k(records, k=1) == 0.0

    def test_recall_at_k_hit_further_down_ranking(self):
        records = [make_record(gold_passage="Payment due within 45 days.",
                                retrieved_passages=["unrelated", "Payment due within 45 days.", "also unrelated"])]
        assert recall_at_k(records, k=1) == 0.0
        assert recall_at_k(records, k=2) == 1.0

    def test_precision_at_k(self):
        records = [make_record(gold_passage="Payment due within 45 days.",
                                retrieved_passages=["Payment due within 45 days.", "other"])]
        assert precision_at_k(records, k=1) == 1.0
        assert precision_at_k(records, k=2) == 0.5

    def test_mrr(self):
        records = [
            make_record(gold_passage="A", retrieved_passages=["A", "B"]),
            make_record(gold_passage="C", retrieved_passages=["B", "C"]),
        ]
        # first record hits at rank 1 (1/1), second at rank 2 (1/2) => mean = 0.75
        assert mean_reciprocal_rank(records) == 0.75

    def test_records_without_gold_passage_are_excluded(self):
        records = [make_record(gold_passage=None, retrieved_passages=["x"])]
        assert recall_at_k(records, k=1) == 0.0
        assert precision_at_k(records, k=1) == 0.0


class TestGenerationMetrics:
    def test_hallucination_rate_from_issup(self):
        records = [
            make_record(issup="[Fully supported]"),
            make_record(issup="[No support / Contradictory]"),
        ]
        assert hallucination_rate(records) == 0.5

    def test_hallucination_rate_from_eigenscore(self):
        records = [
            make_record(eigenscore=-3.0),
            make_record(eigenscore=1.0),
        ]
        assert hallucination_rate(records, eigenscore_threshold=-2.0) == 0.5

    def test_fallback_records_excluded_from_hallucination_denominator(self):
        records = [
            make_record(used_fallback=True, issup=None),
            make_record(issup="[Fully supported]"),
        ]
        assert hallucination_rate(records) == 0.0

    def test_faithfulness_score(self):
        records = [
            make_record(issup="[Fully supported]"),
            make_record(issup="[Partially supported]"),
            make_record(issup="[No support / Contradictory]"),
        ]
        assert faithfulness_score(records) == 0.5

    def test_fallback_rate(self):
        records = [make_record(used_fallback=True), make_record(used_fallback=False)]
        assert fallback_rate(records) == 0.5


class TestOperationalMetrics:
    def test_latency_percentiles(self):
        records = [make_record(latency_seconds=v) for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        result = latency_percentiles(records, percentiles=[50, 95])
        assert result["p50"] == 5
        assert result["p95"] == 10

    def test_latency_percentiles_empty(self):
        assert latency_percentiles([]) == {"p50": 0.0, "p95": 0.0}


class TestComputeReport:
    def test_passes_when_under_thresholds(self):
        records = [make_record(issup="[Fully supported]", eigenscore=-3.0, latency_seconds=1.0)]
        report = compute_report(records, max_hallucination_rate=0.5, max_latency_p95_seconds=5.0)
        assert report.passed
        assert report.failures == []

    def test_fails_when_hallucination_rate_exceeds_threshold(self):
        records = [make_record(issup="[No support / Contradictory]", latency_seconds=1.0)]
        report = compute_report(records, max_hallucination_rate=0.05)
        assert not report.passed
        assert any("hallucination_rate" in f for f in report.failures)

    def test_fails_when_latency_sla_exceeded(self):
        records = [make_record(issup="[Fully supported]", latency_seconds=10.0)]
        report = compute_report(records, max_hallucination_rate=1.0, max_latency_p95_seconds=1.0)
        assert not report.passed
        assert any("latency" in f for f in report.failures)

    def test_report_serializes_to_dict(self):
        records = [make_record(issup="[Fully supported]", latency_seconds=1.0)]
        report = compute_report(records)
        d = report.to_dict()
        assert "passed" in d and "hallucination_rate" in d
