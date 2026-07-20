# LegalInsight Benchmark Results

Generated: 2026-07-20T22:18:51.085060+00:00

## Retrieval quality (n=150)

| Metric | Value |
|---|---|
| Precision@1 | 64.7% |
| Recall@5 | 91.3% |
| MRR | 0.780 |
| Latency p50 / p95 | 2.9ms / 3.2ms |
| Unique passages in sample | 10 |

## Guardrails audit (n=500 + 8 adversarial probes)

| Metric | Value |
|---|---|
| Query false-positive rate | 0.0% |
| Answer false-positive rate | 0.0% |
| Adversarial input catch rate | 100.0% |
| Adversarial output catch rate | 100.0% |

## Generation quality with the real Self-RAG model (n=30)

| | Oracle-passage single-shot | Self-healing (real retrieval + retry) |
|---|---|---|
| Hallucination rate | 20.0% | 0.0% (among accepted) |
| Faithfulness | 61.7% | -- |
| Fallback rate | 0% (always answers) | 13.3% |
| Accepted rate | 100% | 86.7% |
| Avg attempts | 1 | 1.20 |
| Latency p50 | 4.2s | 5.9s |
