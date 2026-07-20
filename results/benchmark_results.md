# LegalInsight Benchmark Results

Generated: 2026-07-20T21:37:03.320980+00:00

## Retrieval quality (n=150)

| Metric | Value |
|---|---|
| Precision@1 | 64.7% |
| Recall@5 | 91.3% |
| MRR | 0.780 |
| Latency p50 / p95 | 9.0ms / 11.1ms |
| Unique passages in sample | 10 |

## Guardrails audit (n=500 + 8 adversarial probes)

| Metric | Value |
|---|---|
| Query false-positive rate | 0.0% |
| Answer false-positive rate | 0.0% |
| Adversarial input catch rate | 100.0% |
| Adversarial output catch rate | 100.0% |

## Generation quality with the real Self-RAG model (n=15)

| | Oracle-passage single-shot | Self-healing (real retrieval + retry) |
|---|---|---|
| Hallucination rate | 20.0% | 0.0% (among accepted) |
| Faithfulness | 56.7% | -- |
| Fallback rate | 0% (always answers) | 13.3% |
| Accepted rate | 100% | 86.7% |
| Avg attempts | 1 | 1.20 |
| Latency p50 | 17.8s | 18.2s |
