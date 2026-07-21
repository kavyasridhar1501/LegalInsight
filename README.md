# LegalInsight

**AI-powered legal contract analysis with Self-RAG and hallucination detection**

**[Live Demo](https://kavyasridhar1501.github.io/LegalInsight/)**

Paste or upload a contract, ask a question, get an answer grounded in the retrieved text — or an honest "I don't have enough information," never a guessed fact. No API key to set up; the backend holds its own key server-side.

---

## How it works

```
Browser (GitHub Pages)                LegalInsight Backend (Railway)
  PDF.js + app.js          HTTPS      Flask · guardrails (in) · FAISS retrieval
  no config, no API key  ────────▶    self-healing graph (LangGraph) · LLM
                          ◀────────   guardrails (out)
```

**Self-healing loop** (`src/self_rag/self_healing_graph.py`): retrieve → generate → critique → accept, or heal:
- passage irrelevant → reformulate the query and retry
- passage relevant but answer unsupported → resample generation on the same passage at a higher temperature
- still rejected after `max_attempts` → honest fallback, never a guess

**Guardrails** (`src/guardrails/`): PII/prompt-injection detection on input; toxicity, schema, and policy checks on output — both YAML-configurable (`configs/guardrails_policy.yaml`).

**Request isolation**: each request gets its own FAISS index, built and torn down per-request rather than shared across requests — closes a real cross-request contamination bug found on the live deployment (see `tests/test_retriever.py`).

## Tech stack

- **Frontend**: HTML/CSS/vanilla JS, PDF.js — static, deployed via GitHub Pages (`docs/`)
- **Backend**: Flask, deployed on Railway (`railway.toml`) / Render (`render.yaml`)
- **Retrieval**: FAISS + sentence-transformers (`src/retrieval/`)
- **Generation**: hosted LLM API by default (OpenAI/Anthropic, `src/self_rag/llm_api_inference.py`); local Self-RAG 7B GGUF optional (`src/self_rag/gguf_inference.py`)
- **Orchestration**: LangGraph (`src/self_rag/self_healing_graph.py`)
- **Eval**: LegalBench-RAG golden dataset (6,858 queries), CI gate on every push (`.github/workflows/eval.yml`)

## Results

Real numbers from `scripts/run_full_benchmark.py`, persisted at [`results/benchmark_results.md`](results/benchmark_results.md).

**Retrieval (n=150):** Precision@1 64.7% · Recall@5 91.3% · MRR 0.780 · p50/p95 latency 2.9ms / 3.2ms

**Guardrails (n=500 + 8 adversarial probes):** 0.0% false-positive rate · 100% adversarial catch rate (input & output)

**Generation, real Self-RAG model (n=30):** oracle-passage single-shot vs. this repo's self-healing loop

| | Oracle single-shot | Self-healing |
|---|---|---|
| Hallucination rate | 20.0% | 0.0% (among accepted) |
| Fallback rate | 0% | 13.3% |
| Faithfulness | 61.7% | -- |

The self-healing loop trades a 13% "insufficient info" fallback rate for zero hallucinations among the answers it does give.

## Running your own backend

1. Deploy this repo on [Railway](https://railway.app) (`railway.toml` is preconfigured) or [Render](https://render.com) (`render.yaml`).
2. Set `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` + `GENERATION_BACKEND=anthropic`) in the platform's dashboard.
3. Point `BACKEND_URL` in `frontend/app.js` **and** `docs/app.js` at your deployed URL, commit, push.

`GET /health` returns the deployed commit hash — useful for confirming a push actually rolled out.

Local model instead of a hosted API: `GENERATION_BACKEND=local_gguf`, `pip install llama-cpp-python`, `python scripts/download_model.py`. Hitting a SIGILL crash on a virtualized CPU? See the AVX-512 note in `scripts/download_model.py`.

## Tests

```
pytest tests/
```

69 tests covering guardrails, the self-healing graph, retriever isolation, LLM API inference, and eval metrics — all mocked, no network calls or model downloads required.

## Acknowledgments

- [Self-RAG](https://arxiv.org/abs/2310.11511) — retrieve/generate/critique architecture
- [LangGraph](https://github.com/langchain-ai/langgraph) — the self-healing state machine
- [LegalBench-RAG](https://arxiv.org/abs/2408.10343) — evaluation dataset
- [INSIDE](https://arxiv.org/abs/2402.03744) — hallucination detection via internal states
