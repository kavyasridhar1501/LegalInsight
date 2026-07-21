# LegalInsight

**AI-Powered Legal Contract Analysis with Self-RAG and Hallucination Detection**

**[Live Demo](https://kavyasridhar1501.github.io/LegalInsight/)**

---

## What is LegalInsight?

LegalInsight is an AI-powered system for legal contract analysis built on **Self-RAG (Self-Reflective Retrieval-Augmented Generation)**: real document retrieval, a self-healing retrieve → generate → critique → retry loop, and PII/injection guardrails on both input and output.

Paste or upload a contract, ask a question in natural language, and get an answer that's either grounded in the retrieved contract text or an honest "I don't have enough information" — never a guess presented as fact.

The frontend (`frontend/`, deployed to GitHub Pages via `docs/`) is a single static page with no configuration screen: it always talks to the LegalInsight backend, which holds its own LLM API key server-side. Nothing to set up, no API key to paste in.

- **Real retrieval**: FAISS + sentence-transformers embeddings find the relevant clause for each question, not just the whole document dumped into a prompt.
- **Self-healing loop**: if the critic finds the retrieved passage irrelevant, it reformulates the query and retries; if the passage was relevant but the answer wasn't well-supported, it resamples the generation — see `src/self_rag/self_healing_graph.py`.
- **Guardrails**: PII/prompt-injection detection on input, toxicity/schema/policy checks on output (`src/guardrails/`).
- **Structured key-term extraction** and **conversational follow-ups**, both running through the same backend.

---

## Dataset

### LegalBench-RAG Dataset

LegalInsight includes the **full LegalBench-RAG dataset** with 6,858 legal contract queries.

**Dataset Statistics:**
- **Total Queries**: 6,858
- **Total Characters**: ~1.96 million
- **Estimated Pages**: ~654
- **Sources**: CUAD, ContractNLI, MAUD, PrivacyQA

**Query Types:**
1. Termination conditions
2. Party identification
3. Payment terms
4. Liability limitations
5. Confidentiality obligations
6. Governing law
7. Indemnification provisions
8. Contract duration/term
9. Warranty provisions
10. Dispute resolution

---

## Features

### Core Capabilities

- **Real RAG**: FAISS + sentence-transformers retrieval finds the relevant clause per question; only that passage (not the whole contract) grounds the answer.
- **Self-Healing Loop**: retrieve → generate → critique → (reformulate & retry, or resample) → answer or honest fallback. See `src/self_rag/self_healing_graph.py`.
- **Guardrails**: PII/prompt-injection detection on input; toxicity, schema, and policy checks on output. See `src/guardrails/`.
- **Structured Key-Term Extraction**: parties, dates, payment terms, liability cap, and risks as structured JSON, rendered as a card.
- **Conversational Follow-ups**: each follow-up re-runs the same retrieve/critique loop against the contract for the new question.
- **PDF Support**: Upload and analyse PDF contracts directly in your browser.
- **Time Tracking**: Quantifiable efficiency improvements vs. manual contract review.

### User Experience

- **No setup**: no API key to enter, no configuration screen — the backend holds its own LLM API key server-side.
- **Responsive Design**: Works on desktop, tablet, and mobile.
- **Real-time Analytics**: Track total analyses, time saved, and efficiency across sessions (stored locally in your browser).

### Security & Privacy

- **No client-side API keys**: the frontend never asks for or stores a key; all LLM calls happen server-side.
- **Guardrails on every request**: PII in a query gets blocked before it's sent to the model; toxic/off-topic/policy-violating output gets withheld before it's shown to you.
- **No Data Storage**: contract text is used only for the current request; it isn't persisted server-side.

---

## Technology Stack

### Frontend
- **Core**: HTML5, CSS3, Vanilla JavaScript (ES6+), no build step
- **PDF Processing**: PDF.js for client-side PDF parsing
- **Storage**: Browser `localStorage` for session analytics only (no keys, no contract text)
- **Deployment**: GitHub Pages (`docs/`, fully static, kept in sync with `frontend/`)

### Backend (what the frontend actually talks to)
- **Framework**: Flask (Python), deployed to Railway (`railway.toml`) or Render (`render.yaml`)
- **Generation engine**: hosted LLM API by default (`GENERATION_BACKEND=openai`/`anthropic`, `src/self_rag/llm_api_inference.py`) — structured critique prompt returns `{answer, isrel, issup}` as JSON. `local_gguf` mode (the original local Self-RAG 7B model) is also supported.
- **Retrieval**: FAISS + sentence-transformers embeddings (`src/retrieval/`)
- **Self-Healing Orchestration**: LangGraph (`src/self_rag/self_healing_graph.py`)
- **Guardrails**: PII/injection/schema/policy gateway (`src/guardrails/`)
- **Dataset**: LegalBench-RAG (6,858 queries) for evaluation, `src/evaluation/`
- **Eval CI/CD**: Golden-dataset regression gate (`.github/workflows/eval.yml`), plus `scripts/run_full_benchmark.py` for on-demand real-model benchmarking (see [Performance Metrics](#performance-metrics) below).

---

## Architecture

```
┌────────────────────────────┐        ┌──────────────────────────────────────┐
│   Browser (GitHub Pages)   │        │      LegalInsight Backend (Railway)    │
│                            │        │                                        │
│  PDF.js parser             │  HTTPS │  Flask (backend/api.py)               │
│  contract text / query ────┼───────▶│    ├─ Guardrails (input check)        │
│  input, results display    │        │    ├─ Retrieval (FAISS + embeddings)  │
│  (app.js, no config UI,    │        │    ├─ Self-healing graph (LangGraph)  │
│   no API key)              │◀───────┼──  │   retrieve→generate→critique→    │
│                            │        │    │   retry/resample/fallback        │
└────────────────────────────┘        │    ├─ Generation engine               │
                                       │    │   (hosted LLM API by default,    │
                                       │    │    local Self-RAG GGUF optional) │
                                       │    └─ Guardrails (output check)       │
                                       └──────────────────────────────────────┘
```

The frontend (`frontend/`, mirrored to `docs/` for GitHub Pages) is a thin client: PDF parsing and result rendering only. Every analysis, follow-up, and key-term extraction is a request to the one backend, which holds its own LLM API key server-side and runs the actual Self-RAG pipeline (`src/self_rag/`, `src/retrieval/`, `src/guardrails/`).

---

## Performance Metrics

Backend reliability metrics (retrieval precision/recall/MRR, guardrails
false-positive/catch rates, and — when a Self-RAG GGUF model is available —
hallucination rate and faithfulness with vs. without the self-healing loop)
are generated by `scripts/run_full_benchmark.py` and persisted at
[`results/benchmark_results.md`](results/benchmark_results.md).

### Calculation Methodology

**Manual Time Estimation:**
```
Pages = Contract Length ÷ 3,000 characters
Manual Time = Pages × 7.5 minutes/page
```
*Based on industry standard of 7.5 minutes per page for legal contract review.*

**Performance Metrics:**
```
Time Saved = Manual Time - AI Time
Efficiency = (Time Saved ÷ Manual Time) × 100%
Speedup Factor = Manual Time ÷ AI Time
```

### Hallucination Detection

Rather than generating multiple responses and scoring their similarity after the fact, the backend prevents hallucinated answers from being returned in the first place:

| Result shown | Meaning |
|---|---|
| **Grounded** | The critic (ISREL/ISSUP-equivalent) found the retrieved passage relevant and the answer supported by it. |
| **Insufficient Info** | After retries, no answer could be grounded in the retrieved contract text — the pipeline returned an honest fallback instead of a guess. |

**Methodology:** retrieve → generate → critique (is the passage relevant? is the answer supported by it?) → on rejection, reformulate the query and re-retrieve, or resample generation on the same passage → accept or fall back after `max_attempts`. See [Self-Healing RAG Loop](#self-healing-rag-loop) below and `src/self_rag/self_healing_graph.py`.

Real measured numbers (oracle-passage baseline vs. this loop) are in [`results/benchmark_results.md`](results/benchmark_results.md).

---

## Backend Reliability Pipeline (Self-RAG server)

The Flask backend (`backend/api.py`) is what makes this a real RAG system
instead of a static page calling a provider blind: real document retrieval,
a self-healing retrieve → generate → critique → retry loop, and PII/
injection/policy guardrails on input and output. The frontend has no
configuration screen and no provider choice — every request from
`frontend/app.js` (mirrored in `docs/app.js`) goes to this backend, whose
URL is the single `BACKEND_URL` constant near the top of that file.

### Generation engine: hosted API by default, local model optional

`GENERATION_BACKEND` picks what answers and self-critiques the retrieved
passage:
- **`openai` (default) / `anthropic`** — `src/self_rag/llm_api_inference.py`
  calls a hosted API with a structured critique prompt (the model returns
  `{"answer", "isrel", "issup"}` as JSON). Fast (2-5s), no model to host,
  requires `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
- **`local_gguf`** — the original local Self-RAG 7B model
  (`src/self_rag/gguf_inference.py`), using its native reflection tokens.
  No per-query API cost, but slow on CPU (15-20s/query) and needs
  `pip install llama-cpp-python` plus the model downloaded via
  `python scripts/download_model.py` (~4.1GB, set `SELFRAG_MODEL_PATH`).

Either way, the retrieve → critique → retry graph
(`src/self_rag/self_healing_graph.py`) is identical — it only depends on a
`.generate(question, passage, temperature)` method, not on which engine
implements it.

### Deploying to Railway

The live demo's backend is already deployed this way. To deploy your own:

1. Push this repo to GitHub.
2. On [railway.app](https://railway.app), **New Project → Deploy from GitHub repo**, pick this repo. `railway.toml` is already configured (`python backend/api.py`, health check on `/health`).
3. In the Railway dashboard's **Variables** tab, set:
   - `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` + `GENERATION_BACKEND=anthropic`)
   - Railway sets `PORT` automatically — `backend/api.py` already reads it.
4. Deploy. Once live, copy the Railway-assigned URL (e.g. `https://your-app.up.railway.app`).
5. Open `frontend/app.js` **and** `docs/app.js`, set `BACKEND_URL` (near the top, in the State section) to that URL, commit, and push. (Two copies because `docs/` is what GitHub Pages actually serves; `frontend/` is kept in sync with it.)

Render works the same way via `render.yaml` (also already configured) if you prefer it over Railway.

**If running `local_gguf` and `llama-cpp-python` crashes with an
illegal-instruction / SIGILL error partway through generation** (not at
import or model-load time, but during actual inference): some virtualized/
cloud CPUs advertise AVX-512 support in `/proc/cpuinfo` that the hypervisor
doesn't reliably execute. The prebuilt wheel auto-detects and uses it, then
traps. Rebuild with AVX-512 disabled:
```bash
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON \
  -DGGML_F16C=ON -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF \
  -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF" \
  pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python
```
`GGML_NATIVE=OFF` is the important one — without it, `-march=native` at
compile time re-enables AVX-512 regardless of the individual flags above.

### Self-Healing RAG Loop

`src/self_rag/self_healing_graph.py` models retrieval as a stateful,
cyclical **LangGraph** workflow instead of a single retrieve-and-generate
pass:

```
retrieve → generate → critique ──accept──▶ return answer
                          │
                       reject
                          ▼
                    reformulate query → retrieve (retry)
                          │
                (still rejected after max_attempts)
                          ▼
              "I don't have enough information" (graceful fallback)
```

The critic rejects an answer when the Self-RAG reflection tokens indicate the
retrieved passage is irrelevant (`ISREL`) or the answer isn't supported by it
(`ISSUP`), or when EigenScore flags high semantic divergence across
resamples. On rejection, the query is reformulated (via the model itself, or
a rule-based fallback) and retrieval runs again, up to `max_attempts`.
Exposed via `POST /analyze_contract_self_healing`.

### Guardrails Gateway

`src/guardrails/` sits between the user and the LLM:
- **Input guardrails** (`input_guardrails.py`): detects and blocks prompt
  injection / jailbreak phrasing, and detects + redacts PII (credit cards
  via Luhn check, SSNs, emails, phone numbers) typed into a query.
- **Output guardrails** (`output_guardrails.py`): validates the structured
  key-term extraction JSON against a schema, and blocks toxic or off-topic
  responses.
- **Policy engine** (`policy.py`): a YAML rules file
  (`configs/guardrails_policy.yaml`) so non-engineers can add rules like
  "never discuss competitors" or "block medical advice" without touching
  code.

Both `/analyze_contract` and `/analyze_contract_self_healing` run input
checks before generation and output checks before returning a response; a
blocked response is replaced with an explanation of which rule fired instead
of being silently returned.

### LLM Eval CI/CD Pipeline

`src/evaluation/` turns the LegalBench-RAG dataset already bundled in this
repo (`data/full_legalbench_qa.json`, 6,858 labeled Q&A pairs) into a golden
regression set:
- **Metrics** (`metrics.py`): retrieval precision@k / recall@k / MRR,
  hallucination rate (EigenScore + ISSUP), faithfulness, fallback rate, and
  p50/p95 latency.
- **Runner** (`run_eval.py`): `python -m src.evaluation.run_eval
  --sample-size 100` samples the golden set, runs it through the pipeline,
  writes a JSON report, and exits non-zero if hallucination rate or latency
  breach a configurable threshold.
- **CI gate** (`.github/workflows/eval.yml`): runs on every push/PR —
  guardrail and self-healing unit tests always run; a retrieval-only
  precision/recall/MRR/latency gate runs using a lightweight embedding model
  (no GGUF download required). The full generation eval (hallucination rate,
  faithfulness against the Self-RAG model) is a manual/scheduled job, since
  it needs the multi-GB Self-RAG GGUF weights cached rather than downloaded
  on every push.

---

## Usage

1. **Upload Contract**
   - Drag & drop a PDF, click **Upload PDF Contract**, or paste text directly. No sign-in, no API key to enter.

2. **Analyse**
   - Enter an optional specific question
   - Click **Analyse Contract** for full analysis, or **Quick Summary** for a brief overview
   - The backend retrieves the relevant passage and runs it through the self-healing critique/retry loop

3. **Review Results**
   - **Performance Metrics**: Time saved, efficiency, speedup
   - **Analysis**: the grounded answer (or an honest "insufficient information" if nothing could be grounded)
   - **Guardrails**: whether the input/output passed PII, injection, toxicity, and policy checks
   - **Self-Healing Trace**: show/hide each retry attempt and why it was accepted or rejected

4. **Extract Key Terms**
   - Click **Extract Key Terms** in the results section
   - A card appears with parties, dates, payment terms, liability cap, and risks

5. **Ask Follow-up Questions**
   - After analysis, a **Follow-up Questions** panel appears
   - Type a question and press **Enter** or click **Ask** — each follow-up re-runs retrieval + the self-healing loop against the same contract
   - Click **Clear History** to reset the conversation

---

## Acknowledgments

- **LangGraph**: [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) — the self-healing retrieve/critique/retry state machine (`src/self_rag/self_healing_graph.py`)
- **Self-RAG**: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- **LegalBench-RAG**: [A Benchmark for Retrieval-Augmented Generation in the Legal Domain](https://arxiv.org/abs/2408.10343)
- **INSIDE**: [INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection](https://arxiv.org/abs/2402.03744)
