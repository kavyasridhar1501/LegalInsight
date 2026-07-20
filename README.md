# LegalInsight

**AI-Powered Legal Contract Analysis with Self-RAG and Hallucination Detection**

**[Live Demo](https://kavyasridhar1501.github.io/LegalInsight-SelfRAG-HallucinationDetection/)**

---

## What is LegalInsight?

LegalInsight is an AI-powered system for legal contract analysis that combines **Self-RAG (Self-Reflective Retrieval-Augmented Generation)** with **hallucination detection**. It is powered by **[LangChain.js](https://js.langchain.com/)** running entirely in the browser.

Upload contracts, ask questions in natural language, and receive analysis while tracking time savings compared to manual review.

Built for legal professionals, LegalInsight provides:
- **Smart document chunking** via LangChain's `RecursiveCharacterTextSplitter` (true client-side RAG)
- **Structured prompt management** via LangChain's `ChatPromptTemplate`
- **Semantic hallucination detection** using LangChain-powered Jaccard similarity scoring
- **Structured key-term extraction** with LangChain's output parsing
- **Conversational follow-up** with LangChain conversation history formatting
- **Multiple AI providers** (OpenAI, Anthropic, Google Gemini, Groq, Cohere, Mistral)
- **Client-side processing** for privacy and security (no server required)

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

- **LangChain RAG Chunking**: Contracts are split into overlapping chunks with `RecursiveCharacterTextSplitter`. Only the most query-relevant sections are sent to the LLM, staying within token limits and focusing the model's attention.
- **LangChain Prompt Templates**: A single `ChatPromptTemplate` defines the legal analyst system message and user template, reused consistently across all six AI providers.
- **Hallucination Detection**: Three responses are generated at different temperatures and scored for semantic consistency using Jaccard word-overlap similarity (computed by the LangChain engine).
- **Structured Key-Term Extraction**: A dedicated LangChain extraction prompt asks the LLM for structured JSON — parties, dates, payment terms, liability cap, key risks — rendered as a clean card.
- **Conversational Follow-ups**: After each analysis, LangChain re-chunks the contract for the new question and injects the last three exchanges as conversation history, keeping follow-ups in context.
- **Contract Analysis**: Comprehensive summarisation, key term extraction, and risk identification.
- **PDF Support**: Upload and analyse PDF contracts directly in your browser.
- **Time Tracking**: Quantifiable efficiency improvements vs. manual contract review.

### User Experience

- **Client-Side Processing**: All analysis happens in your browser — no server required, your data stays private.
- **Multiple AI Providers**: OpenAI, Anthropic, Google Gemini, Groq, Cohere, or Mistral.
- **Demo Mode**: Try the system without an API key.
- **Responsive Design**: Works on desktop, tablet, and mobile.
- **Real-time Analytics**: Track total analyses, time saved, and efficiency across sessions.

### Security & Privacy

- **Local API Keys**: Stored only in your browser's `localStorage`.
- **No Server Communication**: Direct client-to-AI provider communication via HTTPS.
- **No Data Storage**: Contracts and queries are never saved or transmitted to any server.

---

## Technology Stack

### Frontend
- **Core**: HTML5, CSS3, Vanilla JavaScript (ES6+)
- **LangChain.js**: Loaded as an ES module via `esm.sh` CDN (no npm/build required):
  - `@langchain/core` — `ChatPromptTemplate`, message types, output parsers
  - `@langchain/textsplitters` — `RecursiveCharacterTextSplitter`
- **PDF Processing**: PDF.js for client-side PDF parsing
- **Storage**: Browser `localStorage` for API keys and analytics
- **Deployment**: GitHub Pages (fully static)

### AI Providers (via direct API calls)
- **OpenAI**: GPT-4o
- **Anthropic**: Claude 3.5 Sonnet
- **Google**: Gemini 1.5 Pro
- **Groq**: Llama 3.1 8B Instant
- **Cohere**: Command R+
- **Mistral AI**: Mistral Large

### Analysis Methodology
- **LangChain RAG**: `RecursiveCharacterTextSplitter` (2 000-char chunks, 200-char overlap) + keyword-scored chunk selection
- **LangChain Prompts**: `ChatPromptTemplate` for analysis, follow-up, and structured extraction
- **Hallucination Scoring**: Pairwise Jaccard similarity across three generated responses
- **Performance Tracking**: Time estimation based on industry standard (7.5 min/page)

### Backend (Optional — not required for GitHub Pages)
- **Framework**: Flask (Python)
- **Vector Store**: FAISS for semantic search
- **Embeddings**: BGE-M3 (sentence-transformers)
- **Dataset**: LegalBench-RAG (6,858 queries)
- **Self-Healing Orchestration**: LangGraph (`src/self_rag/self_healing_graph.py`)
- **Guardrails**: Custom PII/injection/schema/policy gateway (`src/guardrails/`)
- **Eval CI/CD**: Golden-dataset regression gate (`src/evaluation/`, `.github/workflows/eval.yml`)

---

## Architecture

### Client-Side Architecture (GitHub Pages Deployment)

```
┌──────────────────────────────────────────────────────────────────┐
│                         Browser (Client)                         │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                  LegalInsight Frontend                    │   │
│  │                                                           │   │
│  │  ┌─────────────┐   ┌──────────────────────────────────-┐  │   │
│  │  │  PDF.js     │   │     LangChain.js Engine           │  │   │
│  │  │  Parser     │   │  (langchain-engine.js, ES module) │  │   │
│  │  └──────┬──────┘   │                                   │  │   │
│  │         │          │  • RecursiveCharacterTextSplitter │  │   │
│  │         ↓          │  • ChatPromptTemplate             │  │   │
│  │  ┌──────────────┐  │  • Jaccard consistency scorer     │  │   │
│  │  │  Contract    │→ │  • Structured output parser       │  │   │
│  │  │  Text        │  │  • Conversation history formatter │  │   │
│  │  └──────────────┘  └──────────────┬────────────────────┘  │   │
│  │                                   │ formatted messages[]  │   │
│  │                                   ↓                       │   │
│  │                         ┌─────────────────┐               │   │
│  │                         │   app.js        │               │   │
│  │                         │  Provider calls │               │   │
│  │                         │  Results display│               │   │
│  │                         └────────┬────────┘               │   │
│  └──────────────────────────────────┼────────────────────────┘   │
│                                     │                            │
│                               localStorage                       │
│                          (API Keys, Analytics)                   │
└─────────────────────────────────────┬────────────────────────────┘
                                      │ HTTPS
                                      ↓
              ┌─────────────────────────────────────┐
              │         AI Provider APIs            │
              │  • OpenAI      • Anthropic          │
              │  • Gemini      • Groq               │
              │  • Cohere      • Mistral            │
              └─────────────────────────────────────┘
```

### LangChain Engine (`langchain-engine.js`)

The engine loads as an ES module and exposes `window.LegalInsightLC`. `app.js` waits up to 6 s for it to be ready and falls back gracefully if the CDN is slow.

| Component | LangChain API | Purpose |
|---|---|---|
| Document chunker | `RecursiveCharacterTextSplitter` | Split contracts into 2 000-char overlapping chunks |
| Chunk selector | keyword scoring | Pick the top-N most query-relevant chunks |
| Analysis prompt | `ChatPromptTemplate` | Structured legal analyst system + user template |
| Extraction prompt | `ChatPromptTemplate` | JSON key-term extraction template |
| Consistency scorer | Jaccard similarity | Compare word-overlap across 3 LLM responses |
| Message formatter | `HumanMessage`, `AIMessage` | Convert LangChain messages to provider `{role, content}[]` |
| Output parser | JSON parse + fallback regex | Extract structured data from LLM response |

### System Components

1. **Input Layer** — PDF parsing (PDF.js), text paste, query input
2. **LangChain Layer** — chunking, prompt formatting, message conversion
3. **AI Integration Layer** — direct API calls to selected provider (×3 for consistency scoring)
4. **Output Layer** — analysis display, hallucination risk, structured key terms, follow-up conversation

---

## Performance Metrics

Backend reliability metrics (retrieval precision/recall/MRR, guardrails
false-positive/catch rates, and — when a Self-RAG GGUF model is available —
hallucination rate and faithfulness with vs. without the self-healing loop)
are generated by `scripts/run_full_benchmark.py` and persisted at
[`results/benchmark_results.md`](results/benchmark_results.md).

### Evaluation Results

LegalInsight has been evaluated on a subset of the **LegalBench-RAG dataset** (100 legal contract queries).

### Time Savings Analysis

| Metric | Value |
|---|---|
| Total Queries Processed | 98 |
| Average Manual Time (min) | 1.66 |
| Average AI Time (sec) | 22.60 |
| Total Time Saved (hours) | 2.10 |
| Average Time Saved per Query (min) | 1.28 |
| Average Efficiency (%) | 64.40 |
| Average Speedup Factor | 4.26 |
| Average Consistency Score (%) | 74.65 |
| Median Efficiency (%) | 70.92 |
| Min Efficiency (%) | -77.10 |
| Max Efficiency (%) | 94.25 |

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

The system uses LangChain to generate and score three responses for consistency:

| Consistency Score | Risk Level | Interpretation |
|---|---|---|
| **85–100%** | Low | Highly consistent responses, reliable analysis |
| **70–84%** | Medium | Some variance, review carefully |
| **< 70%** | High | Significant variance, verify with source |

**Methodology (LangChain-powered):**
1. LangChain's `ChatPromptTemplate` formats the analysis prompt
2. Three responses are generated at temperatures 0.1, 0.5, and 0.9
3. The LangChain engine computes **pairwise Jaccard similarity** on each response's word sets
4. The average similarity (mapped to 0–100%) is displayed as the consistency score

This replaces the earlier length-variance proxy with a genuine semantic comparison.

---

## Backend Reliability Pipeline (Self-RAG server)

The optional Flask backend (`backend/api.py`) adds three subsystems on top of
Self-RAG + EigenScore, aimed specifically at improving the metrics above
(hallucination rate, faithfulness, and trustworthiness of what gets returned).
The frontend (`frontend/`, mirrored to `docs/` for the GitHub Pages deploy) has
a **"Self-RAG Reliability Backend"** option in the provider dropdown that
routes analysis through this backend instead of calling a provider directly —
enter the backend's URL, click **Initialize Backend** once, then analyze as
usual. Running it requires:
1. `python backend/api.py` (locally, or deployed via the included
   `render.yaml` / `railway.toml` / `Procfile`), and
2. A Self-RAG GGUF model downloaded to `data/models/` via
   `python scripts/download_model.py` (~4.1GB) — without it, `/initialize`
   reports the model as unavailable but the retriever still initializes.

**If `llama-cpp-python` crashes with an illegal-instruction / SIGILL error
partway through generation** (not at import or model-load time, but during
actual inference), it's almost certainly this: some virtualized/cloud CPUs
advertise AVX-512 support in `/proc/cpuinfo` that the hypervisor doesn't
reliably execute. The prebuilt wheel auto-detects and uses it, then traps.
Rebuild with AVX-512 disabled:
```bash
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON \
  -DGGML_F16C=ON -DGGML_AVX512=OFF -DGGML_AVX512_VBMI=OFF \
  -DGGML_AVX512_VNNI=OFF -DGGML_AVX512_BF16=OFF" \
  pip install --no-cache-dir --force-reinstall --no-binary llama-cpp-python llama-cpp-python
```
`GGML_NATIVE=OFF` is the important one — without it, `-march=native` at
compile time re-enables AVX-512 regardless of the individual flags above.

Everything else on this page (LangChain chunking, direct provider calls,
Jaccard consistency scoring) works exactly as before and needs neither of these.

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

1. **Configure API Provider**
   - Select provider from dropdown (OpenAI, Anthropic, Gemini, Groq, Cohere, Mistral, or Demo)
   - Enter your API key and click **Save Key** (stored in `localStorage` only)

2. **Upload Contract**
   - Drag & drop a PDF, click **Upload PDF Contract**, or paste text directly

3. **Analyse**
   - Enter an optional specific question
   - Click **Analyse Contract** for full analysis, or **Quick Summary** for a brief overview
   - LangChain chunks the contract and selects relevant sections automatically

4. **Review Results**
   - **Performance Metrics**: Time saved, efficiency, speedup
   - **RAG badge**: Shows how many sections LangChain selected (e.g. "Analysed 3 of 11 sections")
   - **Analysis**: AI-generated contract summary, grounded in the relevant chunks
   - **Hallucination Analysis**: Jaccard-based consistency score and risk level
   - **Verification Responses**: Show/hide the two additional generated responses

5. **Extract Key Terms** *(LangChain structured extraction)*
   - Click **Extract Key Terms** in the results section
   - LangChain sends a structured JSON extraction prompt to the LLM
   - A card appears with parties, dates, payment terms, liability cap, obligations, and risks

6. **Ask Follow-up Questions** *(LangChain conversation memory)*
   - After analysis, a **Follow-up Questions** panel appears
   - Type a question and press **Enter** or click **Ask**
   - LangChain re-chunks the contract for each question and injects the last three exchanges as history
   - Click **Clear History** to reset the conversation

---

## Acknowledgments

- **LangChain.js**: [js.langchain.com](https://js.langchain.com/) — document splitting, prompt templates, and message formatting used in `langchain-engine.js`
- **Self-RAG**: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- **LegalBench-RAG**: [A Benchmark for Retrieval-Augmented Generation in the Legal Domain](https://arxiv.org/abs/2408.10343)
- **INSIDE**: [INSIDE: LLMs' Internal States Retain the Power of Hallucination Detection](https://arxiv.org/abs/2402.03744)
