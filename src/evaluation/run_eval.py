"""
CLI entry point for the eval CI/CD gate.

Usage:
    python -m src.evaluation.run_eval --sample-size 100 --output results/eval_report.json

By default this runs a retrieval-only evaluation using a lightweight
sentence-transformers embedding model, so it can run on every push without
requiring the multi-GB Self-RAG GGUF weights. Pass --with-generation with
SELFRAG_MODEL_PATH set to also gate on hallucination rate / faithfulness
using the full generation pipeline (intended for a scheduled/manual job
where the model weights are cached, not every push).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.evaluation.golden_dataset import load_golden_dataset
from src.evaluation.metrics import EvalRecord, compute_report
from src.retrieval.embedding import EmbeddingModel
from src.retrieval.retriever import LegalRetriever


def build_retrieval_only_records(examples, embedding_model_name, top_k):
    embedding_model = EmbeddingModel(model_name=embedding_model_name, device="cpu")
    retriever = LegalRetriever(embedding_model=embedding_model, top_k=top_k)

    documents = []
    seen = set()
    for ex in examples:
        passage = ex.get("passage", "")
        if passage and passage not in seen:
            seen.add(passage)
            documents.append({"text": passage, "metadata": {}})
    retriever.index_documents(documents, chunk_documents=False)

    records = []
    for ex in examples:
        start = time.time()
        results = retriever.retrieve(ex["question"], top_k=top_k)
        latency = time.time() - start
        records.append(EvalRecord(
            query=ex["question"],
            gold_passage=ex.get("passage"),
            retrieved_passages=[r["text"] for r in results],
            latency_seconds=latency,
        ))
    return records


def build_generation_records(examples, model_path, top_k, max_tokens=200):
    from src.self_rag.gguf_inference import SelfRAGGGUFInference

    inference = SelfRAGGGUFInference(model_path=model_path)
    records = []
    for i, ex in enumerate(examples):
        print(f"  generating {i + 1}/{len(examples)}...", file=sys.stderr)
        start = time.time()
        output = inference.generate(ex["question"], passage=ex.get("passage"), max_tokens=max_tokens)
        latency = time.time() - start
        records.append(EvalRecord(
            query=ex["question"],
            gold_passage=ex.get("passage"),
            retrieved_passages=[ex.get("passage")] if ex.get("passage") else [],
            predicted_answer=output.answer,
            issup=output.issup,
            isrel=output.isrel,
            latency_seconds=latency,
        ))
    return records


def main():
    parser = argparse.ArgumentParser(description="Run the LegalInsight eval CI/CD gate")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-hallucination-rate", type=float, default=0.05)
    parser.add_argument("--max-latency-p95-seconds", type=float, default=5.0)
    parser.add_argument("--min-faithfulness", type=float, default=None)
    parser.add_argument("--output", default="eval_report.json")
    parser.add_argument(
        "--with-generation", action="store_true",
        help="Also run the Self-RAG generation pipeline (requires SELFRAG_MODEL_PATH)",
    )
    parser.add_argument("--max-tokens", type=int, default=200,
                         help="Max generation tokens per example (bounds CPU inference time)")
    args = parser.parse_args()

    examples = load_golden_dataset(path=args.dataset, sample_size=args.sample_size, seed=args.seed)
    print(f"Loaded {len(examples)} golden examples")

    model_path = os.getenv("SELFRAG_MODEL_PATH")
    if args.with_generation and model_path and os.path.exists(model_path):
        print(f"Running full generation eval with model at {model_path}")
        records = build_generation_records(examples, model_path, args.top_k, args.max_tokens)
    else:
        if args.with_generation:
            print(
                "SELFRAG_MODEL_PATH not set or model file missing -- falling back "
                "to retrieval-only eval",
                file=sys.stderr,
            )
        print("Running retrieval-only eval")
        records = build_retrieval_only_records(examples, args.embedding_model, args.top_k)

    report = compute_report(
        records,
        k_values=args.k_values,
        max_hallucination_rate=args.max_hallucination_rate,
        max_latency_p95_seconds=args.max_latency_p95_seconds,
        min_faithfulness=args.min_faithfulness,
    )

    report_dict = report.to_dict()
    print(json.dumps(report_dict, indent=2))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    if not report.passed:
        print(f"EVAL GATE FAILED: {report.failures}", file=sys.stderr)
        sys.exit(1)

    print("EVAL GATE PASSED")


if __name__ == "__main__":
    main()
