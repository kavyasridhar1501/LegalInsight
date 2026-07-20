"""Evaluation metrics and CI/CD gate for the legal RAG pipeline."""

from src.evaluation.metrics import EvalRecord, EvalReport, compute_report
from src.evaluation.golden_dataset import load_golden_dataset

__all__ = ["EvalRecord", "EvalReport", "compute_report", "load_golden_dataset"]
