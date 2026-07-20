"""
Loads the golden QA dataset used by the eval CI/CD pipeline.

Backed by data/full_legalbench_qa.json, which already ships 6,858 labeled
legal Q&A pairs (question / passage / answer / tags) -- more than the 100+
called for by a golden-dataset eval gate. `load_golden_dataset` gives a
deterministic sample so CI runs are fast and reproducible.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "full_legalbench_qa.json"


def load_golden_dataset(
    path: Optional[str] = None,
    sample_size: Optional[int] = None,
    seed: int = 42,
    tags: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Load golden QA examples.

    Args:
        path: Path to the golden dataset JSON. Defaults to data/full_legalbench_qa.json.
        sample_size: If set, deterministically sample this many examples.
        seed: Random seed for sampling, kept fixed so CI runs are reproducible.
        tags: If set, only keep examples whose `tags` intersect this list.
    """
    dataset_path = Path(path) if path else DEFAULT_DATASET_PATH
    if not dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at {dataset_path}")

    with open(dataset_path, "r") as f:
        examples = json.load(f)

    if tags:
        tag_set = set(tags)
        examples = [e for e in examples if tag_set.intersection(e.get("tags", []))]

    if sample_size is not None and sample_size < len(examples):
        rng = random.Random(seed)
        examples = rng.sample(examples, sample_size)

    return examples
