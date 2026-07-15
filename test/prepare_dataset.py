#!/usr/bin/env python3
"""
Convert LegalBench-RAG benchmark format to simplified format for evaluation.

This script reads the full LegalBench-RAG dataset and converts it into
a format that's easier to work with in the performance evaluation notebook.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any


def load_benchmark_files(benchmarks_dir: str) -> List[Dict[str, Any]]:
    """Load all benchmark JSON files from the benchmarks directory."""
    benchmarks_path = Path(benchmarks_dir)
    all_tests = []

    print(f"Loading benchmark files from: {benchmarks_path}")

    for json_file in benchmarks_path.glob("*.json"):
        print(f"  Loading: {json_file.name}")
        with open(json_file, 'r') as f:
            data = json.load(f)
            if 'tests' in data:
                all_tests.extend(data['tests'])
            else:
                # If it's already a list of tests
                if isinstance(data, list):
                    all_tests.extend(data)
                else:
                    all_tests.append(data)

    print(f"Total queries loaded: {len(all_tests)}")
    return all_tests


def load_corpus_file(corpus_dir: str, file_path: str) -> str:
    """Load content from a corpus file."""
    full_path = Path(corpus_dir) / file_path
    with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def convert_to_qa_format(
    benchmarks_dir: str,
    corpus_dir: str,
    output_file: str,
    max_queries: int = None
) -> None:
    """
    Convert LegalBench-RAG format to simplified Q&A format.

    Args:
        benchmarks_dir: Path to benchmarks directory
        corpus_dir: Path to corpus directory
        output_file: Path to output JSON file
        max_queries: Maximum number of queries to process (None for all)
    """
    tests = load_benchmark_files(benchmarks_dir)

    if max_queries:
        tests = tests[:max_queries]
        print(f"Processing {max_queries} queries only")

    qa_data = []
    corpus_cache = {}  # Cache corpus files to avoid repeated reads

    print(f"\nConverting {len(tests)} queries...")

    for idx, test in enumerate(tests):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(tests)} queries...")

        query = test.get('query', '')
        snippets = test.get('snippets', [])
        tags = test.get('tags', [])

        # Combine all ground truth snippets as the "passage"
        passages = []
        for snippet in snippets:
            file_path = snippet.get('file_path', '')
            span = snippet.get('span', [0, 0])

            if file_path not in corpus_cache:
                try:
                    corpus_cache[file_path] = load_corpus_file(corpus_dir, file_path)
                except Exception as e:
                    print(f"    Warning: Could not load {file_path}: {e}")
                    continue

            content = corpus_cache[file_path]
            snippet_text = content[span[0]:span[1]]
            passages.append(snippet_text)

        combined_passage = " ... ".join(passages)

        qa_data.append({
            'question': query,
            'passage': combined_passage,
            'answer': combined_passage[:500] + "..." if len(combined_passage) > 500 else combined_passage,
            'tags': tags,
            'source': f"LegalBench-RAG query {idx + 1}"
        })

    with open(output_file, 'w') as f:
        json.dump(qa_data, f, indent=2)

    print(f"\nConverted {len(qa_data)} queries")
    print(f"Saved to: {output_file}")
    print(f"\nDataset Statistics:")
    print(f"  Total queries: {len(qa_data)}")
    print(f"  Average passage length: {sum(len(q['passage']) for q in qa_data) / len(qa_data):.0f} characters")
    print(f"  Total characters: {sum(len(q['passage']) for q in qa_data):,}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert LegalBench-RAG dataset to simplified Q&A format"
    )
    parser.add_argument(
        '--benchmarks-dir',
        default='legalbench_full/benchmarks',
        help='Path to benchmarks directory'
    )
    parser.add_argument(
        '--corpus-dir',
        default='legalbench_full/corpus',
        help='Path to corpus directory'
    )
    parser.add_argument(
        '--output',
        default='data/full_legalbench_qa.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--max-queries',
        type=int,
        default=None,
        help='Maximum number of queries to process (default: all)'
    )

    args = parser.parse_args()

    if not os.path.exists(args.benchmarks_dir):
        print(f"Error: Benchmarks directory not found: {args.benchmarks_dir}")
        print("\nPlease run ./download_dataset.sh first to download the full dataset.")
        return 1

    if not os.path.exists(args.corpus_dir):
        print(f"Error: Corpus directory not found: {args.corpus_dir}")
        print("\nPlease run ./download_dataset.sh first to download the full dataset.")
        return 1

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    convert_to_qa_format(
        args.benchmarks_dir,
        args.corpus_dir,
        args.output,
        args.max_queries
    )

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
