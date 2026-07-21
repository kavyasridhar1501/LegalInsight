#!/usr/bin/env python3
"""
Generate full_legalbench_qa.json from available legal datasets.

This script downloads legal datasets from HuggingFace and creates
a comprehensive Q&A dataset for performance evaluation.
"""

import json
import os
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

def download_cuad_dataset(max_samples=None):
    """Download CUAD (Contract Understanding Atticus Dataset) from HuggingFace."""
    print("Downloading CUAD dataset from HuggingFace...")

    try:
        dataset = load_dataset("theatticusproject/cuad-qa", split="train")

        qa_pairs = []
        print(f"Loaded {len(dataset)} samples from CUAD")

        samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

        for idx, item in enumerate(tqdm(samples, desc="Processing CUAD")):
            # CUAD format: context, question, answers
            context = item.get('context', '')
            question = item.get('question', '')
            answers = item.get('answers', {})

            answer_text = ''
            if answers and 'text' in answers:
                answer_texts = answers['text']
                if answer_texts:
                    answer_text = answer_texts[0] if isinstance(answer_texts, list) else str(answer_texts)

            if context and question:
                qa_pairs.append({
                    'question': question,
                    'passage': context,
                    'answer': answer_text if answer_text else context[:500] + "...",
                    'tags': ['CUAD', 'contract-review'],
                    'source': f'CUAD query {idx + 1}'
                })

        print(f"Processed {len(qa_pairs)} Q&A pairs from CUAD")
        return qa_pairs

    except Exception as e:
        print(f"Error loading CUAD: {e}")
        return []

def download_legalbench_consumer_contracts(max_samples=None):
    """Download LegalBench consumer contracts dataset."""
    print("\nDownloading LegalBench Consumer Contracts...")

    try:
        dataset = load_dataset("mteb/legalbench_consumer_contracts_qa", split="test")

        qa_pairs = []
        print(f"Loaded {len(dataset)} samples")

        samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

        for idx, item in enumerate(tqdm(samples, desc="Processing LegalBench")):
            query = item.get('query', item.get('question', ''))
            corpus = item.get('corpus', '')

            if query:
                qa_pairs.append({
                    'question': query,
                    'passage': corpus,
                    'answer': corpus[:500] + "..." if len(corpus) > 500 else corpus,
                    'tags': ['LegalBench', 'consumer-contracts'],
                    'source': f'LegalBench Consumer Contracts query {idx + 1}'
                })

        print(f"Processed {len(qa_pairs)} Q&A pairs from LegalBench")
        return qa_pairs

    except Exception as e:
        print(f"Error loading LegalBench: {e}")
        return []

def download_contractnli(max_samples=None):
    """Download ContractNLI dataset."""
    print("\nDownloading ContractNLI...")

    try:
        dataset = load_dataset("coastalcph/lex_glue", "contractnli", split="train")

        qa_pairs = []
        print(f"Loaded {len(dataset)} samples")

        samples = dataset if max_samples is None else dataset.select(range(min(max_samples, len(dataset))))

        for idx, item in enumerate(tqdm(samples, desc="Processing ContractNLI")):
            premise = item.get('premise', '')
            hypothesis = item.get('hypothesis', '')

            if premise and hypothesis:
                question = f"Does the following statement apply to this contract: {hypothesis}"
                qa_pairs.append({
                    'question': question,
                    'passage': premise,
                    'answer': f"Based on the contract: {hypothesis}",
                    'tags': ['ContractNLI', 'contract-inference'],
                    'source': f'ContractNLI query {idx + 1}'
                })

        print(f"Processed {len(qa_pairs)} Q&A pairs from ContractNLI")
        return qa_pairs

    except Exception as e:
        print(f"Error loading ContractNLI: {e}")
        return []

def main():
    """Main function to generate the full dataset."""
    print("="*60)
    print("LegalBench-RAG Full Dataset Generator")
    print("="*60)
    print("\nThis will download legal datasets from HuggingFace and")
    print("create a comprehensive full_legalbench_qa.json file.\n")

    all_qa_pairs = []

    # Sample counts chosen to give a substantial but manageable dataset; CUAD has ~13,000+ samples total
    cuad_data = download_cuad_dataset(max_samples=4000)
    all_qa_pairs.extend(cuad_data)

    legalbench_data = download_legalbench_consumer_contracts(max_samples=1000)
    all_qa_pairs.extend(legalbench_data)

    contractnli_data = download_contractnli(max_samples=1858)
    all_qa_pairs.extend(contractnli_data)

    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total Q&A pairs generated: {len(all_qa_pairs):,}")
    print(f"\nBreakdown:")
    print(f"  - CUAD: {len(cuad_data):,}")
    print(f"  - LegalBench Consumer Contracts: {len(legalbench_data):,}")
    print(f"  - ContractNLI: {len(contractnli_data):,}")

    total_chars = sum(len(q['passage']) for q in all_qa_pairs)
    avg_length = total_chars / len(all_qa_pairs) if all_qa_pairs else 0

    print(f"\nText Statistics:")
    print(f"  - Total characters: {total_chars:,}")
    print(f"  - Average passage length: {avg_length:.0f} characters")
    print(f"  - Estimated pages: {total_chars / 3000:.0f}")

    output_path = Path(__file__).parent.parent / "data" / "full_legalbench_qa.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(all_qa_pairs, f, indent=2)

    print(f"\nDataset saved to: {output_path}")
    print(f"\nYou can now run the performance evaluation notebook.")
    print("="*60)

if __name__ == '__main__':
    main()
