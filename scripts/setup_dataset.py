#!/usr/bin/env python3
"""
Setup LegalBench-RAG Dataset
Downloads and prepares the LegalBench dataset for use with LegalInsight
"""
import os
import sys
import json
import urllib.request
import zipfile
from pathlib import Path
from tqdm import tqdm

DATASET_URL = "https://github.com/HazyResearch/legalbench/archive/refs/heads/main.zip"
LEGALBENCH_RAG_URL = "https://huggingface.co/datasets/nguha/legalbench"

class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url: str, output_path: Path):
    """Download file with progress bar"""
    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

    print(f"\nDownload complete: {output_path}")

def setup_legalbench():
    """Setup LegalBench dataset"""
    print("=" * 60)
    print("LegalInsight - LegalBench Dataset Setup")
    print("=" * 60)

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "legalbench"

    print("\n1. Creating directory structure...")
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "corpus").mkdir(exist_ok=True)
    (data_dir / "queries").mkdir(exist_ok=True)

    print("\n2. Downloading LegalBench dataset...")
    print("\nNote: The full LegalBench dataset is large (~several GB)")
    print("For this project, we'll use the mini version (776 queries)")

    print("\n3. Creating sample dataset configuration...")

    sample_config = {
        "dataset": "LegalBench-RAG",
        "version": "mini",
        "total_queries": 776,
        "datasets": ["ContractNLI", "CUAD", "MAUD", "PrivacyQA"],
        "corpus_path": str(data_dir / "corpus"),
        "queries_path": str(data_dir / "queries" / "legalbench_queries_mini.json")
    }

    config_path = data_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(sample_config, f, indent=2)

    print(f"Configuration saved to: {config_path}")

    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("\nTo use the full LegalBench dataset:")
    print("\n1. Visit HuggingFace:")
    print(f"   {LEGALBENCH_RAG_URL}")
    print("\n2. Download the dataset files:")
    print("   - legalbench_queries_mini.json (for 776 queries)")
    print("   - legalbench_corpus/ (legal documents)")
    print(f"\n3. Place files in: {data_dir}")
    print("\n4. Update config.yaml with your dataset paths")

    print("\n" + "=" * 60)
    print("ALTERNATIVE: Use your own legal documents")
    print("=" * 60)
    print(f"\nPlace your legal documents in: {data_dir / 'corpus'}")
    print("Supported formats: .txt, .pdf (will be converted)")

    print("\n4. Creating sample legal documents for testing...")

    samples = [
        {
            "filename": "sample_contract_1.txt",
            "content": """EMPLOYMENT AGREEMENT

This Employment Agreement is entered into as of March 1, 2024, between Tech Innovations Inc. and Jane Smith.

1. POSITION: Employee shall serve as Senior Software Engineer.

2. COMPENSATION: Annual salary of $120,000, payable bi-weekly.

3. BENEFITS: Health insurance, 401(k) matching up to 6%, and 15 days PTO annually.

4. TERM: Employment begins March 1, 2024. Either party may terminate with 30 days notice.

5. CONFIDENTIALITY: Employee agrees to maintain confidentiality of proprietary information.

6. NON-COMPETE: Employee agrees not to work for direct competitors for 12 months post-termination.
"""
        },
        {
            "filename": "sample_contract_2.txt",
            "content": """LEASE AGREEMENT

This Lease Agreement is made on February 1, 2024, between Property LLC (Landlord) and John Doe (Tenant).

1. PROPERTY: 123 Main Street, Apartment 4B, City, State 12345

2. TERM: 12 months beginning March 1, 2024, ending February 28, 2025.

3. RENT: $2,000 per month, due on the 1st of each month. Late fees of $50 apply after the 5th.

4. SECURITY DEPOSIT: $4,000, refundable upon move-out subject to property condition.

5. UTILITIES: Tenant responsible for electricity and internet. Landlord covers water and trash.

6. PETS: One cat permitted with $300 non-refundable pet deposit.

7. MAINTENANCE: Landlord responsible for structural repairs. Tenant handles minor repairs under $100.
"""
        },
        {
            "filename": "sample_contract_3.txt",
            "content": """SOFTWARE LICENSE AGREEMENT

This Agreement is between DataCorp Solutions ("Licensor") and Enterprise Client Inc. ("Licensee").

1. LICENSE GRANT: Licensor grants Licensee a non-exclusive, non-transferable license to use the Software.

2. TERM: This license is perpetual, subject to compliance with terms.

3. FEES: One-time license fee of $50,000, plus annual maintenance of $10,000.

4. SUPPORT: Licensor provides 24/7 email support and quarterly updates.

5. RESTRICTIONS: Licensee may not reverse engineer, sublicense, or distribute the Software.

6. WARRANTY: Software provided "as is" with 90-day warranty for material defects.

7. LIMITATION OF LIABILITY: Licensor's liability limited to fees paid in the 12 months prior to claim.

8. TERMINATION: Licensor may terminate if Licensee breaches terms and fails to cure within 30 days.
"""
        }
    ]

    corpus_dir = data_dir / "corpus"
    for sample in samples:
        sample_path = corpus_dir / sample["filename"]
        with open(sample_path, 'w') as f:
            f.write(sample["content"])
        print(f"Created: {sample_path}")

    print("\n" + "=" * 60)
    print("Dataset setup complete")
    print("=" * 60)
    print(f"\nDataset location: {data_dir}")
    print("\nYou can now:")
    print("1. Use the sample documents for testing")
    print("2. Add your own legal documents to the corpus directory")
    print("3. Run the backend API to start analyzing contracts")

if __name__ == "__main__":
    setup_legalbench()
