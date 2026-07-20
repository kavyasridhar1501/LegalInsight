#!/usr/bin/env python3
"""
Download Self-RAG model for LegalInsight
Downloads the quantized GGUF model from HuggingFace
"""
import os
import sys
import urllib.request
from pathlib import Path
from tqdm import tqdm

MODELS = {
    # The official selfrag/selfrag_llama2_7b HuggingFace repo only ships the
    # PyTorch weights, not GGUF -- this community quantization is the one
    # that actually resolves. Verified working as of this writing.
    "selfrag-7b-q4": {
        "url": "https://huggingface.co/m4r1/selfrag_llama2_7b-GGUF/resolve/main/selfrag_llama2_7b.q4_k_m.gguf",
        "filename": "selfrag_llama2_7b.q4_k_m.gguf",
        "size": "4.1GB"
    },
}

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

def main():
    print("=" * 60)
    print("LegalInsight - Self-RAG Model Download")
    print("=" * 60)

    print("\nAvailable models:")
    for i, (key, model) in enumerate(MODELS.items(), 1):
        print(f"{i}. {key} ({model['size']})")

    choice = input("\nSelect model (1 or 2, default=1): ").strip() or "1"
    model_key = list(MODELS.keys())[int(choice) - 1]
    model = MODELS[model_key]

    project_root = Path(__file__).parent.parent
    output_dir = project_root / "data" / "models"
    output_path = output_dir / model["filename"]

    if output_path.exists():
        overwrite = input(f"\n{output_path.name} already exists. Overwrite? (y/N): ").strip().lower()
        if overwrite != 'y':
            print("Download cancelled.")
            return

    try:
        download_file(model["url"], output_path)
        print(f"\nModel ready at: {output_path}")
        print("\nYou can now run the backend server:")
        print(f"  export SELFRAG_MODEL_PATH={output_path}")
        print("  python backend/api.py")
    except Exception as e:
        print(f"\nDownload failed: {e}")
        print("\nAlternative: Download manually from HuggingFace")
        print(f"  URL: {model['url']}")
        print(f"  Save to: {output_path}")
        sys.exit(1)

if __name__ == "__main__":
    main()
