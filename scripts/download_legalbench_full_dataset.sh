#!/bin/bash
# Download and Setup LegalBench-RAG Full Dataset (6,858 queries)

echo "=========================================="
echo "LegalBench-RAG Dataset Download Script"
echo "=========================================="
echo ""
echo "This script will help you download the full LegalBench-RAG dataset."
echo "Dataset size: ~6,858 queries over 79 million characters"
echo ""

mkdir -p legalbench_full

echo "Step 1: Download Dataset from Dropbox"
echo "--------------------------------------"
echo ""
echo "Please visit this link to download the dataset:"
echo "https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=0"
echo ""
echo "After downloading:"
echo "1. Extract the downloaded files"
echo "2. You should see two folders: 'corpus' and 'benchmarks'"
echo "3. Move both folders into: $(pwd)/legalbench_full/"
echo ""
echo "Expected structure:"
echo "  legalbench_full/"
echo "  ├── corpus/       (contains .txt files with legal documents)"
echo "  └── benchmarks/   (contains .json files with queries)"
echo ""
read -p "Press Enter when you have completed the download and extraction..."

echo ""
echo "Step 2: Verifying Dataset Structure"
echo "------------------------------------"

if [ -d "legalbench_full/corpus" ] && [ -d "legalbench_full/benchmarks" ]; then
    echo "Found corpus directory"
    echo "Found benchmarks directory"

    corpus_files=$(find legalbench_full/corpus -type f -name "*.txt" | wc -l)
    benchmark_files=$(find legalbench_full/benchmarks -type f -name "*.json" | wc -l)

    echo ""
    echo "Dataset Statistics:"
    echo "  Corpus files: $corpus_files"
    echo "  Benchmark files: $benchmark_files"

    if [ $benchmark_files -gt 0 ]; then
        echo ""
        echo "Dataset is ready"
        echo ""
        echo "You can now run the notebook with:"
        echo "  jupyter notebook performance_evaluation.ipynb"
    else
        echo ""
        echo "Error: No benchmark files found"
        echo "Please check that you extracted the files correctly."
        exit 1
    fi
else
    echo "Error: Dataset directories not found"
    echo ""
    echo "Please ensure you have:"
    echo "  1. Downloaded the dataset from the Dropbox link"
    echo "  2. Extracted the files"
    echo "  3. Moved the 'corpus' and 'benchmarks' folders to $(pwd)/legalbench_full/"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup Complete"
echo "=========================================="
