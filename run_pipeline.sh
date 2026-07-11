#!/bin/bash
set -euo pipefail

export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1
# Enable logging output from Python modules
export PYTHONPATH="."

echo "1. Downloading datasets..."
./venv/bin/python setup_data.py

echo "2. Normalising and preparing datasets..."
./venv/bin/python -m server.scraper.prepare_datasets

echo "3. Removing legacy vector store..."
rm -rf server/vectorstore/chroma_db

echo "4. Building vector store (local embeddings — no API key needed)..."
./venv/bin/python -m server.vectorstore.build_vectorstore

echo "Done! Pipeline completed successfully."
