#!/bin/sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-/app/data}"
CHROMA_PATH="${CHROMA_PATH:-$DATA_DIR/chroma}"
JSONL_PATH="${JSONL_PATH:-$DATA_DIR/processed/the_batch_articles.jsonl}"

mkdir -p "$DATA_DIR"

#if database already exists and is non-empty, skip re-ingestion for faster restarts
if [ -d "$CHROMA_PATH" ] && [ -n "$(ls -A "$CHROMA_PATH" 2>/dev/null)" ]; then
  echo "[ENTRYPOINT] Existing Chroma DB found at $CHROMA_PATH; skipping retrieval/ingestion."
else
  echo "[ENTRYPOINT] Chroma DB missing/empty. Running retrieval + ingestion..."
  python -m src.scripts.retrieve_data
  python -m src.scripts.ingest_chroma_db
fi

if [ "$#" -eq 0 ]; then
  set -- streamlit run app.py --server.port="${STREAMLIT_PORT:-8501}" --server.address="${STREAMLIT_ADDRESS:-0.0.0.0}"
fi

echo "[ENTRYPOINT] exec: $*"
exec "$@"
