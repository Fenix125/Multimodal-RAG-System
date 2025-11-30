import json
from pathlib import Path
from typing import List

from src.config import config
from src.the_batch.templates import Article
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from src.vector_db.chroma_store import TheBatchChromaIndexer


def load_articles_from_jsonl(path: Path) -> List[Article]:
    """
    Load The Batch articles from a JSONL file produced by TheBatchIngestor.
    Each line is a JSON object corresponding to the Article schema.
    """
    articles: List[Article] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            articles.append(Article(**data))

    print(f"[INFO] Loaded {len(articles)} articles from {path}")
    return articles

def main():
    jsonl_path = Path("data/processed/the_batch_articles.jsonl")
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Input JSONL file not found at {jsonl_path}. "
            "Run the ingestion step first."
        )

    articles = load_articles_from_jsonl(jsonl_path)
    if not articles:
        print("[WARN] No articles loaded; nothing to index.")
        return

    print(f"[INFO] Using text embedding model: {config.text_embed_model_name}")
    text_embedder = TextEmbedder(
        model_name=config.text_embed_model_name,
        device=config.device,
    )

    print(f"[INFO] Using image embedding model: {config.clip_model_name}")
    clip_embedder = ImageEmbedder(
        model_name=config.clip_model_name,
        device=config.device,
    )
    indexer = TheBatchChromaIndexer(
        text_embedder=text_embedder,
        clip_embedder=clip_embedder,
    )

    print("[STEP] Starting Chroma indexing...")
    indexer.index(articles)
    print("[DONE] Database population complete.")

if __name__ == "__main__":
    main()
