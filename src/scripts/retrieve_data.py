from pathlib import Path
from src.the_batch.ingestor import TheBatchIngestor

if __name__ == "__main__":
    ingestor = TheBatchIngestor()
    output_path = Path("data/processed/the_batch_articles.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    topics = ["letters", "data-points", "research", "business", "science", "culture", "hardware"]
    ingestor.ingest_all_topics(topics=topics, output_jsonl=output_path)