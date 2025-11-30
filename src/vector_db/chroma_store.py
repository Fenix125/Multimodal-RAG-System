from __future__ import annotations

from typing import List, Dict, Any

import chromadb

from src.config import config
from src.the_batch.templates import Article
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TheBatchChromaIndexer:
    def __init__(self, text_embedder: TextEmbedder, clip_embedder: ImageEmbedder, chunk_size : int = 1024, overlap: int = 512):
        self.client = chromadb.PersistentClient(path=config.chroma_path)
        self.text_embedder = text_embedder
        self.clip_embedder = clip_embedder

        self.articles_text = self.client.get_or_create_collection(
            name="the_batch_articles_text"
        )
        self.article_images = self.client.get_or_create_collection(
            name="the_batch_article_images"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " "],
        )

    def base_article_metadata(self, art: Article):
        """
        Common metadata used for both text chunks and images.
        """
        hero_image_url = art.images[0].url if art.images and art.images[0].url else None

        return {
            "article_id": art.article_id,
            "title": art.title,
            "url": art.url,
            "primary_topic": art.primary_topic,
            "published_at": art.published_at.isoformat() if art.published_at else None,
            "hero_image_url": hero_image_url,
        }
    
    def index(self, articles: List[Article]) -> None:
        """
        Index both article text and article images into their respective Chroma collections.
        """

        text_ids: List[str] = []
        text_docs: List[str] = []
        text_metadatas: List[Dict[str, Any]] = []

        image_ids: List[str] = []
        image_urls: List[str] = []
        image_metadatas: List[Dict[str, Any]] = []

        for art in articles:
            base_meta = self.base_article_metadata(art)

            body = (art.body_text or "").strip()
            if body:
                chunks = self.text_splitter.split_text(body)

                for idx, chunk in enumerate(chunks):
                    chunk_id = f"{art.article_id}:chunk_{idx}"

                    text_ids.append(chunk_id)
                    text_docs.append(chunk)

                    meta = dict(base_meta)
                    meta.update(
                        {
                            "chunk_index": idx,
                            "num_chunks": len(chunks),
                        }
                    )
                    text_metadatas.append(meta)

            for img in art.images:
                if not img.url:
                    continue

                image_id = img.image_id or f"{art.article_id}::image::{len(image_ids)}"

                image_ids.append(image_id)
                image_urls.append(img.url)

                meta = dict(base_meta)
                meta.update(
                        {
                            "image_id": image_id,
                            "image_url": img.url,
                            "image_alt": img.alt,
                        }
                    )
                image_metadatas.append(meta)

        if text_ids:
            print(f"[INFO] Embedding {len(text_docs)} text chunks...")
            text_embeddings = self.text_embedder.embed_documents_to_list(text_docs)

            print("[INFO] Adding text chunks to Chroma collection 'the_batch_articles_text'...")
            self.articles_text.add(
                ids=text_ids,
                documents=text_docs,
                metadatas=text_metadatas,
                embeddings=text_embeddings,
            )
        else:
            print("[INFO] No text chunks to index.")

        if image_ids:
            print(f"[INFO] Embedding {len(image_urls)} images...")
            image_embeddings = self.clip_embedder.embed_images_to_list(image_urls)

            print("[INFO] Adding images to Chroma collection 'the_batch_article_images'...")
            self.article_images.add(
                ids=image_ids,
                metadatas=image_metadatas,
                embeddings=image_embeddings,
            )
        else:
            print("[INFO] No images to index.")

        print("[DONE] Chroma indexing complete.")