from __future__ import annotations

import chromadb

from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import config
from src.the_batch.templates import Article
from src.embeddings.text_embedder import TextEmbedder
from src.embeddings.image_embedder import ImageEmbedder


class TheBatchChromaIndexer:
    def __init__(self, text_embedder: TextEmbedder, clip_embedder: ImageEmbedder, chunk_size : int = 1024, overlap: int = 256):
        self.client = chromadb.PersistentClient(path=config.chroma_path)
        self.text_embedder = text_embedder
        self.clip_embedder = clip_embedder

        self.articles_text = self.client.get_or_create_collection(
            name="the_batch_articles_text",
            metadata={"hnsw:space": "cosine"},
        )
        self.article_images = self.client.get_or_create_collection(
            name="the_batch_article_images",
            metadata={"hnsw:space": "cosine"},
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

    def search_text(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Similarity search over text chunks.
        """
        if not query.strip():
            return []

        query_emb = self.text_embedder.embed_documents_to_list([query])

        results = self.articles_text.query(
            query_embeddings=query_emb,
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for idd, doc, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
            hits.append(
                {
                    "source": "text",
                    "id": idd,
                    "distance": dist,
                    "document": doc,
                    "metadata": meta,
                }
            )
        return hits
    
    def search_images(self, image_query: str, k: int = 4) -> List[Dict[str, Any]]:
        """
        Similarity search over image embeddings
        """
        if not image_query or not image_query.strip():
            return []

        query_emb = self.clip_embedder.embed_texts_to_list([image_query])

        results = self.article_images.query(
            query_embeddings=query_emb,
            n_results=k,
            include=["metadatas", "distances"],
        )

        hits: List[Dict[str, Any]] = []
        ids_list = results.get("ids", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for idd, meta, dist in zip(ids_list, metas_list, dists_list):
            hits.append(
                {
                    "source": "image",
                    "id": idd,
                    "distance": dist,
                    "document": None,
                    "metadata": meta,
                }
            )
        return hits
    
    def search_multimodal(self, text_query: str, image_query = None, k_text: int = 4, k_image: int = 4) -> List[Dict[str, Any]]:
        """
        Combined search:
          - text_query over text chunks
          - image_query (short image caption) over image embeddings

        Returns aggregated results deduped by article_id.
        """
        text_hits = self.search_text(text_query, k=k_text) if text_query else []
        image_hits = self.search_images(image_query, k=k_image) if image_query else []
        
        combined: Dict[str, Dict[str, Any]] = {}

        def add_hit(hit: Dict[str, Any]) -> None:
            meta = hit["metadata"] or {}
            article_id = meta.get("article_id")
            if not article_id:
                return

            entry = combined.get(article_id)
            if entry is None:
                entry = {
                    "article_id": article_id,
                    "title": meta.get("title"),
                    "url": meta.get("url"),
                    "primary_topic": meta.get("primary_topic"),
                    "published_at": meta.get("published_at"),
                    "text_snippets": [],
                    "image_urls": [],
                    "image_alts": [],
                    "min_distance": hit["distance"],
                    "sources": set(),
                }
                combined[article_id] = entry

            if hit["distance"] < entry["min_distance"]:
                entry["min_distance"] = hit["distance"]

            entry["sources"].add(hit["source"])

            if hit["source"] == "text" and hit["document"]:
                if hit["document"] not in entry["text_snippets"]:
                    entry["text_snippets"].append(hit["document"])

            if hit["source"] == "image":
                img_url = meta.get("image_url")
                img_alt = meta.get("image_alt")
            else:
                img_url = meta.get("hero_image_url")
                img_alt = meta.get("hero_image_alt")

            if img_url and img_url not in entry["image_urls"]:
                entry["image_urls"].append(img_url)
                entry["image_alts"].append(img_alt)

        for h in text_hits:
            add_hit(h)
        for h in image_hits:
            add_hit(h)

        results = list(combined.values())

        results.sort(key=lambda r: r["min_distance"])

        for r in results:
            r["sources"] = list(r["sources"])

        return results