from __future__ import annotations

import json
from typing import List, Optional
from langchain_core.tools import tool

def make_multimodal_search_tool(indexer):
    """
    Makes multimodal search tool using provided indexer
    """
    @tool
    def the_batch_multimodal_search(text_query: str, image_query: Optional[str] = None) -> str:
        """
        Search The Batch vector index and return structured JSON.

        Args:
            text_query: natural language question.
            image_query: short text description for images
                (e.g. "robot", "conference audience", "Andrew Ng").

        Returns:
            JSON string with the query context and a 'results' list of articles
            and images that match, JSON structured like:
            {
                "query": {"text": "...", "image": "..."},
                "results": [
                    {
                        "article_id": "...",
                        "title": "...",
                        "url": "...",
                        "topic": "...",
                        "published_at": "...",
                        "sources": ["text", "image"],
                        "text_snippets": ["..."],
                        "image_urls": ["..."],
                        "image_alts": ["..."],
                        "score": 0.0
                    }
                ]
            }
        """
        query_payload = {"text": text_query, "image": image_query}

        if not text_query and not image_query:
            return json.dumps(
                {"query": query_payload, "results": [], "message": "No query provided."}
            )

        results = indexer.search_multimodal(
            text_query=text_query,
            image_query=image_query
        )

        if not results:
            return json.dumps(
                {
                    "query": query_payload,
                    "results": [],
                    "message": "No relevant articles found in The Batch news.",
                }
            )

        articles: List[dict] = []
        for r in results:
            min_distance = r.get("min_distance")
            score = None
            if min_distance is not None:
                score = 1.0 / (1.0 + float(min_distance))
      
            image_urls = r.get("image_urls") or []
            image_alts = r.get("image_alts") or []
            if image_alts and len(image_alts) < len(image_urls):
                image_alts = image_alts + [None] * (len(image_urls) - len(image_alts))
            elif not image_alts:
                image_alts = [None] * len(image_urls)

            sources = sorted(r.get("sources") or [])
            text_snippets = (r.get("text_snippets") or [])[:3]

            articles.append(
                {
                    "article_id": r.get("article_id"),
                    "title": r.get("title"),
                    "url": r.get("url"),
                    "topic": r.get("primary_topic"),
                    "published_at": r.get("published_at"),
                    "sources": sources,
                    "text_snippets": text_snippets,
                    "image_urls": image_urls,
                    "image_alts": image_alts,
                    "score": score,
                }
            )

        payload = {
            "query": query_payload,
            "results": articles,
        }

        return json.dumps(payload)
    return the_batch_multimodal_search

def make_image_search_tool(indexer):
    @tool
    def image_search(image_path: str) -> str:
        """
        Search image index using a local image file path.
        Args:
            image_path: path to a local image file.
        Returns:
            JSON string with 'query' info and 'results' list.
        """
        if not image_path:
            return json.dumps({"query": {"image_path": image_path}, "results": [], "message": "No image provided."})

        hits = indexer.search_images_by_image(image_path=image_path)
        if not hits:
            return json.dumps({"query": {"image_path": image_path}, "results": [], "message": "No matching images found."})

        combined = []
        for h in hits:
            meta = h.get("metadata") or {}
            min_dist = h.get("distance")
            score = None
            if min_dist is not None:
                score = 1.0 / (1.0 + float(min_dist))
      
            img_url = meta.get("image_url")
            img_alt = meta.get("image_alt")
            combined.append(
                {
                    "article_id": meta.get("article_id"),
                    "title": meta.get("title") or "(untitled article)",
                    "url": meta.get("url"),
                    "topic": meta.get("primary_topic"),
                    "published_at": meta.get("published_at"),
                    "sources": ["image"],
                    "text_snippets": [],
                    "image_urls": [img_url] if img_url else [],
                    "image_alts": [img_alt] if img_alt else [],
                    "score": score,
                }
            )

        return json.dumps({"query": {"image_path": image_path}, "results": combined})
    return image_search