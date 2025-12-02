from __future__ import annotations

from typing import List, Optional
from langchain_core.tools import tool

def make_multimodal_search_tool(indexer):
    """
    Makes multimodal search tool using provided indexer
    """
    @tool
    def the_batch_multimodal_search(text_query: str, image_query: Optional[str] = None, k_text: int = 4, k_image: int = 4) -> str:
        """
        Search The Batch vector index.

        Args:
            text_query: natural language question
            image_query: OPTIONAL short text description for images (e.g. "robot", "conference audience", "Andrew Ng").
            k_text: how many text chunks to retrieve.
            k_image: how many images to retrieve.

        Returns:
            A human-readable summary of the most relevant articles,
            including snippets and image URLs.
        """
        if not text_query and not image_query:
            return "No text query provided."

        results = indexer.search_multimodal(
            text_query=text_query,
            image_query=image_query,
            k_text=k_text,
            k_image=k_image,
        )

        if not results:
            return "No relevant articles found in The Batch news."

        lines: List[str] = []
        for idx, r in enumerate(results, start=1):
            title = r.get("title") or "(untitled article)"
            url = r.get("url") or "N/A"
            published = r.get("published_at") or "unknown date"
            sources = ", ".join(sorted(r.get("sources", []))) or "text"

            header = f"{idx}. {title}\n Published: {published}\n Topic: {r.get("primary_topic")}\n URL: {url}\n Sources: {sources}"

            snippet_lines: List[str] = []
            text_snippets = r.get("text_snippets") or []
            if text_snippets:
                for s_idx, s in enumerate(text_snippets[:3], start=1):
                    snippet_lines.append(f"   [text snippet {s_idx}] {s}")

            image_urls = r.get("image_urls") or []
            image_alts = r.get("image_alts") or []
            if image_urls:
                snippet_lines.append(" Images:")
                for u, alt in zip(image_urls, image_alts):
                    if alt:
                        snippet_lines.append(f"     - {u}  (alt: {alt})")
                    else:
                        snippet_lines.append(f"     - {u}")

            if snippet_lines:
                lines.append(header + "\n" + "\n".join(snippet_lines))
            else:
                lines.append(header)

        return "\n\n".join(lines)
    return the_batch_multimodal_search