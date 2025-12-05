from __future__ import annotations

import json
import time
import re
import requests

from typing import List, Optional, Dict, Iterable

from bs4 import BeautifulSoup

from datetime import datetime
from pathlib import Path

from src.the_batch.templates import Article, ImageMeta, TOPICS, BASE_ARTICLE_URL


def clean_html_fragment_to_text(html_fragment: str) -> str:
    """
    Cleans HTML into a plain text.
    """
    if not html_fragment:
        return ""

    soup = BeautifulSoup(html_fragment, "html.parser")

    for tag in soup(["script", "style"]):
        tag.decompose()

    for katex in soup.select("span.katex"):
        katex.replace_with(katex.get_text(separator="", strip=True))

    text = soup.get_text(separator="\n")
    
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    text = text.strip()

    return text

class TheBatchIngestor:
    """
    Ingests articles from https://www.deeplearning.ai/the-batch/ for specific topics.

    For each topic:
      - Fetch https://www.deeplearning.ai/the-batch/tag/{slug}/
      - Parse __NEXT_DATA__ to get 'posts' list (cards on the page)
      - For each post, fetch the article page /the-batch/{article_name}/
        and parse 'post.html' from __NEXT_DATA__ for full text.
    """

    def __init__(self, session: Optional[requests.Session] = None, delay_seconds: float = 1.0):
        self.session = session or requests.Session()
        self.delay_seconds = delay_seconds

        self.session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (compatible; TheBatchRAGBot/0.1 (+https://github.com/Fenix125/Multimodal-RAG-System)"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            }
        )

    def clean_body_text(self, body_html: str) -> str:
        """
        Convert article HTML into clean plain text:
        - strip scripts/styles (handled by clean_html_fragment_to_text)
        - remove ElevenLabs audio intro if present
        - flatten single newlines inside paragraphs to spaces
          while preserving paragraph breaks (double newlines).
        """
        audio_intro = re.compile(
            r"^Loading the\s+Elevenlabs Text to Speech\s+AudioNative Player\.\.\.\s*",
            flags=re.IGNORECASE | re.MULTILINE,
        )

        text = clean_html_fragment_to_text(body_html)

        text = re.sub(audio_intro, "", text, count=1).lstrip()

        paragraphs = text.split("\n\n")

        cleaned_paragraphs = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            p = re.sub(r"\s*\n\s*", " ", p)
            cleaned_paragraphs.append(p)

        return "\n\n".join(cleaned_paragraphs)
    
    def ingest_all_topics(self, topics: Optional[Iterable[str]] = None, output_jsonl: Optional[Path] = None) -> List[Article]:
        """
        Ingest articles for the selected topics from deeplearning.ai/the-batch.

        topics: list of topic keys (If None, uses all keys defined in TOPICS)
        """
        if topics is None:
            topics = TOPICS.keys()

        articles_by_slug: Dict[str, Article] = {}

        for topic_key in topics:
            if topic_key not in TOPICS:
                print(f"[WARN] Unknown topic key '{topic_key}', skipping.")
                continue

            topic_info = TOPICS[topic_key]
            tag_slug = topic_info["slug"]
            topic_label = topic_info["label"]

            print(f"[INFO] Ingesting topic '{topic_label}', tag='{tag_slug}')")

            posts = self.fetch_posts_for_tag(tag_slug)
            print(f"[INFO]   Tag '{tag_slug}': found {len(posts)} posts on article page.")

            for post_meta in posts:
                slug = post_meta.get("slug")
                if not slug:
                    continue

                if slug in articles_by_slug:
                    existing = articles_by_slug[slug]
                    if topic_key not in existing.tags:
                        existing.tags.append(topic_key)
                    continue

                try:
                    article = self.build_article_from_post_meta(
                        topic_key=topic_key,
                        topic_label=topic_label,
                        post_meta=post_meta,
                    )
                    articles_by_slug[slug] = article
                except Exception as e:
                    print(f"[ERROR] Failed to build article for slug '{slug}': {e}")

        articles = list(articles_by_slug.values())
        articles.sort(key=lambda a: (a.published_at or datetime.min), reverse=True)

        if output_jsonl is not None:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with output_jsonl.open("w", encoding="utf-8") as f:
                for art in articles:
                    f.write(art.model_dump_json() + "\n")
            print(f"[INFO] Saved {len(articles)} articles to {output_jsonl}")

        print(f"[DONE] Ingested {len(articles)} unique articles in total.")
        return articles

    def tag_page_url(self, tag_slug: str) -> str:
        """
        For each topic, we use the front-end tag page:

        https://www.deeplearning.ai/the-batch/tag/{slug}/
        """
        return f"{BASE_ARTICLE_URL}/tag/{tag_slug}/"

    def fetch_posts_for_tag(self, tag_slug: str) -> List[dict]:
        """
        Fetch posts list for a given tag from /the-batch/tag/{slug}/.

        Already contains a list of article cards for the topic in __NEXT_DATA__
        """
        url = self.tag_page_url(tag_slug)
        html = self.get(url)
        if not html:
            print(f"[WARN] Empty or failed tag page for tag='{tag_slug}'.")
            return []

        next_data = self.extract_next_data(html)
        posts = self.extract_posts_from_tag_next_data(next_data)
        return posts

    def build_article_from_post_meta(self, topic_key: str, topic_label: str, post_meta: dict) -> Article:
        slug = post_meta["slug"]
        article_url = f"{BASE_ARTICLE_URL}/{slug}/"

        print(f"[INFO]   Fetching article '{slug}' -> {article_url}")
        html = self.get(article_url)
        if not html:
            raise RuntimeError(f"Empty HTML for article '{slug}'")

        next_data = self.extract_next_data(html)
        post_obj = self.extract_post_from_article_next_data(next_data)

        full_title = (post_meta.get("title") or "").strip()

        custom_excerpt = (post_meta.get("custom_excerpt") or "").strip()
        excerpt = (post_meta.get("excerpt") or "").strip()

        body_html = post_obj.get("html") or ""
        body_text = self.clean_body_text(body_html)

        published_at = None
        published_raw = post_obj.get("published_at")
        if published_raw:
            try:
                published_at = datetime.fromisoformat(
                    published_raw.replace("Z", "+00:00")
                )
            except ValueError:
                pass

        tags = [t.get("name") for t in (post_meta.get("tags") or []) if t.get("name")]

        feature_image = post_meta.get("feature_image")
        feature_image_alt = post_meta.get("feature_image_alt") or None

        images: List[ImageMeta] = []
        if feature_image:
            image_id = f"{slug}_hero"
            images.append(
                ImageMeta(
                    image_id=image_id,
                    url=feature_image,
                    local_path=None,
                    alt=feature_image_alt
                )
            )

        article = Article(
            article_id=slug,
            title=full_title,
            url=article_url,
            primary_topic=topic_key,
            topic_label=topic_label,
            tags=tags,
            published_at=published_at,
            body_text=body_text,
            images=images,
        )

        return article

    def get(self, url: str) -> Optional[str]:
        try:
            resp = self.session.get(url, timeout=20)
            if resp.status_code != 200:
                print(f"[WARN] GET {url} -> {resp.status_code}")
                return None
            time.sleep(self.delay_seconds)
            return resp.text
        except requests.RequestException as e:
            print(f"[ERROR] Request failed for {url}: {e}")
            return None

    def extract_next_data(self, html: str) -> dict:
        """
        Extract Next.js __NEXT_DATA__ JSON from an HTML page.
        """
        soup = BeautifulSoup(html, "html.parser")
        script = soup.find("script", id="__NEXT_DATA__", type="application/json")
        if not script or not script.string:
            raise RuntimeError("Could not find __NEXT_DATA__ script tag")
        try:
            data = json.loads(script.string)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse __NEXT_DATA__ JSON: {e}") from e
        return data

    def extract_posts_from_tag_next_data(self, next_data: dict) -> List[dict]:
        """
        On tag pages, posts are at next_data['props']['pageProps']['posts'].
        """
        props = next_data.get("props", {})
        page_props = props.get("pageProps", {})
        posts = page_props.get("posts") or []
        if not isinstance(posts, list):
            return []
        return posts

    def extract_post_from_article_next_data(self, next_data: dict) -> dict:
        """
        On article pages, the full article is in next_data['props']['pageProps']['post'].
        """
        props = next_data.get("props", {})
        page_props = props.get("pageProps", {})
        post = page_props.get("post")
        if not isinstance(post, dict):
            raise RuntimeError("No 'post' object found in article __NEXT_DATA__")
        return post