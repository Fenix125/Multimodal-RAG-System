from __future__ import annotations

from datetime import datetime
from typing import List, Optional, Dict

from pydantic import BaseModel

BASE_ARTICLE_URL = "https://www.deeplearning.ai/the-batch"

TOPICS: Dict[str, Dict[str, str]] = {
    "letters": {"slug": "letters", "label": "Andrew's Letters"},
    "data-points": {"slug": "data-points", "label": "Data Points"},
    "research": {"slug": "research", "label": "ML Research"},
    "business": {"slug": "business", "label": "Business"},
    "science": {"slug": "science", "label": "Science"},
    "culture": {"slug": "culture", "label": "Culture"},
    "hardware": {"slug": "hardware", "label": "Hardware"},
}

class ImageMeta(BaseModel):
    image_id: str
    url: str
    local_path: Optional[str] = None
    alt: Optional[str] = None

class Article(BaseModel):
    article_id: str
    title: str
    url: str
    primary_topic: str
    topic_label: Optional[str] = None
    tags: List[str] = []
    published_at: Optional[datetime] = None
    body_text: str
    images: List[ImageMeta] = []
