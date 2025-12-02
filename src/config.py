from __future__ import annotations

import os
import torch
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv(".env", override=False)
load_dotenv(".env.example", override=False)

def determine_device():
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device

@dataclass
class AppConfig:
    text_embed_model_name: str = field(
        default_factory=lambda: os.getenv(
            "TEXT_EMBED_MODEL",
            "intfloat/multilingual-e5-base",
        )
    )

    clip_model_name: str = field(
        default_factory=lambda: os.getenv(
            "CLIP_MODEL",
            "openai/clip-vit-base-patch32",
        )
    )

    chroma_path: str = field(
        default_factory=lambda: os.getenv(
            "CHROMA_PATH",
            "data/chroma",
        )
    )
    device: str = determine_device()
    
    google_ai_api_key: str = field(
        default_factory=lambda: os.getenv(
            "GOOGLE_AI_API_KEY",
            None,
        )
    )
    gemini_model_name: str = field(
        default_factory=lambda: os.getenv(
            "GEMINI_MODEL",
            "gemini-2.5-flash",
        )
    )

config = AppConfig()
