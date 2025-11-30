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
        print("CUDA is available! Using GPU.")
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("MPS is available. Using MPS.")
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

config = AppConfig()

