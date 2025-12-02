import torch
import torch.nn.functional as F

from typing import List, Union
from io import BytesIO
from urllib.parse import urlparse

import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel

ImageInput = Union[Image.Image, str]

class ImageEmbedder:
    """
    Wrapper for CLIP HF models that provide:
      - model.get_image_features(...)
      - model.get_text_features(...)

    Works with:
      - "openai/clip-vit-base-patch32"
      - "openai/clip-vit-base-patch16"
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device

        self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=False)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def ensure_image_list(self, images: Union[ImageInput, List[ImageInput]]) -> List[ImageInput]:
        if isinstance(images, list):
            return images
        return [images]

    def ensure_text_list(self, texts: Union[str, List[str]]) -> List[str]:
        if isinstance(texts, str):
            return [texts]
        return list(texts)

    def is_url(self, path: str) -> bool:
        try:
            parsed = urlparse(path)
            return parsed.scheme in ("http", "https")
        except Exception:
            return False
        
    def load_image_from_url(self, url: str) -> Image.Image:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content)).convert("RGB")
    
    def to_pil(self, image: ImageInput) -> Image.Image:
        """
        Accepts:
          - PIL.Image.Image -> returns as is
          - str (URL) -> fetch via HTTP and open
          - str (file path) -> open local file
        """
        if isinstance(image, Image.Image):
            return image

        if isinstance(image, str) and self.is_url(image):
            return self.load_image_from_url(image)

        return Image.open(image).convert("RGB")
    

    @torch.no_grad()
    def embed_images(self, images: Union[ImageInput, List[ImageInput]]) -> torch.Tensor:
        """
        Returns a (B, D) tensor of L2-normalized image embeddings.
        """
        img_list = self.ensure_image_list(images)
        pil_images = [self.to_pil(im) for im in img_list]

        inputs = self.processor(images=pil_images, return_tensors="pt").to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features

    @torch.no_grad()
    def embed_texts(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Returns a (B, D) tensor of L2-normalized text embeddings (in same space as images).
        """
        text_list = self.ensure_text_list(texts)

        inputs = self.processor(text=text_list, padding=True, return_tensors="pt").to(self.device)
        text_features = self.model.get_text_features(**inputs)
        text_features = F.normalize(text_features, p=2, dim=-1)
        return text_features

    def embed_images_to_list(self, images: Union[ImageInput, List[ImageInput]]) -> List[List[float]]:
        return self.embed_images(images).cpu().tolist()

    def embed_texts_to_list(self, texts: Union[str, List[str]]) -> List[List[float]]:
        return self.embed_texts(texts).cpu().tolist()
