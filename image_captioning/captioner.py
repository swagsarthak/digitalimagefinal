from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


@dataclass(frozen=True)
class CaptionConfig:
    """Runtime settings for BLIP caption generation."""

    model_name: str = "Salesforce/blip-image-captioning-base"
    device: str | None = None
    max_new_tokens: int = 50
    num_beams: int = 5


class ImageCaptioner:
    """Small wrapper around a BLIP image captioning model."""

    def __init__(self, config: CaptionConfig | None = None) -> None:
        self.config = config or CaptionConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(self.config.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def load_image(image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    @torch.inference_mode()
    def caption(self, image: str | Path | Image.Image, prompt: str | None = None) -> str:
        """Generate one caption for an image."""

        pil_image = self.load_image(image)
        if prompt:
            inputs = self.processor(pil_image, prompt, return_tensors="pt")
        else:
            inputs = self.processor(pil_image, return_tensors="pt")

        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            num_beams=self.config.num_beams,
        )
        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

    def product_description(self, image: str | Path | Image.Image) -> str:
        """Generate an e-commerce style product description for an image."""

        return self.caption(image, prompt="a detailed product description of")

    def caption_many(self, images: Iterable[str | Path], prompt: str | None = None) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for image_path in images:
            rows.append({"image_path": str(image_path), "caption": self.caption(image_path, prompt=prompt)})
        return rows
