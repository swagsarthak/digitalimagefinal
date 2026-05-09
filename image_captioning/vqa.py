from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor


TEXT_QUESTION_TERMS = {
    "display",
    "displayed",
    "letter",
    "letters",
    "label",
    "license",
    "number",
    "numbers",
    "plate",
    "print",
    "printed",
    "read",
    "readable",
    "reads",
    "say",
    "says",
    "sign",
    "text",
    "word",
    "words",
    "write",
    "written",
    "writing",
}

UNCLEAR_ANSWERS = {
    "",
    "i do not know",
    "i don't know",
    "not sure",
    "unknown",
    "unsure",
}


@dataclass(frozen=True)
class VQAConfig:
    """Runtime settings for BLIP visual question answering."""

    model_name: str = "Salesforce/blip-vqa-base"
    device: str | None = None
    max_new_tokens: int = 30
    fallback_answer: str = "I could not determine that clearly from the image."


class ImageQuestionAnswerer:
    """Small wrapper around a BLIP visual question answering model."""

    def __init__(self, config: VQAConfig | None = None) -> None:
        self.config = config or VQAConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = BlipProcessor.from_pretrained(self.config.model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def load_image(image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        return Image.open(image).convert("RGB")

    @staticmethod
    def is_text_question(question: str) -> bool:
        cleaned = question.lower().replace("?", " ")
        words = {word.strip(".,:;!()[]{}\"'") for word in cleaned.split()}
        return bool(words.intersection(TEXT_QUESTION_TERMS))

    @torch.inference_mode()
    def answer(self, image: str | Path | Image.Image, question: str) -> str:
        """Answer one natural-language question about an image."""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        pil_image = self.load_image(image)
        inputs = self.processor(pil_image, cleaned_question, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
        )
        answer = self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
        return self._fallback_if_unclear(answer)

    def answer_with_context(
        self,
        image: str | Path | Image.Image,
        question: str,
        extracted_text: str | None = None,
    ) -> str:
        """Answer with optional OCR text used for text-reading questions."""

        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("Question cannot be empty.")

        cleaned_text = (extracted_text or "").strip()
        if cleaned_text and self.is_text_question(cleaned_question):
            return cleaned_text

        return self.answer(image, cleaned_question)

    def answer_many(self, items: Iterable[tuple[str | Path | Image.Image, str]]) -> list[dict[str, str]]:
        """Answer multiple image-question pairs while reusing the loaded model."""

        rows: list[dict[str, str]] = []
        for image, question in items:
            rows.append(
                {
                    "image_path": str(image),
                    "question": question,
                    "answer": self.answer(image, question),
                }
            )
        return rows

    def _fallback_if_unclear(self, answer: str) -> str:
        cleaned = answer.strip()
        if cleaned.lower() in UNCLEAR_ANSWERS:
            return self.config.fallback_answer
        return cleaned
