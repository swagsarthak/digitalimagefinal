from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    DetrForObjectDetection,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from .captioner import ImageCaptioner


@dataclass(frozen=True)
class ObjectDetectionConfig:
    """Runtime settings for DETR object detection."""

    model_name: str = "facebook/detr-resnet-50"
    device: str | None = None
    threshold: float = 0.7
    max_objects: int = 12


@dataclass(frozen=True)
class OCRConfig:
    """Runtime settings for OCR text extraction."""

    model_name: str = "microsoft/trocr-base-printed"
    device: str | None = None
    max_new_tokens: int = 64


@dataclass(frozen=True)
class DetectedObject:
    label: str
    score: float
    box: list[float]


@dataclass(frozen=True)
class ImageAnalysis:
    caption: str
    alt_text: str
    objects: list[DetectedObject]
    extracted_text: str
    tags: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ImageObjectDetector:
    """Object detector backed by DETR."""

    def __init__(self, config: ObjectDetectionConfig | None = None) -> None:
        self.config = config or ObjectDetectionConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(self.config.model_name)
        self.model = DetrForObjectDetection.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def detect(self, image: Image.Image) -> list[DetectedObject]:
        pil_image = image.convert("RGB")
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([pil_image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(
            outputs,
            target_sizes=target_sizes,
            threshold=self.config.threshold,
        )[0]

        detected: list[DetectedObject] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"], strict=False):
            detected.append(
                DetectedObject(
                    label=self.model.config.id2label[int(label.item())],
                    score=round(float(score.item()), 3),
                    box=[round(float(value), 2) for value in box.tolist()],
                )
            )

        detected.sort(key=lambda item: item.score, reverse=True)
        return detected[: self.config.max_objects]


class ImageOCRReader:
    """OCR reader backed by TrOCR for printed text in images."""

    def __init__(self, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(self.config.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.config.model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def extract_text(self, image: Image.Image) -> str:
        pil_image = image.convert("RGB")
        pixel_values = self.processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
        output_ids = self.model.generate(pixel_values, max_new_tokens=self.config.max_new_tokens)
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()


class VisualIntelligenceAnalyzer:
    """Combines captioning, object detection, OCR, tags, and alt text."""

    def __init__(
        self,
        captioner: ImageCaptioner,
        detector: ImageObjectDetector,
        ocr_reader: ImageOCRReader,
    ) -> None:
        self.captioner = captioner
        self.detector = detector
        self.ocr_reader = ocr_reader

    def analyze(self, image: str | Path | Image.Image) -> ImageAnalysis:
        pil_image = ImageCaptioner.load_image(image)
        caption = self.captioner.caption(pil_image)
        objects = self.detector.detect(pil_image)
        extracted_text = self.ocr_reader.extract_text(pil_image)
        tags = generate_tags(caption, objects, extracted_text)
        alt_text = generate_alt_text(caption, objects, extracted_text)
        return ImageAnalysis(
            caption=caption,
            alt_text=alt_text,
            objects=objects,
            extracted_text=extracted_text,
            tags=tags,
        )


def generate_alt_text(caption: str, objects: Iterable[DetectedObject], extracted_text: str) -> str:
    labels = unique_labels(objects)
    pieces = [caption.rstrip(".")]
    if labels:
        pieces.append("Visible objects include " + ", ".join(labels[:5]))
    if extracted_text:
        pieces.append(f'Text in the image reads "{extracted_text}"')
    return ". ".join(piece for piece in pieces if piece) + "."


def generate_tags(caption: str, objects: Iterable[DetectedObject], extracted_text: str) -> list[str]:
    stop_words = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "in",
        "is",
        "near",
        "of",
        "on",
        "the",
        "to",
        "with",
    }
    words = [
        word.strip(".,:;!?()[]{}\"'").lower()
        for word in f"{caption} {extracted_text}".split()
    ]
    caption_tags = [word for word in words if len(word) > 2 and word not in stop_words]
    object_tags = [label.lower() for label in unique_labels(objects)]

    counts = Counter(object_tags + caption_tags)
    return [tag for tag, _ in counts.most_common(12)]


def unique_labels(objects: Iterable[DetectedObject]) -> list[str]:
    labels: list[str] = []
    for detected in objects:
        if detected.label not in labels:
            labels.append(detected.label)
    return labels
