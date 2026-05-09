from __future__ import annotations

import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from image_captioning import (
    CaptionConfig,
    ImageCaptioner,
    ImageHistoryStore,
    ImageOCRReader,
    ImageObjectDetector,
    ImageQuestionAnswerer,
    OCRConfig,
    ObjectDetectionConfig,
    VQAConfig,
    VisualIntelligenceAnalyzer,
)


DEFAULT_MODEL = os.getenv("CAPTION_MODEL", "checkpoints/blip-captioner")
DEFAULT_VQA_MODEL = os.getenv("VQA_MODEL", "Salesforce/blip-vqa-base")
DEFAULT_DEVICE = os.getenv("CAPTION_DEVICE") or None
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("CAPTION_MAX_NEW_TOKENS", "50"))
DEFAULT_NUM_BEAMS = int(os.getenv("CAPTION_NUM_BEAMS", "5"))
DEFAULT_VQA_MAX_NEW_TOKENS = int(os.getenv("VQA_MAX_NEW_TOKENS", "30"))
DEFAULT_DETECTION_MODEL = os.getenv("DETECTION_MODEL", "facebook/detr-resnet-50")
DEFAULT_DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.7"))
DEFAULT_MAX_OBJECTS = int(os.getenv("DETECTION_MAX_OBJECTS", "12"))
DEFAULT_LOCAL_OCR_MODEL = "checkpoints/trocr-ocr" if Path("checkpoints/trocr-ocr").exists() else "microsoft/trocr-base-printed"
DEFAULT_OCR_MODEL = os.getenv("OCR_MODEL", DEFAULT_LOCAL_OCR_MODEL)
DEFAULT_OCR_MAX_NEW_TOKENS = int(os.getenv("OCR_MAX_NEW_TOKENS", "64"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "25000000"))
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}
LIVE_ASSISTANT_PATH = Path(__file__).with_name("live_assistant.html")
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("API_CORS_ORIGINS", "*").split(",")
    if origin.strip()
]

app = FastAPI(title="Image Understanding API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HealthResponse(BaseModel):
    status: str
    caption_model: str
    vqa_model: str
    detection_model: str
    ocr_model: str


class ModelReadiness(BaseModel):
    model: str
    available: bool
    source: str


class ReadyResponse(BaseModel):
    status: str
    models: dict[str, ModelReadiness]


class CaptionResponse(BaseModel):
    filename: str
    caption: str


class VQAResponse(BaseModel):
    filename: str
    question: str
    answer: str


class ProductDescriptionResponse(BaseModel):
    filename: str
    description: str


class DetectedObjectResponse(BaseModel):
    label: str
    score: float
    box: list[float]


class ImageAnalysisResponse(BaseModel):
    caption: str
    alt_text: str
    objects: list[DetectedObjectResponse]
    extracted_text: str
    tags: list[str]


class AnalyzeResponse(BaseModel):
    filename: str
    id: str
    analysis: ImageAnalysisResponse


class LiveAnalyzeResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    analysis: ImageAnalysisResponse
    question: str | None = None
    answer: str | None = None
    saved_id: str | None = None


class HistoryRecordResponse(BaseModel):
    id: str
    filename: str
    image_path: str
    created_at: str
    analysis: dict[str, Any] = Field(default_factory=dict)


@lru_cache(maxsize=1)
def get_captioner() -> ImageCaptioner:
    config = CaptionConfig(
        model_name=DEFAULT_MODEL,
        device=DEFAULT_DEVICE,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        num_beams=DEFAULT_NUM_BEAMS,
    )
    return ImageCaptioner(config)


@lru_cache(maxsize=1)
def get_question_answerer() -> ImageQuestionAnswerer:
    config = VQAConfig(
        model_name=DEFAULT_VQA_MODEL,
        device=DEFAULT_DEVICE,
        max_new_tokens=DEFAULT_VQA_MAX_NEW_TOKENS,
    )
    return ImageQuestionAnswerer(config)


@lru_cache(maxsize=1)
def get_object_detector() -> ImageObjectDetector:
    config = ObjectDetectionConfig(
        model_name=DEFAULT_DETECTION_MODEL,
        device=DEFAULT_DEVICE,
        threshold=DEFAULT_DETECTION_THRESHOLD,
        max_objects=DEFAULT_MAX_OBJECTS,
    )
    return ImageObjectDetector(config)


@lru_cache(maxsize=1)
def get_ocr_reader() -> ImageOCRReader:
    config = OCRConfig(
        model_name=DEFAULT_OCR_MODEL,
        device=DEFAULT_DEVICE,
        max_new_tokens=DEFAULT_OCR_MAX_NEW_TOKENS,
    )
    return ImageOCRReader(config)


@lru_cache(maxsize=1)
def get_history_store() -> ImageHistoryStore:
    return ImageHistoryStore()


def model_readiness(model_name: str) -> ModelReadiness:
    path = Path(model_name)
    if path.exists():
        return ModelReadiness(model=model_name, available=True, source="local")

    looks_local = path.is_absolute() or model_name.startswith((".", "checkpoints")) or "\\" in model_name
    return ModelReadiness(
        model=model_name,
        available=not looks_local,
        source="missing_local" if looks_local else "remote_or_cached",
    )


def model_error(exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=f"Model inference failed: {exc}",
    )


def history_response(record: object) -> dict[str, object]:
    return record.__dict__


def caption_sync(image: Image.Image, prompt: str | None) -> str:
    return get_captioner().caption(image, prompt=prompt)


def answer_sync(image: Image.Image, question: str) -> str:
    question_answerer = get_question_answerer()
    extracted_text = None
    if question_answerer.is_text_question(question):
        extracted_text = get_ocr_reader().extract_text(image)
    return question_answerer.answer_with_context(image, question, extracted_text=extracted_text)


def product_description_sync(image: Image.Image) -> str:
    return get_captioner().product_description(image)


def analyze_sync(image: Image.Image, filename: str) -> tuple[dict[str, object], str]:
    analyzer = VisualIntelligenceAnalyzer(
        get_captioner(),
        get_object_detector(),
        get_ocr_reader(),
    )
    analysis = analyzer.analyze(image).to_dict()
    saved = get_history_store().save(image, filename, analysis)
    return analysis, saved.id


def live_analyze_sync(
    image: Image.Image,
    filename: str,
    question: str | None,
    save: bool,
) -> tuple[dict[str, object], str | None, str | None]:
    analyzer = VisualIntelligenceAnalyzer(
        get_captioner(),
        get_object_detector(),
        get_ocr_reader(),
    )
    analysis = analyzer.analyze(image).to_dict()
    answer = (
        get_question_answerer().answer_with_context(
            image,
            question,
            extracted_text=str(analysis.get("extracted_text") or ""),
        )
        if question
        else None
    )
    saved_id = get_history_store().save(image, filename, analysis).id if save else None
    return analysis, answer, saved_id


async def read_image(upload: UploadFile) -> Image.Image:
    if upload.content_type and upload.content_type not in ALLOWED_IMAGE_TYPES | {"application/octet-stream"}:
        raise HTTPException(status_code=415, detail=f"Unsupported image type: {upload.content_type}.")

    content = await upload.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"Image upload exceeds {max_mb:.1f} MB.")

    try:
        pil_image = Image.open(io.BytesIO(content))
        if pil_image.width * pil_image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image dimensions are too large.")
        return pil_image.convert("RGB")
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="Image dimensions are too large.") from exc
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail="Uploaded image could not be read.") from exc


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        caption_model=DEFAULT_MODEL,
        vqa_model=DEFAULT_VQA_MODEL,
        detection_model=DEFAULT_DETECTION_MODEL,
        ocr_model=DEFAULT_OCR_MODEL,
    )


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    models = {
        "caption": model_readiness(DEFAULT_MODEL),
        "vqa": model_readiness(DEFAULT_VQA_MODEL),
        "detection": model_readiness(DEFAULT_DETECTION_MODEL),
        "ocr": model_readiness(DEFAULT_OCR_MODEL),
    }
    status = "ready" if all(model.available for model in models.values()) else "degraded"
    return ReadyResponse(status=status, models=models)


@app.get("/live", response_class=HTMLResponse)
def live_assistant() -> HTMLResponse:
    if not LIVE_ASSISTANT_PATH.exists():
        raise HTTPException(status_code=404, detail="Live assistant page not found.")
    return HTMLResponse(LIVE_ASSISTANT_PATH.read_text(encoding="utf-8"))


@app.post("/caption", response_model=CaptionResponse)
async def caption_image(
    image: UploadFile = File(...),
    prompt: str | None = Form(default=None),
) -> CaptionResponse:
    pil_image = await read_image(image)
    cleaned_prompt = prompt.strip() if prompt and prompt.strip() else None
    try:
        caption = await run_in_threadpool(caption_sync, pil_image, cleaned_prompt)
    except Exception as exc:
        raise model_error(exc) from exc
    return CaptionResponse(filename=image.filename or "", caption=caption)


@app.post("/vqa", response_model=VQAResponse)
async def answer_image_question(
    image: UploadFile = File(...),
    question: str = Form(...),
) -> VQAResponse:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pil_image = await read_image(image)
    cleaned_question = question.strip()
    try:
        answer = await run_in_threadpool(answer_sync, pil_image, cleaned_question)
    except Exception as exc:
        raise model_error(exc) from exc
    return VQAResponse(filename=image.filename or "", question=cleaned_question, answer=answer)


@app.post("/product-description", response_model=ProductDescriptionResponse)
async def describe_product(
    image: UploadFile = File(...),
) -> ProductDescriptionResponse:
    pil_image = await read_image(image)
    try:
        description = await run_in_threadpool(product_description_sync, pil_image)
    except Exception as exc:
        raise model_error(exc) from exc
    return ProductDescriptionResponse(filename=image.filename or "", description=description)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(
    image: UploadFile = File(...),
) -> AnalyzeResponse:
    pil_image = await read_image(image)
    try:
        analysis_dict, saved_id = await run_in_threadpool(analyze_sync, pil_image, image.filename or "")
    except Exception as exc:
        raise model_error(exc) from exc

    return AnalyzeResponse(filename=image.filename or "", id=saved_id, analysis=analysis_dict)


@app.post("/live/analyze", response_model=LiveAnalyzeResponse)
async def analyze_live_frame(
    image: UploadFile = File(...),
    question: str | None = Form(default=None),
    save: bool = Form(default=False),
) -> LiveAnalyzeResponse:
    pil_image = await read_image(image)
    cleaned_question = question.strip() if question and question.strip() else None
    try:
        analysis_dict, answer, saved_id = await run_in_threadpool(
            live_analyze_sync,
            pil_image,
            image.filename or "live-frame.jpg",
            cleaned_question,
            save,
        )
    except Exception as exc:
        raise model_error(exc) from exc

    return LiveAnalyzeResponse(
        filename=image.filename or "",
        image_width=pil_image.width,
        image_height=pil_image.height,
        analysis=analysis_dict,
        question=cleaned_question,
        answer=answer,
        saved_id=saved_id,
    )


@app.get("/history", response_model=list[HistoryRecordResponse])
def list_history() -> list[dict[str, object]]:
    return [history_response(record) for record in get_history_store().list()]


@app.delete("/history", status_code=204)
def clear_history() -> Response:
    get_history_store().clear()
    return Response(status_code=204)


@app.get("/history/search", response_model=list[HistoryRecordResponse])
def search_history(q: str = "") -> list[dict[str, object]]:
    return [history_response(record) for record in get_history_store().search(q)]


@app.get("/history/{image_id}", response_model=HistoryRecordResponse)
def get_history_record(image_id: str) -> dict[str, object]:
    record = get_history_store().get(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="History record not found.")
    return history_response(record)


@app.get("/history/{image_id}/image")
def get_history_image(image_id: str) -> FileResponse:
    record = get_history_store().get(image_id)
    if record is None:
        raise HTTPException(status_code=404, detail="History record not found.")

    image_path = Path(record.image_path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Stored image file not found.")
    return FileResponse(image_path)


@app.delete("/history/{image_id}", status_code=204)
def delete_history_record(image_id: str) -> Response:
    deleted = get_history_store().delete(image_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="History record not found.")
    return Response(status_code=204)
