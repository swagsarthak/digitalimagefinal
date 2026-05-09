from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

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


st.set_page_config(page_title="Visual Intelligence Assistant", page_icon="image", layout="wide")

DEFAULT_CAPTION_MODEL = "checkpoints/blip-captioner"
DEFAULT_VQA_MODEL = "Salesforce/blip-vqa-base"
DEFAULT_DETECTION_MODEL = "facebook/detr-resnet-50"
DEFAULT_OCR_MODEL = "checkpoints/trocr-ocr" if Path("checkpoints/trocr-ocr").exists() else "microsoft/trocr-base-printed"
BOX_COLORS = ["#0f766e", "#b45309", "#6d28d9", "#be123c", "#0369a1", "#4d7c0f"]


@st.cache_resource(show_spinner="Loading captioning model...")
def get_captioner(model_name: str, max_new_tokens: int, num_beams: int) -> ImageCaptioner:
    config = CaptionConfig(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
    )
    return ImageCaptioner(config)


@st.cache_resource(show_spinner="Loading visual question answering model...")
def get_question_answerer(model_name: str, max_new_tokens: int) -> ImageQuestionAnswerer:
    config = VQAConfig(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
    )
    return ImageQuestionAnswerer(config)


@st.cache_resource(show_spinner="Loading object detection model...")
def get_object_detector(model_name: str, threshold: float, max_objects: int) -> ImageObjectDetector:
    config = ObjectDetectionConfig(
        model_name=model_name,
        threshold=threshold,
        max_objects=max_objects,
    )
    return ImageObjectDetector(config)


@st.cache_resource(show_spinner="Loading OCR model...")
def get_ocr_reader(model_name: str, max_new_tokens: int) -> ImageOCRReader:
    config = OCRConfig(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
    )
    return ImageOCRReader(config)


@st.cache_resource
def get_history_store() -> ImageHistoryStore:
    return ImageHistoryStore()


def render_analysis(analysis: dict[str, object]) -> None:
    st.subheader("Caption")
    st.write(analysis.get("caption") or "")

    st.subheader("Alt text")
    st.write(analysis.get("alt_text") or "")

    tags = analysis.get("tags") or []
    if isinstance(tags, list) and tags:
        st.subheader("Searchable tags")
        st.write(", ".join(str(tag) for tag in tags))

    extracted_text = str(analysis.get("extracted_text") or "")
    st.subheader("OCR text")
    st.write(extracted_text if extracted_text else "No readable text detected.")

    objects = analysis.get("objects") or []
    st.subheader("Detected objects")
    if isinstance(objects, list) and objects:
        st.dataframe(
            [
                {
                    "object": item.get("label"),
                    "confidence": item.get("score"),
                    "box": item.get("box"),
                }
                for item in objects
                if isinstance(item, dict)
            ],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.write("No objects detected above the selected confidence threshold.")


def detected_object_value(item: object, field: str) -> object:
    if isinstance(item, dict):
        return item.get(field)
    return getattr(item, field, None)


def draw_detected_objects(image: Image.Image, objects: object) -> Image.Image:
    if not isinstance(objects, list) or not objects:
        return image

    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    image_width, image_height = canvas.size

    for index, item in enumerate(objects):
        box = detected_object_value(item, "box")
        if not isinstance(box, list) or len(box) != 4:
            continue

        try:
            x1, y1, x2, y2 = [float(value) for value in box]
        except (TypeError, ValueError):
            continue

        x1 = max(0, min(image_width, x1))
        y1 = max(0, min(image_height, y1))
        x2 = max(0, min(image_width, x2))
        y2 = max(0, min(image_height, y2))
        if x2 <= x1 or y2 <= y1:
            continue

        label = str(detected_object_value(item, "label") or "object")
        score = detected_object_value(item, "score")
        if isinstance(score, int | float):
            label = f"{label} {score:.0%}"

        color = BOX_COLORS[index % len(BOX_COLORS)]
        line_width = max(2, round(min(image_width, image_height) * 0.004))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

        text_bbox = draw.textbbox((x1, y1), label)
        text_width = text_bbox[2] - text_bbox[0] + 10
        text_height = text_bbox[3] - text_bbox[1] + 8
        label_y = max(0, y1 - text_height)
        draw.rectangle([x1, label_y, x1 + text_width, label_y + text_height], fill=color)
        draw.text((x1 + 5, label_y + 4), label, fill="white")

    return canvas


st.title("Visual Intelligence Assistant")

with st.sidebar:
    st.header("Models")
    caption_model_name = st.text_input("Caption and alt-text model", value=DEFAULT_CAPTION_MODEL)
    vqa_model_name = st.text_input("VQA model", value=DEFAULT_VQA_MODEL)
    detection_model_name = st.text_input("Object detection model", value=DEFAULT_DETECTION_MODEL)
    ocr_model_name = st.text_input("OCR model", value=DEFAULT_OCR_MODEL)

    st.header("Generation")
    max_new_tokens = st.slider("Caption length", min_value=10, max_value=100, value=50, step=5)
    num_beams = st.slider("Beam search", min_value=1, max_value=10, value=5, step=1)
    vqa_max_new_tokens = st.slider("VQA answer length", min_value=5, max_value=50, value=30, step=5)
    ocr_max_new_tokens = st.slider("OCR length", min_value=16, max_value=128, value=64, step=16)

    st.header("Detection")
    detection_threshold = st.slider("Confidence threshold", min_value=0.1, max_value=0.95, value=0.7, step=0.05)
    max_objects = st.slider("Maximum objects", min_value=3, max_value=30, value=12, step=1)

history_store = get_history_store()
assistant_tab, history_tab = st.tabs(["Assistant", "History Search"])

with assistant_tab:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        upload_key = f"{uploaded_file.name}:{uploaded_file.size}"
        if st.session_state.get("last_upload_key") != upload_key:
            st.session_state["last_upload_key"] = upload_key
            st.session_state.pop("current_analysis", None)
            st.session_state.pop("current_answer", None)
            st.session_state.pop("current_image_id", None)

        left_col, right_col = st.columns([0.9, 1.1], vertical_alignment="top")

        with left_col:
            image_slot = st.empty()

            if st.button("Analyze Image", type="primary", use_container_width=True):
                captioner = get_captioner(caption_model_name, max_new_tokens, num_beams)
                detector = get_object_detector(detection_model_name, detection_threshold, max_objects)
                ocr_reader = get_ocr_reader(ocr_model_name, ocr_max_new_tokens)
                analyzer = VisualIntelligenceAnalyzer(captioner, detector, ocr_reader)
                analysis = analyzer.analyze(image).to_dict()
                saved = history_store.save(image, uploaded_file.name, analysis)
                st.session_state["current_analysis"] = analysis
                st.session_state["current_image_id"] = saved.id

            current_analysis = st.session_state.get("current_analysis")
            if isinstance(current_analysis, dict):
                image_slot.image(
                    draw_detected_objects(image, current_analysis.get("objects")),
                    caption="Uploaded image with detected objects",
                    use_container_width=True,
                )
            else:
                image_slot.image(image, caption="Uploaded image", use_container_width=True)

            question = st.text_input(
                "Ask a question about this image",
                value="What is happening in this image?",
                placeholder="Example: What color is the car?",
            )
            if st.button("Answer Question", use_container_width=True):
                if not question.strip():
                    st.warning("Enter a question first.")
                else:
                    question_answerer = get_question_answerer(vqa_model_name, vqa_max_new_tokens)
                    analysis = st.session_state.get("current_analysis")
                    extracted_text = ""
                    if isinstance(analysis, dict):
                        extracted_text = str(analysis.get("extracted_text") or "")
                    if not extracted_text and question_answerer.is_text_question(question):
                        ocr_reader = get_ocr_reader(ocr_model_name, ocr_max_new_tokens)
                        extracted_text = ocr_reader.extract_text(image)
                    st.session_state["current_answer"] = question_answerer.answer_with_context(
                        image,
                        question,
                        extracted_text=extracted_text,
                    )

            if "current_answer" in st.session_state:
                st.subheader("Answer")
                st.write(st.session_state["current_answer"])

        with right_col:
            analysis = st.session_state.get("current_analysis")
            if isinstance(analysis, dict):
                render_analysis(analysis)
            else:
                st.info("Upload an image and run analysis to generate caption, objects, OCR text, tags, and alt text.")

with history_tab:
    top_col, action_col = st.columns([1, 0.25])
    with top_col:
        query = st.text_input("Search history", placeholder="Search by caption, tag, detected object, or OCR text")
    with action_col:
        st.write("")
        st.write("")
        if st.button("Clear History", use_container_width=True):
            history_store.clear()
            st.rerun()

    records = history_store.search(query)
    if not records:
        st.info("No matching uploaded images found.")
    else:
        for record in records:
            with st.expander(f"{record.filename} - {record.created_at}", expanded=False):
                image_path = Path(record.image_path)
                if image_path.exists():
                    st.image(str(image_path), width=280)
                render_analysis(record.analysis)
