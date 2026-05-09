# Visual Intelligence Assistant

## Final Project Report

**Course:** Digital Image Processing  
**Project:** Image Captioning and Visual Question Answering System  
**Date:** May 6, 2026

---

## Abstract

This project implements a multimodal visual intelligence assistant that can analyze an input image and produce a natural-language caption, answer visual questions, detect objects, extract readable text, generate accessibility-style alt text, and save searchable image-analysis history. The main training work focuses on image captioning: a BLIP captioning model was fine-tuned on the project caption dataset and compared against the original pretrained BLIP baseline. Pretrained BLIP VQA, DETR, and TrOCR models were used as supporting modules for question answering, object localization, and OCR so the trained captioning model could be integrated into a complete visual intelligence system. A Streamlit application provides an interactive user interface, while a FastAPI service exposes the same functionality through HTTP endpoints.

The project was evaluated on a Flickr-style image-caption dataset containing 8,091 images and 40,455 captions. The captioning model was trained on the training split and evaluated on the validation split using BLEU scores. Caption-derived pseudo-VQA sets were also generated to measure the accuracy of the pretrained VQA component across action, color, count, object, and scene question types. The strongest empirical VQA evaluation used the full validation-derived pseudo-VQA set with 9,231 questions from 1,605 validation images. On this larger test set, the VQA system achieved 47.47% exact accuracy and 52.37% relaxed accuracy.

---

## 1. Introduction

Image captioning is the task of generating a natural-language description for an image. It connects computer vision and natural language processing by requiring a model to identify visual content and express it as readable text. Visual question answering extends this idea by allowing a user to ask a natural-language question about an image and receive a direct answer.

Traditional image-processing systems usually stop at low-level or mid-level features such as edges, color histograms, keypoints, or detected objects. This project moves toward semantic image understanding by using transformer-based models that connect visual features with language. The resulting assistant can be used for accessibility, image search, content organization, e-commerce descriptions, and interactive visual inspection.

---

## 2. Objectives

The main objectives of this project were:

1. Fine-tune a BLIP-based image captioning model that generates descriptive captions for uploaded images.
2. Add visual question answering so users can ask questions about image content.
3. Detect visible objects and display bounding boxes.
4. Extract readable text from images using OCR.
5. Generate alt text and searchable tags from the combined analysis.
6. Store previous image analyses in a local searchable history.
7. Provide both a user-facing web app and a programmatic API.
8. Evaluate captioning and VQA performance using measurable metrics.

---

## 3. Dataset

The project uses a Flickr-style caption dataset. The local dataset is stored in `data/` and contains image files under `data/Images/` with caption annotations in `data/captions.txt`.

### Dataset Summary

| Split/File | Rows | Unique Images | Description |
|---|---:|---:|---|
| `data/captions.txt` | 40,455 | 8,091 | Full caption annotation file |
| `data/train.csv` | 32,360 | 6,472 | Training split |
| `data/val.csv` | 8,095 | 1,619 | Validation/evaluation split |
| `data/vqa_test.csv` | 50 | 17 | Small caption-derived pseudo-VQA sanity test |
| `data/vqa_val_test.csv` | 500 | 251 | Balanced validation-derived pseudo-VQA subset |
| `data/vqa_val_full.csv` | 9,231 | 1,605 | Full validation-derived pseudo-VQA test set |

Each image has multiple human-written captions. For example, one image may have five different descriptions written from slightly different perspectives. This is useful for captioning because there is not only one correct way to describe an image.

### VQA Test Set

The original dataset contains captions, not manually annotated visual questions. To evaluate VQA, pseudo-VQA datasets were generated from the captions using `build_vqa_from_captions.py`. The full validation-derived file is `data/vqa_val_full.csv`. It contains all extractable VQA examples from `data/val.csv`, so it is a stronger empirical test than the original 50-question sample.

The full validation-derived VQA test set contains five question categories:

| Question Type | Count |
|---|---:|
| action | 1,301 |
| color | 768 |
| count | 3,214 |
| object | 2,802 |
| scene | 1,146 |

Because the VQA questions are generated from captions, this evaluation should be treated as a practical project evaluation rather than a fully human-annotated benchmark. The larger validation-derived set improves empirical strength by increasing coverage, but it still inherits noise from caption wording and automatic question generation.

---

## 4. System Design

The system is organized as a modular Python project. Each major capability is implemented in a separate module and then combined in the Streamlit app and FastAPI service.

### Main Components

| Component | File | Purpose |
|---|---|---|
| Caption generation | `image_captioning/captioner.py` | Generates captions and product descriptions using the trained BLIP captioning checkpoint |
| Visual question answering | `image_captioning/vqa.py` | Answers questions about an image using the pretrained BLIP VQA component |
| Object detection, OCR, tags, alt text | `image_captioning/intelligence.py` | Combines the trained captioner with pretrained DETR, TrOCR/OCR, tags, and alt text |
| History storage | `image_captioning/history.py` | Saves image analyses and supports local search |
| Web app | `app.py` | Streamlit interface for upload, analysis, VQA, and history |
| API service | `api.py` | FastAPI endpoints for captioning, VQA, analysis, and history |
| Caption evaluation | `evaluate_captions.py` | Computes BLEU metrics |
| VQA evaluation | `evaluate_vqa.py` | Computes exact and relaxed VQA accuracy |
| VQA data generation | `build_vqa_from_captions.py` | Builds pseudo-VQA questions from captions |

### Processing Pipeline

The image-processing pipeline works as follows:

1. The user uploads an image through the Streamlit app or sends an image to the API.
2. The image is loaded with PIL and converted to RGB.
3. The fine-tuned BLIP captioning model generates a natural-language caption.
4. The pretrained DETR detector identifies objects and returns labels, confidence scores, and bounding boxes.
5. The OCR module extracts readable text from the image using TrOCR.
6. The system generates tags from the caption, detected objects, and OCR output.
7. Alt text is generated by combining the caption, object labels, and OCR text.
8. The user can ask a question about the image.
9. If the question asks about visible text, OCR output is used as context before falling back to BLIP VQA.
10. The analysis can be saved to local history and searched later.

---

## 5. Models and Methods

### 5.1 BLIP for Image Captioning

BLIP is the primary model used for image captioning. It combines visual representation learning with language generation, allowing the system to convert an input image into a natural-language description. In this project, the captioning task was not limited to direct pretrained inference. The pretrained Hugging Face BLIP captioning model was used as the starting checkpoint, then fine-tuned on the project caption dataset to create a trained image-captioning model. The original pretrained BLIP model was kept as a baseline for comparison.

The `ImageCaptioner` class wraps the BLIP captioning model and exposes methods for normal captions, product-style descriptions, and batch caption generation. The model accepts an RGB image and optionally a text prompt, then generates a text sequence using beam search. Beam search is a decoding strategy that improves caption quality by considering multiple likely candidate sequences instead of greedily selecting only the most likely next token.

### 5.2 BLIP VQA for Visual Question Answering

The VQA module uses `Salesforce/blip-vqa-base` as a pretrained supporting model. This part of the project was used for visual question answering and accuracy evaluation, while the main trained model in the project is the fine-tuned BLIP captioner. The VQA model receives an image and a natural-language question, then generates an answer. The system also includes OCR-aware handling for text-reading questions. For example, if the user asks "What does the sign say?", the system first checks whether OCR extracted text from the image. If OCR text is available, that text is returned directly because OCR is usually more reliable for reading printed text than a general VQA model.

### 5.3 DETR for Object Detection

DETR is used as a pretrained object detection support model. Unlike traditional object detectors that rely heavily on anchor boxes and non-maximum suppression, DETR formulates detection as a direct set prediction problem. In this project, DETR returns object labels, confidence scores, and bounding boxes. The Streamlit app draws these bounding boxes on top of the uploaded image so the user can visually inspect detections.

### 5.4 TrOCR for Optical Character Recognition

TrOCR is used for OCR text extraction as a supporting component. The project includes a training script for fine-tuning TrOCR, and the runtime defaults to a local fine-tuned OCR checkpoint if `checkpoints/trocr-ocr` exists; otherwise it uses `microsoft/trocr-base-printed`. OCR output is included in the generated alt text, searchable tags, and VQA text-question handling.

### 5.5 History Search

The local history module stores each uploaded image and its analysis in `outputs/history/`. A JSON file stores metadata, captions, tags, objects, OCR text, and generated alt text. The Streamlit app supports searching previous images by caption, tag, object name, or OCR text.

---

## 6. Implementation

The implementation is divided into reusable modules rather than placing all logic in the app file. This makes the project easier to test, reuse, and extend.

### Model Training

The main trained model in the project is the BLIP image-captioning model. The `train_blip.py` script fine-tunes `Salesforce/blip-image-captioning-base` on `data/train.csv`, validates it using `data/val.csv`, and saves the trained checkpoint to `checkpoints/blip-captioner`. The Streamlit app and FastAPI service use this checkpoint as the default captioning model, while the original pretrained BLIP model is retained as a baseline for comparison during evaluation.

The project also includes `train_ocr.py` for optional TrOCR fine-tuning when OCR-specific training data is available. In the current system design, VQA, object detection, and OCR are supporting components around the trained captioning model.

### Streamlit Application

The Streamlit app supports:

- image upload
- caption generation
- detected-object visualization with bounding boxes
- OCR text display
- VQA question input
- generated alt text
- searchable tags
- local history search

The app caches model objects using `st.cache_resource`, which prevents the large transformer models from being reloaded on every interaction.

### FastAPI Service

The API exposes endpoints for:

- `/caption`
- `/vqa`
- `/analyze`
- `/product-description`
- `/live/analyze`
- `/history`
- `/history/search`

The API validates image type, image size, and upload size before inference. It also uses `run_in_threadpool` so model inference does not block the async request handler directly.

### Evaluation Scripts

Two evaluation scripts are included:

- `evaluate_captions.py` computes BLEU-1 and BLEU-4 for generated captions.
- `evaluate_vqa.py` computes exact accuracy and relaxed accuracy for VQA.

The relaxed VQA metric accepts answers that are semantically close in simple ways, such as matching numeric answers or allowing short expected answers to appear inside a longer predicted answer.

---

## 7. Experimental Results

### 7.1 Captioning Results

The captioning experiment evaluated the trained BLIP captioning checkpoint against the original pretrained BLIP baseline on 8,095 validation caption rows.

| Model Run | Prediction File | Rows | BLEU-1 | BLEU-4 |
|---|---|---:|---:|---:|
| Pretrained BLIP baseline | `outputs/eval_predictions.csv` | 8,095 | 0.4345 | 0.1166 |
| Fine-tuned BLIP captioner | `outputs/eval_finetuned_predictions.csv` | 8,095 | 0.4183 | 0.1149 |

The BLEU values in the table are strict single-reference row-wise scores. Each generated caption is compared against one reference caption row at a time, even though the Flickr-style dataset contains multiple valid captions for the same image. The fine-tuned captioner received a BLEU-1 score of 0.4183 and a BLEU-4 score of 0.1149. The pretrained baseline received a BLEU-1 score of 0.4345 and a BLEU-4 score of 0.1166, so the baseline remained slightly higher in this saved run. BLEU-4 is lower because exact four-word phrase matches are much harder when the generated caption uses different wording. For example, a caption can correctly describe a black dog running through water while still differing from a reference phrase such as "running in the surf." In the aggregate evaluation file, the baseline predictions had higher multi-reference image-wise scores, with BLEU-1 of 0.7388 and BLEU-4 of 0.3229.

The fine-tuned prediction file did not outperform the saved baseline prediction file in the current run. This does not mean that no training was performed; it means that the completed fine-tuning run did not improve over the strong pretrained BLIP baseline under these settings. Possible reasons include limited training time, hyperparameter settings, dataset size, or the fact that the base BLIP model already performs well on general image-caption data.

### 7.2 Visual Question Answering Results

The VQA accuracy results are for the pretrained BLIP VQA supporting component, not for the trained captioning checkpoint. The main VQA evaluation was run on the full validation-derived pseudo-VQA set. For execution stability, the 9,231 questions were evaluated in chunks and the prediction files were merged into `outputs/vqa_val_full_eval_predictions.csv`. This is equivalent to evaluating:

```powershell
python evaluate_vqa.py --test-csv data\vqa_val_full.csv --image-root data --model Salesforce/blip-vqa-base --output outputs\vqa_val_full_eval_predictions.csv --device cuda
```

Overall VQA results:

| Test File | Questions | Unique Images | Exact Accuracy | Relaxed Accuracy |
|---|---:|---:|---:|---:|
| `data/vqa_test.csv` | 50 | 17 | 0.5600 | 0.6000 |
| `data/vqa_val_test.csv` | 500 | 251 | 0.4940 | 0.5880 |
| `data/vqa_val_full.csv` | 9,231 | 1,605 | 0.4747 | 0.5237 |

Per-question-type result on the full validation-derived set:

| Question Type | Questions | Exact Accuracy | Relaxed Accuracy |
|---|---:|---:|---:|
| action | 1,301 | 0.3812 | 0.4650 |
| color | 768 | 0.6927 | 0.7682 |
| count | 3,214 | 0.6217 | 0.6217 |
| object | 2,802 | 0.4383 | 0.4568 |
| scene | 1,146 | 0.1117 | 0.3150 |

The pretrained VQA component performed best on color and count questions. Object and action questions were more difficult, and scene questions performed poorly because the generated labels were often narrow, such as `road` or `grass`, while the model might answer with broader phrases such as `outside`, `field`, or `on road`. This shows that scene evaluation needs either manually curated labels or a better synonym-aware scoring method. The full validation-derived score is lower than the 50-question score, but it is more reliable because it evaluates 9,231 questions across 1,605 validation images.

---

## 8. Discussion

The project demonstrates that a trained captioning model can be integrated into a larger vision-language assistant. The most successful parts of the system are image captioning, color/object VQA, object detection, and searchable image history. The supporting pretrained modules add practical capabilities around the trained captioner: VQA provides question-answering accuracy, DETR provides object localization, and OCR adds text extraction. The object detection overlay makes the system easier to interpret because users can see what the detector found instead of only reading labels in a table.

The OCR-aware VQA improvement is also important. General VQA models often struggle to read text accurately, especially when the text is small or stylized. By using TrOCR output for text-related questions, the system handles questions about signs, labels, and printed words more directly.

The evaluation results also reveal limitations. BLEU scores do not always reflect human judgment because a generated caption can be correct while using different wording from the reference caption. Similarly, the pseudo-VQA dataset is useful for testing, but it is not as strong as a manually annotated VQA benchmark. The full validation-derived VQA set strengthens the empirical evidence by using 9,231 questions instead of only 50, but the low scene-question score shows that generated labels from captions can be too strict.

---

## 9. Limitations

The main limitations are:

1. The VQA test set is generated from captions, not manually annotated by humans.
2. Scene questions are difficult to score because many answers can be correct.
3. BLEU can penalize valid captions that do not match the reference wording.
4. OCR performance depends on text size, clarity, orientation, and image quality.
5. DETR detects only object categories included in its pretrained label set.
6. The system can be slow on CPU because it loads and runs multiple transformer models.
7. The local JSON history is suitable for a demo but not ideal for a large production system.

---

## 10. Future Work

Future improvements include:

1. Create a manually reviewed VQA test set with more reliable labels.
2. Add synonym-aware scoring for scene and action questions.
3. Continue fine-tuning the captioning model for more epochs and compare training settings.
4. Add image segmentation to identify object regions more precisely.
5. Improve OCR by cropping detected text regions before passing them to TrOCR.
6. Store history in SQLite instead of JSON for better search and scalability.
7. Add a report export button that saves captions, objects, OCR text, and VQA answers as a PDF.
8. Add support for newer multimodal models if stronger GPU resources are available.

---

## 11. Conclusion

This project successfully implements a visual intelligence assistant centered on a fine-tuned BLIP image-captioning model. It combines the trained caption generator with supporting visual question answering, object detection, OCR, alt-text generation, tag generation, history search, a Streamlit interface, and a FastAPI service.

The fine-tuned captioning model achieved a BLEU-1 score of 0.4183 and a BLEU-4 score of 0.1149 on the saved validation run, while the pretrained BLIP baseline achieved 0.4345 and 0.1166. For the pretrained VQA supporting component, the system achieved 47.47% exact accuracy and 52.37% relaxed accuracy on the full validation-derived VQA set with 9,231 questions. The strongest VQA categories were color and count questions, while scene questions were the weakest due to strict caption-derived labels.

Overall, the project demonstrates how a fine-tuned captioning model can be trained, evaluated, and integrated with pretrained support models to create a complete image-understanding application with measurable evaluation results.

---

## References

1. Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. "BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation." arXiv:2201.12086, 2022. https://huggingface.co/papers/2201.120862. Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, and Sergey Zagoruyko. "End-to-End Object Detection with Transformers." ECCV, 2020. https://huggingface.co/papers/2005.12872
3. Minghao Li, Tengchao Lv, Jingye Chen, Lei Cui, Yijuan Lu, Dinei Florencio, Cha Zhang, Zhoujun Li, and Furu Wei. "TrOCR: Transformer-based Optical Character Recognition with Pre-trained Models." AAAI, 2023. https://www.microsoft.com/en-us/research/publication/trocr-transformer-based-optical-character-recognition-with-pre-trained-models/
4. Micah Hodosh, Peter Young, and Julia Hockenmaier. "Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics." Journal of Artificial Intelligence Research, 2013. Flickr8k dataset reference: https://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b
