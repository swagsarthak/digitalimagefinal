# Visual Intelligence Assistant

This project is a Python image understanding application. It can analyze an uploaded image and produce:

- image captions
- visual question answering
- object detection
- OCR text extraction
- alt text
- searchable tags
- product descriptions
- saved image history and history search
- live webcam analysis through a browser page

The main user interface is the Streamlit app in `app.py`. The project also includes a FastAPI service in `api.py`, a command-line captioning script, and training/evaluation scripts.

## Requirements

- Python 3.10 or newer
- Internet access for the first model download from Hugging Face
- A GPU is recommended for training and faster inference, but CPU can run the app slowly

Install dependencies from `requirements.txt`.

## Project Structure

```text
final_project/
  app.py                         Streamlit web app
  api.py                         FastAPI backend and live camera API
  live_assistant.html            Browser live camera page
  cli_caption.py                 Command-line caption generator
  train_blip.py                  Caption model fine-tuning
  train_ocr.py                   OCR model fine-tuning
  evaluate_captions.py           Caption evaluation
  evaluate_vqa.py                VQA evaluation
  build_vqa_from_captions.py     Utility to create simple VQA rows
  image_captioning/              Core Python package
  data/                          Dataset files
  checkpoints/                   Local trained model checkpoints
  outputs/                       Generated outputs
```

## Setup

Open PowerShell in the project folder:

```powershell
cd "C:\Users\91797\Desktop\MSCS\DIGITAL IMAGE PROCESSING\final_project"
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, run this once for the current terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Install the packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the Streamlit App

Start the main app:

```powershell
streamlit run app.py
```

Open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

Use the app like this:

1. Upload a JPG, PNG, or WEBP image.
2. Click `Analyze Image`.
3. Review the caption, detected objects, OCR text, alt text, and tags.
4. Ask a question about the image, such as `What color is the car?`.
5. Use the `History Search` tab to search previously analyzed images.

By default, the app uses:

- caption model: `checkpoints/blip-captioner`
- VQA model: `Salesforce/blip-vqa-base`
- object detection model: `facebook/detr-resnet-50`
- OCR model: `checkpoints/trocr-ocr` if it exists, otherwise `microsoft/trocr-base-printed`

If a local checkpoint is missing, replace it in the app sidebar with a Hugging Face model name such as:

```text
Salesforce/blip-image-captioning-base
```

## Run the FastAPI Service

Start the API:

```powershell
uvicorn api:app --host 127.0.0.1 --port 8000
```

Check that it is running:

```text
http://127.0.0.1:8000/health
```

Check model readiness:

```text
http://127.0.0.1:8000/ready
```

Open the live camera assistant:

```text
http://127.0.0.1:8000/live
```

The live assistant uses your browser camera and sends frames to the API for captioning, object detection, OCR, tags, and optional question answering.

## API Examples

Analyze an image:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/analyze" -F "image=@data\Images\1000268201_693b08cb0e.jpg"
```

Generate only a caption:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/caption" -F "image=@data\Images\1000268201_693b08cb0e.jpg"
```

Ask a question about an image:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/vqa" -F "image=@data\Images\1000268201_693b08cb0e.jpg" -F "question=What is happening in this image?"
```

Generate a product description:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/product-description" -F "image=@data\Images\1000268201_693b08cb0e.jpg"
```

Search saved history:

```powershell
curl.exe "http://127.0.0.1:8000/history/search?q=car"
```

## Run Captioning From the Command Line

Caption one image:

```powershell
python cli_caption.py --input data\Images\1000268201_693b08cb0e.jpg --model checkpoints\blip-captioner
```

Caption every image in a folder and save a CSV:

```powershell
python cli_caption.py --input data\Images --output outputs\captions.csv --model checkpoints\blip-captioner
```

Use the base Hugging Face model instead of a local checkpoint:

```powershell
python cli_caption.py --input data\Images --model Salesforce/blip-image-captioning-base
```

## Environment Variables

The API can be configured with environment variables before running `uvicorn`.

```powershell
$env:CAPTION_MODEL="checkpoints\blip-captioner"
$env:VQA_MODEL="Salesforce/blip-vqa-base"
$env:DETECTION_MODEL="facebook/detr-resnet-50"
$env:OCR_MODEL="microsoft/trocr-base-printed"
$env:CAPTION_DEVICE="cuda"
uvicorn api:app --host 127.0.0.1 --port 8000
```

Use `CAPTION_DEVICE="cpu"` if CUDA is not available or GPU memory is too limited.

## Dataset Format

Caption training CSV files should contain:

```csv
image_path,caption
Images/image_001.jpg,a car driving on a street
Images/image_002.jpg,a person standing beside a bicycle
```

The image path should be relative to the image root passed with `--image-root`.

Expected captioning layout:

```text
data/
  Images/
  train.csv
  val.csv
```

OCR training CSV files should contain:

```csv
image_path,text
ocr_images/sign_001.jpg,Green Valley School
ocr_images/receipt_001.jpg,Total $24.50
```

Expected OCR layout:

```text
data/
  ocr_images/
  ocr_train.csv
  ocr_val.csv
```

## Train the Caption Model

Fine-tune BLIP and save the checkpoint to `checkpoints\blip-captioner`:

```powershell
python train_blip.py `
  --train-csv data\train.csv `
  --val-csv data\val.csv `
  --image-root data `
  --output-dir checkpoints\blip-captioner `
  --epochs 3 `
  --batch-size 4
```

For a quick test run, use fewer samples:

```powershell
python train_blip.py --max-train-samples 100 --max-val-samples 20 --epochs 1
```

If the GPU runs out of memory, reduce `--batch-size`.

## Train the OCR Model

Fine-tune TrOCR and save the checkpoint to `checkpoints\trocr-ocr`:

```powershell
python train_ocr.py `
  --train-csv data\ocr_train.csv `
  --val-csv data\ocr_val.csv `
  --image-root data `
  --output-dir checkpoints\trocr-ocr `
  --epochs 3 `
  --batch-size 4
```

After training, the Streamlit app and API automatically use `checkpoints\trocr-ocr` when it exists.

## Evaluate

Evaluate captioning. This project has `data\val.csv`, so use it for a quick evaluation unless you create a separate `data\test.csv`.

```powershell
python evaluate_captions.py `
  --test-csv data\val.csv `
  --image-root data `
  --model checkpoints\blip-captioner `
  --output outputs\eval_predictions.csv
```

Build a small VQA CSV from captions:

```powershell
python build_vqa_from_captions.py `
  --captions data\captions.txt `
  --output data\vqa_test.csv `
  --max-rows 50
```

Evaluate VQA:

```powershell
python evaluate_vqa.py `
  --test-csv data\vqa_test.csv `
  --image-root data `
  --model Salesforce/blip-vqa-base `
  --output outputs\vqa_eval_predictions.csv
```

Use OCR context for text-reading questions:

```powershell
python evaluate_vqa.py `
  --test-csv data\vqa_test.csv `
  --image-root data `
  --model Salesforce/blip-vqa-base `
  --use-ocr-context
```

## Common Problems

If model loading fails, check whether the local checkpoint folder exists. If it does not, use a Hugging Face model name or train the model first.

If the first run is slow, wait for the model downloads to finish. Later runs are faster because the models are cached.

If imports fail, make sure the virtual environment is active and run:

```powershell
pip install -r requirements.txt
```

If training runs out of memory, reduce `--batch-size` or use CPU with smaller sample counts.

If the API live camera page does not open, confirm that `uvicorn` is still running and visit:

```text
http://127.0.0.1:8000/live
```

## Recommended Workflow

For normal use:

1. Set up the virtual environment.
2. Run `streamlit run app.py`.
3. Upload an image and click `Analyze Image`.
4. Ask image questions or search saved history.

For API use:

1. Run `uvicorn api:app --host 127.0.0.1 --port 8000`.
2. Open `/live` for camera analysis or call the API endpoints with `curl.exe`.

For model development:

1. Prepare CSV files in the required format.
2. Train with `train_blip.py` or `train_ocr.py`.
3. Evaluate with `evaluate_captions.py` or `evaluate_vqa.py`.
4. Use the trained checkpoint in the app, API, or CLI.
