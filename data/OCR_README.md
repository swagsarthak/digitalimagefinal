# OCR Dataset Format

Use two CSV files for OCR training:

```csv
image_path,text
ocr_images/sign_001.jpg,Green Valley School
ocr_images/receipt_001.jpg,Total $24.50
```

Expected layout:

```text
data/
  ocr_train.csv
  ocr_val.csv
  ocr_images/
    sign_001.jpg
    receipt_001.jpg
```

The `image_path` column can be relative to `data/` or an absolute path. The `text` column must contain the exact text visible in the image.

For best results, use cropped images where the text is large and readable. Full scene images with small text are harder for OCR training.
