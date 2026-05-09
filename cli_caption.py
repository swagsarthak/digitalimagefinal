from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from image_captioning import CaptionConfig, ImageCaptioner


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate captions for one image or a folder of images.")
    parser.add_argument("--input", required=True, type=Path, help="Image file or directory.")
    parser.add_argument("--output", type=Path, default=Path("outputs/captions.csv"), help="CSV output path.")
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-base", help="Hugging Face model name/path.")
    parser.add_argument("--prompt", default=None, help="Optional conditional prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--device", default=None, help="Use cuda, cpu, or leave empty for auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_paths = collect_images(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.input}")

    captioner = ImageCaptioner(
        CaptionConfig(
            model_name=args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
    )

    rows = []
    for image_path in tqdm(image_paths, desc="Captioning"):
        rows.append({"image_path": str(image_path), "caption": captioner.caption(image_path, prompt=args.prompt)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Saved {len(rows)} captions to {args.output}")


if __name__ == "__main__":
    main()
