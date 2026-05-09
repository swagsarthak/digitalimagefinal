from __future__ import annotations

import argparse
from pathlib import Path


class CaptionDataset:
    def __init__(
        self,
        csv_path: Path,
        image_root: Path,
        processor: BlipProcessor,
        max_text_length: int,
        max_samples: int | None = None,
    ) -> None:
        import pandas as pd

        self.rows = pd.read_csv(csv_path)
        if max_samples is not None:
            self.rows = self.rows.head(max_samples)
        self.image_root = image_root
        self.processor = processor
        self.max_text_length = max_text_length

        required_columns = {"image_path", "caption"}
        missing = required_columns.difference(self.rows.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_image_path(self, image_path: str) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        return self.image_root / path

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        from PIL import Image

        row = self.rows.iloc[idx]
        image = Image.open(self._resolve_image_path(str(row["image_path"]))).convert("RGB")
        caption = str(row["caption"])

        encoding = self.processor(
            images=image,
            text=caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        labels = item["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        item["labels"] = labels
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BLIP for image captioning.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/val.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data"))
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/blip-captioner"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-text-length", type=int, default=64)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    missing = []
    if not args.train_csv.is_file():
        missing.append(f"--train-csv {args.train_csv}")
    if not args.val_csv.is_file():
        missing.append(f"--val-csv {args.val_csv}")
    if not args.image_root.is_dir():
        missing.append(f"--image-root {args.image_root}")

    if missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            "Missing required training inputs:\n"
            f"  {joined}\n\n"
            "Create data/train.csv and data/val.csv with columns "
            "'image_path,caption', and place the referenced images under data/."
        )


def main() -> None:
    args = parse_args()
    validate_paths(args)

    import torch
    from transformers import (
        BlipForConditionalGeneration,
        BlipProcessor,
        Trainer,
        TrainingArguments,
    )

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    print(f"Training device: {device_name}")
    print(f"CUDA available: {cuda_available}")

    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model)
    if cuda_available:
        model = model.to("cuda")

    train_dataset = CaptionDataset(
        args.train_csv,
        args.image_root,
        processor,
        args.max_text_length,
        args.max_train_samples,
    )
    eval_dataset = CaptionDataset(
        args.val_csv,
        args.image_root,
        processor,
        args.max_text_length,
        args.max_val_samples,
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        remove_unused_columns=False,
        dataloader_pin_memory=cuda_available,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to {args.output_dir}")


if __name__ == "__main__":
    main()
