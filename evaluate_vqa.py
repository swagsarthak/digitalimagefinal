from __future__ import annotations

import argparse
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from image_captioning import ImageOCRReader, ImageQuestionAnswerer, OCRConfig, VQAConfig


ARTICLES = {"a", "an", "the"}
NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate visual question answering accuracy and answer overlap.")
    parser.add_argument("--test-csv", type=Path, default=Path("data/vqa_test.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data"))
    parser.add_argument("--model", default="Salesforce/blip-vqa-base", help="Hugging Face model name/path.")
    parser.add_argument("--output", type=Path, default=Path("outputs/vqa_eval_predictions.csv"))
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--device", default=None, help="Use cuda, cpu, or leave empty for auto.")
    parser.add_argument(
        "--use-ocr-context",
        action="store_true",
        help="Use TrOCR text for questions that ask what text is visible.",
    )
    parser.add_argument(
        "--ocr-model",
        default="checkpoints/trocr-ocr"
        if Path("checkpoints/trocr-ocr").exists()
        else "microsoft/trocr-base-printed",
        help="OCR model used only with --use-ocr-context.",
    )
    parser.add_argument("--ocr-max-new-tokens", type=int, default=64)
    return parser.parse_args()


def resolve_image_path(image_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return image_root / path


def normalize_answer(text: object) -> str:
    cleaned = str(text).lower().strip()
    cleaned = cleaned.translate(str.maketrans({char: " " for char in string.punctuation if char != "|"}))
    tokens = []
    for token in cleaned.split():
        if token in ARTICLES:
            continue
        tokens.append(NUMBER_WORDS.get(token, token))
    return " ".join(tokens)


def reference_answers(answer: object) -> list[str]:
    return [part.strip() for part in str(answer).split("|") if part.strip()]


def exact_match(prediction: str, references: list[str]) -> bool:
    predicted = normalize_answer(prediction)
    return any(predicted == normalize_answer(reference) for reference in references)


def number_tokens(answer: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", answer))


def relaxed_match(prediction: str, references: list[str]) -> bool:
    if exact_match(prediction, references):
        return True

    predicted = normalize_answer(prediction)
    predicted_tokens = set(predicted.split())
    predicted_numbers = number_tokens(predicted)

    for reference in references:
        expected = normalize_answer(reference)
        expected_tokens = set(expected.split())
        expected_numbers = number_tokens(expected)

        if not predicted_tokens or not expected_tokens:
            continue
        if expected_numbers and expected_numbers.issubset(predicted_numbers):
            return True
        if len(expected_tokens) <= 3 and expected_tokens.issubset(predicted_tokens):
            return True
        if len(predicted_tokens) <= 3 and predicted_tokens.issubset(expected_tokens):
            return True

    return False


def token_overlap_scores(reference: str, prediction: str) -> tuple[float, float, float]:
    reference_tokens = normalize_answer(reference).split()
    prediction_tokens = normalize_answer(prediction).split()
    reference_counts = Counter(reference_tokens)
    prediction_counts = Counter(prediction_tokens)
    overlap = sum((reference_counts & prediction_counts).values())

    precision = overlap / len(prediction_tokens) if prediction_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def best_token_overlap(prediction: str, references: list[str]) -> tuple[float, float, float]:
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0

    for reference in references:
        precision, recall, f1 = token_overlap_scores(reference, prediction)
        if f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_f1 = f1

    return best_precision, best_recall, best_f1


def accuracy(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def print_metrics(rows: list[dict[str, object]], has_question_type: bool) -> None:
    exact_values = [bool(row["exact_match"]) for row in rows]
    relaxed_values = [bool(row["relaxed_match"]) for row in rows]
    token_precision_values = [float(row["token_precision"]) for row in rows]
    token_recall_values = [float(row["token_recall"]) for row in rows]
    token_f1_values = [float(row["token_f1"]) for row in rows]

    print(f"Questions: {len(rows)}")
    print(f"Exact accuracy: {accuracy(exact_values):.4f}")
    print(f"Relaxed accuracy: {accuracy(relaxed_values):.4f}")
    print(f"Token precision: {mean(token_precision_values):.4f}")
    print(f"Token recall: {mean(token_recall_values):.4f}")
    print(f"Token F1: {mean(token_f1_values):.4f}")

    if not has_question_type:
        return

    print("\nPer question type:")
    by_type = sorted({str(row["question_type"]) for row in rows})
    macro_exact = []
    macro_relaxed = []
    macro_token_f1 = []
    for question_type in by_type:
        typed_rows = [row for row in rows if str(row["question_type"]) == question_type]
        typed_exact = [bool(row["exact_match"]) for row in typed_rows]
        typed_relaxed = [bool(row["relaxed_match"]) for row in typed_rows]
        typed_token_f1 = [float(row["token_f1"]) for row in typed_rows]
        exact = accuracy(typed_exact)
        relaxed = accuracy(typed_relaxed)
        token_f1 = mean(typed_token_f1)
        macro_exact.append(exact)
        macro_relaxed.append(relaxed)
        macro_token_f1.append(token_f1)
        print(
            f"{question_type}: "
            f"n={len(typed_rows)}, "
            f"exact={exact:.4f}, "
            f"relaxed={relaxed:.4f}, "
            f"token_f1={token_f1:.4f}"
        )

    print("\nMacro averages:")
    print(f"Macro exact accuracy: {mean(macro_exact):.4f}")
    print(f"Macro relaxed accuracy: {mean(macro_relaxed):.4f}")
    print(f"Macro token F1: {mean(macro_token_f1):.4f}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.test_csv)
    required_columns = {"image_path", "question", "answer"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{args.test_csv} is missing required columns: {sorted(missing)}")

    has_question_type = "question_type" in df.columns
    question_answerer = ImageQuestionAnswerer(
        VQAConfig(
            model_name=args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
        )
    )
    ocr_reader = None
    if args.use_ocr_context:
        ocr_reader = ImageOCRReader(
            OCRConfig(
                model_name=args.ocr_model,
                device=args.device,
                max_new_tokens=args.ocr_max_new_tokens,
            )
        )

    rows: list[dict[str, object]] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Evaluating VQA"):
        image_path_text = str(row.image_path)
        image_path = resolve_image_path(args.image_root, image_path_text)
        question = str(row.question)
        references = reference_answers(row.answer)

        extracted_text = None
        if ocr_reader is not None and question_answerer.is_text_question(question):
            extracted_text = ocr_reader.extract_text(question_answerer.load_image(image_path))

        prediction = question_answerer.answer_with_context(
            image_path,
            question,
            extracted_text=extracted_text,
        )
        exact = exact_match(prediction, references)
        relaxed = relaxed_match(prediction, references)
        token_precision, token_recall, token_f1 = best_token_overlap(prediction, references)

        result = {
            "image_path": image_path_text,
            "question": question,
            "reference_answer": str(row.answer),
            "predicted_answer": prediction,
            "exact_match": exact,
            "relaxed_match": relaxed,
            "token_precision": token_precision,
            "token_recall": token_recall,
            "token_f1": token_f1,
        }
        if has_question_type:
            result["question_type"] = str(row.question_type)
        rows.append(result)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)

    print_metrics(rows, has_question_type)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
