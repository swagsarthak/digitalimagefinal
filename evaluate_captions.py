from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import nltk
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from tqdm import tqdm

from image_captioning import CaptionConfig, ImageCaptioner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate image captions with BLEU and overlap metrics.")
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--image-root", type=Path, default=Path("data"))
    parser.add_argument("--model", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--output", type=Path, default=Path("outputs/eval_predictions.csv"))
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--num-beams", type=int, default=5)
    return parser.parse_args()


def resolve_image_path(image_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    return image_root / path


def tokenize(text: str) -> list[str]:
    return nltk.wordpunct_tokenize(text.lower())


def bleu_scores(
    references: list[list[list[str]]],
    hypotheses: list[list[str]],
) -> dict[str, float]:
    smoothing = SmoothingFunction().method4
    return {
        "BLEU-1": corpus_bleu(
            references,
            hypotheses,
            weights=(1.0, 0, 0, 0),
            smoothing_function=smoothing,
        ),
        "BLEU-2": corpus_bleu(
            references,
            hypotheses,
            weights=(0.5, 0.5, 0, 0),
            smoothing_function=smoothing,
        ),
        "BLEU-3": corpus_bleu(
            references,
            hypotheses,
            weights=(1 / 3, 1 / 3, 1 / 3, 0),
            smoothing_function=smoothing,
        ),
        "BLEU-4": corpus_bleu(
            references,
            hypotheses,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothing,
        ),
    }


def token_overlap_scores(reference: list[str], hypothesis: list[str]) -> tuple[float, float, float]:
    reference_counts = Counter(reference)
    hypothesis_counts = Counter(hypothesis)
    overlap = sum((reference_counts & hypothesis_counts).values())

    precision = overlap / len(hypothesis) if hypothesis else 0.0
    recall = overlap / len(reference) if reference else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_score(reference: list[str], hypothesis: list[str]) -> float:
    if not reference or not hypothesis:
        return 0.0

    lcs = lcs_length(reference, hypothesis)
    precision = lcs / len(hypothesis)
    recall = lcs / len(reference)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def best_reference_overlap(
    references: list[list[str]],
    hypothesis: list[str],
) -> tuple[float, float, float, float]:
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_rouge_l = 0.0

    for reference in references:
        precision, recall, f1 = token_overlap_scores(reference, hypothesis)
        rouge_l = rouge_l_score(reference, hypothesis)
        if f1 > best_f1:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
        best_rouge_l = max(best_rouge_l, rouge_l)

    return best_precision, best_recall, best_f1, best_rouge_l


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def text_overlap_scores(
    references: list[list[list[str]]],
    hypotheses: list[list[str]],
) -> dict[str, float]:
    precisions = []
    recalls = []
    f1_scores = []
    rouge_l_scores = []

    for image_references, hypothesis in zip(references, hypotheses, strict=False):
        precision, recall, f1, rouge_l = best_reference_overlap(image_references, hypothesis)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        rouge_l_scores.append(rouge_l)

    return {
        "Token Precision": mean(precisions),
        "Token Recall": mean(recalls),
        "Token F1": mean(f1_scores),
        "ROUGE-L": mean(rouge_l_scores),
    }


def multi_reference_inputs(predictions: list[dict[str, str]]) -> tuple[list[list[list[str]]], list[list[str]]]:
    rows = pd.DataFrame(predictions)
    references: list[list[list[str]]] = []
    hypotheses: list[list[str]] = []

    for _, group in rows.groupby("image_path", sort=False):
        image_references = [tokenize(caption) for caption in group["reference_caption"].dropna().astype(str)]
        generated = group["generated_caption"].dropna().astype(str)
        if not image_references or generated.empty:
            continue
        references.append(image_references)
        hypotheses.append(tokenize(generated.iloc[0]))

    return references, hypotheses


def print_scores(title: str, scores: dict[str, float]) -> None:
    print(title)
    for metric, score in scores.items():
        print(f"{metric}: {score:.4f}")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.test_csv)
    required_columns = {"image_path", "caption"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{args.test_csv} is missing required columns: {sorted(missing)}")

    captioner = ImageCaptioner(
        CaptionConfig(
            model_name=args.model,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
        )
    )

    predictions = []
    references = []
    hypotheses = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Evaluating"):
        image_path = resolve_image_path(args.image_root, str(row.image_path))
        predicted = captioner.caption(image_path)
        reference = str(row.caption)

        predictions.append(
            {
                "image_path": str(row.image_path),
                "reference_caption": reference,
                "generated_caption": predicted,
            }
        )
        references.append([tokenize(reference)])
        hypotheses.append(tokenize(predicted))

    row_scores = bleu_scores(references, hypotheses)
    row_overlap_scores = text_overlap_scores(references, hypotheses)
    multi_reference_references, multi_reference_hypotheses = multi_reference_inputs(predictions)
    multi_reference_scores = bleu_scores(multi_reference_references, multi_reference_hypotheses)
    multi_reference_overlap_scores = text_overlap_scores(
        multi_reference_references,
        multi_reference_hypotheses,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(args.output, index=False)

    print_scores("Single-reference row-wise BLEU:", row_scores)
    print_scores("Single-reference row-wise overlap metrics:", row_overlap_scores)
    print()
    print(f"Multi-reference image-wise examples: {len(multi_reference_hypotheses)}")
    print_scores("Multi-reference image-wise BLEU:", multi_reference_scores)
    print_scores("Multi-reference image-wise overlap metrics:", multi_reference_overlap_scores)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
