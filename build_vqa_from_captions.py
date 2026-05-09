from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd


COUNT_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
}

COLORS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}

ENTITY_TERMS = {
    "person": {"person", "people"},
    "man": {"man", "men"},
    "woman": {"woman", "women"},
    "boy": {"boy", "boys"},
    "girl": {"girl", "girls"},
    "child": {"child", "children", "kid", "kids"},
    "dog": {"dog", "dogs", "puppy", "puppies"},
    "cat": {"cat", "cats", "kitten", "kittens"},
    "horse": {"horse", "horses"},
    "bike": {"bike", "bikes", "bicycle", "bicycles"},
    "car": {"car", "cars"},
    "ball": {"ball", "balls"},
    "bench": {"bench", "benches"},
    "hat": {"hat", "hats", "cap", "caps"},
    "shirt": {"shirt", "shirts"},
    "dress": {"dress", "dresses"},
}

ANIMAL_TERMS = {"dog", "cat", "horse"}
PERSON_TERMS = {"person", "man", "woman", "boy", "girl", "child"}

ACTION_WORDS = {
    "climbing",
    "driving",
    "fighting",
    "jumping",
    "laying",
    "lying",
    "painting",
    "playing",
    "riding",
    "running",
    "sitting",
    "skateboarding",
    "sleeping",
    "standing",
    "swimming",
    "walking",
}

LOCATION_TERMS = {
    "beach",
    "bench",
    "building",
    "field",
    "grass",
    "mountain",
    "park",
    "pavement",
    "playground",
    "pool",
    "road",
    "snow",
    "stairs",
    "street",
    "surf",
    "track",
    "water",
}

TYPE_ORDER = ["count", "color", "object", "action", "scene"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small pseudo-VQA CSV from caption annotations.")
    parser.add_argument("--captions", type=Path, default=Path("data/captions.txt"))
    parser.add_argument("--output", type=Path, default=Path("data/vqa_test.csv"))
    parser.add_argument("--image-prefix", default="Images", help="Folder prefix added before captions.txt image names.")
    parser.add_argument("--max-rows", type=int, default=50)
    parser.add_argument("--max-per-image", type=int, default=3)
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def canonical_entity(token: str) -> str | None:
    for canonical, variants in ENTITY_TERMS.items():
        if token in variants:
            return canonical
    return None


def pluralize(entity: str) -> str:
    if entity == "man":
        return "men"
    if entity == "woman":
        return "women"
    if entity == "person":
        return "people"
    if entity == "child":
        return "children"
    if entity.endswith("s"):
        return entity
    return entity + "s"


def count_value(token: str) -> int | None:
    if token.isdigit():
        return int(token)
    return COUNT_WORDS.get(token)


def find_next_entity(tokens: list[str], start: int, window: int = 4) -> str | None:
    for word in tokens[start : start + window]:
        entity = canonical_entity(word)
        if entity is not None:
            return entity
    return None


def image_path_for(image_name: str, image_prefix: str) -> str:
    path = Path(str(image_name))
    if path.parent != Path("."):
        return str(path).replace("\\", "/")
    return str(Path(image_prefix) / path).replace("\\", "/")


def add_candidate(
    candidates: list[dict[str, str]],
    image_path: str,
    question: str,
    answer: str,
    question_type: str,
    source_caption: str,
) -> None:
    candidates.append(
        {
            "image_path": image_path,
            "question": question,
            "answer": answer,
            "question_type": question_type,
            "source_caption": source_caption,
        }
    )


def generate_count_questions(image_path: str, caption: str, tokens: list[str]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    article_counts: dict[str, int] = defaultdict(int)

    for index, token in enumerate(tokens[:-1]):
        value = count_value(token)
        if value is None or value < 1:
            continue

        entity = find_next_entity(tokens, index + 1)
        if entity is None:
            continue

        if token in {"a", "an"}:
            article_counts[entity] += 1
            continue

        add_candidate(
            candidates,
            image_path,
            f"How many {pluralize(entity)} are visible?",
            str(value),
            "count",
            caption,
        )

    for entity, value in article_counts.items():
        add_candidate(
            candidates,
            image_path,
            f"How many {pluralize(entity)} are visible?",
            str(value),
            "count",
            caption,
        )

    return candidates


def generate_color_questions(image_path: str, caption: str, tokens: list[str]) -> list[dict[str, str]]:
    by_entity: dict[str, set[str]] = defaultdict(set)
    for index, token in enumerate(tokens[:-1]):
        if token not in COLORS:
            continue
        for word in tokens[index + 1 : index + 4]:
            entity = canonical_entity(word)
            if entity is not None:
                by_entity[entity].add("gray" if token == "grey" else token)
                break

    candidates: list[dict[str, str]] = []
    for entity, colors in by_entity.items():
        if len(colors) != 1:
            continue
        color = next(iter(colors))
        add_candidate(
            candidates,
            image_path,
            f"What color is the {entity}?",
            color,
            "color",
            caption,
        )
    return candidates


def generate_object_questions(image_path: str, caption: str, tokens: list[str]) -> list[dict[str, str]]:
    entities = []
    for token in tokens:
        entity = canonical_entity(token)
        if entity is not None and entity not in entities:
            entities.append(entity)

    candidates: list[dict[str, str]] = []
    animals = [entity for entity in entities if entity in ANIMAL_TERMS]
    people = [entity for entity in entities if entity in PERSON_TERMS]
    other = [entity for entity in entities if entity not in ANIMAL_TERMS | PERSON_TERMS]

    if len(animals) == 1:
        add_candidate(candidates, image_path, "What animal is visible?", animals[0], "object", caption)
    if len(people) == 1:
        add_candidate(candidates, image_path, "What person is visible?", people[0], "object", caption)
    if len(other) == 1:
        add_candidate(candidates, image_path, "What object is visible?", other[0], "object", caption)

    return candidates


def generate_action_questions(image_path: str, caption: str, tokens: list[str]) -> list[dict[str, str]]:
    actions = [token for token in tokens if token in ACTION_WORDS]
    if not actions:
        return []

    subjects = []
    for token in tokens:
        entity = canonical_entity(token)
        if entity in PERSON_TERMS | ANIMAL_TERMS and entity not in subjects:
            subjects.append(entity)

    if len(subjects) != 1:
        return []

    action = "lying" if actions[0] == "laying" else actions[0]
    return [
        {
            "image_path": image_path,
            "question": f"What is the {subjects[0]} doing?",
            "answer": action,
            "question_type": "action",
            "source_caption": caption,
        }
    ]


def generate_scene_questions(image_path: str, caption: str, tokens: list[str]) -> list[dict[str, str]]:
    locations = [token for token in tokens if token in LOCATION_TERMS]
    if len(set(locations)) != 1:
        return []

    return [
        {
            "image_path": image_path,
            "question": "Where is the scene?",
            "answer": locations[0],
            "question_type": "scene",
            "source_caption": caption,
        }
    ]


def caption_candidates(image_path: str, caption: str) -> list[dict[str, str]]:
    tokens = tokenize(caption)
    candidates: list[dict[str, str]] = []
    candidates.extend(generate_count_questions(image_path, caption, tokens))
    candidates.extend(generate_color_questions(image_path, caption, tokens))
    candidates.extend(generate_object_questions(image_path, caption, tokens))
    candidates.extend(generate_action_questions(image_path, caption, tokens))
    candidates.extend(generate_scene_questions(image_path, caption, tokens))
    return candidates


def dedupe(candidates: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, str, str]] = set()
    rows: list[dict[str, str]] = []
    for candidate in candidates:
        key = (candidate["image_path"], candidate["question"], candidate["answer"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(candidate)
    return rows


def select_balanced(
    candidates: list[dict[str, str]],
    max_rows: int,
    max_per_image: int,
) -> list[dict[str, str]]:
    buckets: dict[str, list[dict[str, str]]] = {question_type: [] for question_type in TYPE_ORDER}
    for candidate in candidates:
        buckets[candidate["question_type"]].append(candidate)

    selected: list[dict[str, str]] = []
    per_image: dict[str, int] = defaultdict(int)

    while len(selected) < max_rows and any(buckets.values()):
        added_this_round = False
        for question_type in TYPE_ORDER:
            bucket = buckets[question_type]
            while bucket:
                candidate = bucket.pop(0)
                image_path = candidate["image_path"]
                if per_image[image_path] >= max_per_image:
                    continue
                selected.append(candidate)
                per_image[image_path] += 1
                added_this_round = True
                break
            if len(selected) >= max_rows:
                break
        if not added_this_round:
            break

    return selected


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.captions)
    if "caption" not in df.columns:
        raise ValueError(f"{args.captions} must contain a caption column.")

    image_column = "image_path" if "image_path" in df.columns else "image"
    if image_column not in df.columns:
        raise ValueError(f"{args.captions} must contain an image or image_path column.")

    all_candidates: list[dict[str, str]] = []
    for row in df.itertuples(index=False):
        image_name = getattr(row, image_column)
        caption = str(getattr(row, "caption"))
        image_path = image_path_for(str(image_name), args.image_prefix)
        all_candidates.extend(caption_candidates(image_path, caption))

    rows = select_balanced(dedupe(all_candidates), args.max_rows, args.max_per_image)
    if not rows:
        raise ValueError("No VQA rows could be generated from the captions.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)

    counts = pd.Series([row["question_type"] for row in rows]).value_counts().sort_index()
    print(f"Saved {len(rows)} rows to {args.output}")
    print(counts.to_string())


if __name__ == "__main__":
    main()
