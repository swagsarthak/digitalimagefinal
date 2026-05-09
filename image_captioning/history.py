from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class StoredImage:
    id: str
    filename: str
    image_path: str
    created_at: str
    analysis: dict[str, Any]


class ImageHistoryStore:
    """JSON-backed image analysis history for local demos."""

    def __init__(self, root: str | Path = "outputs/history") -> None:
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.history_path = self.root / "history.json"
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def save(self, image: Image.Image, filename: str, analysis: dict[str, Any]) -> StoredImage:
        image_id = self._make_id(filename)
        safe_name = Path(filename or "uploaded_image.png").name
        image_path = self.image_dir / f"{image_id}_{safe_name}"
        image.convert("RGB").save(image_path)

        item = StoredImage(
            id=image_id,
            filename=safe_name,
            image_path=str(image_path),
            created_at=datetime.now(timezone.utc).isoformat(),
            analysis=analysis,
        )
        records = self.list()
        records.insert(0, item)
        self._write(records)
        return item

    def list(self) -> list[StoredImage]:
        if not self.history_path.exists():
            return []

        with self.history_path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return [StoredImage(**record) for record in data]

    def search(self, query: str) -> list[StoredImage]:
        cleaned_query = query.strip().lower()
        if not cleaned_query:
            return self.list()

        matches: list[StoredImage] = []
        for record in self.list():
            searchable = json.dumps(record.__dict__, ensure_ascii=False).lower()
            if cleaned_query in searchable:
                matches.append(record)
        return matches

    def get(self, image_id: str) -> StoredImage | None:
        for record in self.list():
            if record.id == image_id:
                return record
        return None

    def delete(self, image_id: str) -> bool:
        records = self.list()
        remaining = [record for record in records if record.id != image_id]
        if len(remaining) == len(records):
            return False

        for record in records:
            if record.id == image_id:
                image_path = Path(record.image_path)
                if image_path.exists():
                    image_path.unlink()
                break

        self._write(remaining)
        return True

    def clear(self) -> None:
        if self.root.exists():
            shutil.rmtree(self.root)
        self.image_dir.mkdir(parents=True, exist_ok=True)

    def _write(self, records: list[StoredImage]) -> None:
        with self.history_path.open("w", encoding="utf-8") as file:
            json.dump([record.__dict__ for record in records], file, indent=2)

    @staticmethod
    def _make_id(filename: str) -> str:
        seed = f"{filename}-{datetime.now(timezone.utc).isoformat()}".encode("utf-8")
        return hashlib.sha1(seed).hexdigest()[:12]
