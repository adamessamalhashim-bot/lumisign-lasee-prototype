from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class ReferenceStore:
    def __init__(self, directory: str | Path = "data/references"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(sign_name: str) -> str:
        cleaned = re.sub(r"[^\w\-\u0600-\u06FF]+", "_", sign_name.strip(), flags=re.UNICODE)
        if not cleaned:
            raise ValueError("Sign name cannot be empty")
        return cleaned

    def save(
        self,
        sign_name: str,
        landmarks: np.ndarray,
        handedness: str,
        source_url: str = "",
    ) -> Path:
        path = self.directory / f"{self._safe_name(sign_name)}.json"
        payload = {
            "sign_name": sign_name.strip(),
            "handedness": handedness,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "language": "Saudi Sign Language",
            "official_source_url": source_url,
            "landmarks": landmarks.tolist(),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def list_signs(self) -> list[str]:
        signs = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                signs.append(json.loads(path.read_text(encoding="utf-8"))["sign_name"])
            except (KeyError, json.JSONDecodeError):
                continue
        return signs

    def load(self, sign_name: str) -> tuple[np.ndarray, dict]:
        path = self.directory / f"{self._safe_name(sign_name)}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        return np.asarray(payload["landmarks"], dtype=np.float64), payload
