from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def normalized_features(points: np.ndarray) -> np.ndarray:
    """Convert 21 x/y/z landmarks into a translation- and scale-invariant vector."""
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError("Expected 21 three-dimensional hand landmarks")
    centered = points - points[0]
    palm_scale = np.linalg.norm(centered[9])
    if palm_scale < 1e-8:
        raise ValueError("Invalid hand geometry: palm scale is zero")
    return (centered / palm_scale).reshape(-1)


class SaudiSignDataset:
    """Stores traceable Saudi Sign Language samples as JSONL records."""

    def __init__(self, path: str | Path = "data/saudi_sign_samples.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        sign_name: str,
        landmarks: np.ndarray,
        participant_id: str,
        source_url: str,
        handedness: str,
        detection_confidence: float,
    ) -> dict:
        if not sign_name.strip() or not participant_id.strip() or not source_url.strip():
            raise ValueError("Sign name, participant code and official source URL are required")
        record = {
            "sample_id": str(uuid.uuid4()),
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "language": "Saudi Sign Language",
            "sign_name": sign_name.strip(),
            "participant_id": participant_id.strip(),
            "official_source_url": source_url.strip(),
            "handedness": handedness,
            "detection_confidence": round(float(detection_confidence) * 100.0, 1),
            "landmarks": np.asarray(landmarks, dtype=float).tolist(),
            "features": normalized_features(landmarks).tolist(),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def training_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        rows = self.records()
        if not rows:
            return np.empty((0, 63)), np.empty((0,), dtype=str)
        return np.asarray([row["features"] for row in rows]), np.asarray([row["sign_name"] for row in rows])

    def training_bundle(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rows = self.records()
        if not rows:
            return np.empty((0, 63)), np.empty((0,), dtype=str), np.empty((0,), dtype=str)
        return (
            np.asarray([row["features"] for row in rows]),
            np.asarray([row["sign_name"] for row in rows]),
            np.asarray([row["participant_id"] for row in rows]),
        )

    def summary(self) -> dict:
        rows = self.records()
        counts: dict[str, int] = {}
        participants: set[str] = set()
        for row in rows:
            counts[row["sign_name"]] = counts.get(row["sign_name"], 0) + 1
            participants.add(row["participant_id"])
        return {
            "total_samples": len(rows),
            "signs": counts,
            "participant_count": len(participants),
            "language": "Saudi Sign Language",
        }
