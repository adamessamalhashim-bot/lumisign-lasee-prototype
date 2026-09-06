from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
from supabase import Client, create_client

from core.dataset import normalized_features


class CloudStore:
    """Persistent LumiSign storage backed by private Supabase tables."""

    def __init__(self, url: str, secret_key: str):
        if not url.strip() or not secret_key.strip():
            raise ValueError("Supabase URL and secret key are required")
        self.client: Client = create_client(url.strip().rstrip("/"), secret_key.strip())

    def health_check(self) -> bool:
        self.client.table("signs").select("id").limit(1).execute()
        return True

    def list_signs(self) -> list[dict[str, Any]]:
        response = (
            self.client.table("signs")
            .select("id,sign_name,official_source_url,reference_landmarks,active,created_at,updated_at")
            .eq("active", True)
            .order("sign_name")
            .execute()
        )
        return list(response.data or [])

    def reference_bank(self) -> dict[str, np.ndarray]:
        bank: dict[str, np.ndarray] = {}
        for row in self.list_signs():
            landmarks = row.get("reference_landmarks")
            if not landmarks:
                continue
            points = np.asarray(landmarks, dtype=np.float64)
            if points.shape == (21, 3):
                bank[str(row["sign_name"])] = points
        return bank

    def save_reference(
        self,
        sign_name: str,
        source_url: str,
        landmarks: np.ndarray,
        created_by: str,
    ) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "sign_name": sign_name.strip(),
            "official_source_url": source_url.strip(),
            "reference_landmarks": np.asarray(landmarks, dtype=float).tolist(),
            "active": True,
            "created_by": created_by,
            "updated_at": now,
        }
        response = (
            self.client.table("signs")
            .upsert(payload, on_conflict="sign_name")
            .execute()
        )
        return dict((response.data or [payload])[0])

    def _sign_id(self, sign_name: str, source_url: str, created_by: str) -> str:
        response = (
            self.client.table("signs")
            .select("id")
            .eq("sign_name", sign_name.strip())
            .limit(1)
            .execute()
        )
        if response.data:
            return str(response.data[0]["id"])
        payload = {
            "sign_name": sign_name.strip(),
            "official_source_url": source_url.strip(),
            "active": True,
            "created_by": created_by,
        }
        created = self.client.table("signs").insert(payload).execute()
        return str(created.data[0]["id"])

    def add_training_sample(
        self,
        sign_name: str,
        participant_id: str,
        source_url: str,
        landmarks: np.ndarray,
        detection_confidence: float,
        created_by: str,
    ) -> dict[str, Any]:
        if not sign_name.strip() or not participant_id.strip() or not source_url.strip():
            raise ValueError("اسم الإشارة ورمز المشارك ورابط المصدر مطلوبة")
        sign_id = self._sign_id(sign_name, source_url, created_by)
        payload = {
            "sign_id": sign_id,
            "participant_id": participant_id.strip(),
            "landmarks": np.asarray(landmarks, dtype=float).tolist(),
            "features": normalized_features(landmarks).tolist(),
            "detection_confidence": round(float(detection_confidence) * 100.0, 1),
            "created_by": created_by,
        }
        response = self.client.table("training_samples").insert(payload).execute()
        return dict((response.data or [payload])[0])

    def training_records(self) -> list[dict[str, Any]]:
        signs = {str(row["id"]): str(row["sign_name"]) for row in self.list_signs()}
        response = (
            self.client.table("training_samples")
            .select("id,sign_id,participant_id,landmarks,features,detection_confidence,created_at")
            .order("created_at")
            .execute()
        )
        records: list[dict[str, Any]] = []
        for row in response.data or []:
            sign_name = signs.get(str(row.get("sign_id")))
            if not sign_name:
                continue
            records.append(
                {
                    "sample_id": str(row["id"]),
                    "sign_name": sign_name,
                    "participant_id": str(row["participant_id"]),
                    "landmarks": row["landmarks"],
                    "features": row["features"],
                    "detection_confidence": float(row.get("detection_confidence") or 0.0),
                    "captured_at": row.get("created_at"),
                }
            )
        return records

    def dataset_summary(self) -> dict[str, Any]:
        rows = self.training_records()
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

    def save_attempt(self, payload: dict[str, Any]) -> None:
        self.client.table("user_attempts").insert(payload).execute()

    def recent_attempts(self, limit: int = 30) -> list[dict[str, Any]]:
        response = (
            self.client.table("user_attempts")
            .select(
                "user_email,target_sign,predicted_sign,overall_score,"
                "classifier_confidence,created_at"
            )
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return list(response.data or [])

    def save_model_version(self, version_name: str, accuracy: float, metrics: dict, created_by: str) -> None:
        self.client.table("model_versions").update({"active": False}).eq("active", True).execute()
        self.client.table("model_versions").upsert(
            {
                "version_name": version_name,
                "accuracy": float(accuracy),
                "metrics": metrics,
                "active": True,
                "created_by": created_by,
            },
            on_conflict="version_name",
        ).execute()

    def active_model(self) -> dict[str, Any] | None:
        response = (
            self.client.table("model_versions")
            .select("version_name,accuracy,metrics,created_at")
            .eq("active", True)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
        return dict(response.data[0]) if response.data else None


def records_to_training_bundle(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not records:
        return np.empty((0, 63)), np.empty((0,), dtype=str), np.empty((0,), dtype=str)
    return (
        np.asarray([row["features"] for row in records], dtype=np.float64),
        np.asarray([row["sign_name"] for row in records], dtype=str),
        np.asarray([row["participant_id"] for row in records], dtype=str),
    )
