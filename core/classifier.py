from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


@dataclass
class ModelReport:
    accuracy: float
    labels: list[str]
    confusion_matrix: list[list[int]]
    train_samples: int
    test_samples: int
    random_seed: int
    split_strategy: str
    test_participants: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


class SaudiSignClassifier:
    """A reproducible classifier trained on extracted 3D hand-landmark features."""

    def __init__(self, model_path: str | Path = "data/saudi_sign_classifier.joblib"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model: RandomForestClassifier | None = None

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        participant_ids: np.ndarray | None = None,
        seed: int = 42,
    ) -> ModelReport:
        labels, counts = np.unique(y, return_counts=True)
        if len(labels) < 2:
            raise ValueError("Collect samples for at least two different signs")
        if int(counts.min()) < 4:
            raise ValueError("Collect at least four samples for every sign before training")
        participant_ids = None if participant_ids is None else np.asarray(participant_ids)
        unique_participants = [] if participant_ids is None else sorted(np.unique(participant_ids).tolist())
        if len(unique_participants) >= 2:
            held_out = unique_participants[-1]
            test_mask = participant_ids == held_out
            x_train, x_test = x[~test_mask], x[test_mask]
            y_train, y_test = y[~test_mask], y[test_mask]
            if set(labels) - set(np.unique(y_train)) or set(labels) - set(np.unique(y_test)):
                raise ValueError("Every participant must provide samples for every selected sign")
            split_strategy = "participant_holdout"
            test_participants = [held_out]
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x,
                y,
                test_size=0.25,
                random_state=seed,
                stratify=y,
            )
            split_strategy = "stratified_sample_holdout"
            test_participants = []
        self.model = RandomForestClassifier(
            n_estimators=250,
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=1,
        )
        self.model.fit(x_train, y_train)
        predicted = self.model.predict(x_test)
        report = ModelReport(
            accuracy=round(float(accuracy_score(y_test, predicted)) * 100.0, 1),
            labels=labels.tolist(),
            confusion_matrix=confusion_matrix(y_test, predicted, labels=labels).tolist(),
            train_samples=len(x_train),
            test_samples=len(x_test),
            random_seed=seed,
            split_strategy=split_strategy,
            test_participants=test_participants,
        )
        joblib.dump({"model": self.model, "report": report.to_dict()}, self.model_path)
        self.model_path.with_suffix(".report.json").write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

    def load(self) -> bool:
        if not self.model_path.exists():
            return False
        self.model = joblib.load(self.model_path)["model"]
        return True

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        details = self.predict_details(features)
        return details["predicted_sign"], details["confidence"]

    def predict_details(self, features: np.ndarray, confidence_threshold: float = 65.0) -> dict:
        """Return a transparent prediction with uncertainty and a safety gate."""
        if self.model is None and not self.load():
            raise RuntimeError("Train the classifier first")
        probabilities = self.model.predict_proba(np.asarray(features).reshape(1, -1))[0]
        index = int(np.argmax(probabilities))
        ranked = sorted(
            ((str(label), round(float(probability) * 100.0, 1)) for label, probability in zip(self.model.classes_, probabilities)),
            key=lambda item: item[1],
            reverse=True,
        )
        confidence = ranked[0][1]
        runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = round(confidence - runner_up, 1)
        accepted = confidence >= confidence_threshold
        return {
            "predicted_sign": str(self.model.classes_[index]),
            "confidence": confidence,
            "runner_up_confidence": runner_up,
            "confidence_margin": margin,
            "confidence_threshold": float(confidence_threshold),
            "decision_status": "accepted" if accepted else "needs_human_review",
            "class_probabilities": dict(ranked),
        }
