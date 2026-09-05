from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2


FINGER_CHAINS = {
    "Thumb": (1, 2, 3, 4),
    "Index": (5, 6, 7, 8),
    "Middle": (9, 10, 11, 12),
    "Ring": (13, 14, 15, 16),
    "Pinky": (17, 18, 19, 20),
}


@dataclass
class EvaluationResult:
    overall_score: float
    hand_shape_score: float
    finger_angles_score: float
    palm_orientation_score: float
    hand_position_score: float
    confidence: float
    feedback: list[str]
    worst_feature: str
    raw_errors: dict[str, float]
    component_weights: dict[str, float]
    weighted_contributions: dict[str, float]

    def to_dict(self) -> dict:
        return asdict(self)


class HandAnalyzer:
    """Extracts MediaPipe landmarks and compares them with a saved reference.

    The evaluator is intentionally explainable: every component score is derived
    from a measurable geometric error, not a fabricated percentage.
    """

    def __init__(self, min_detection_confidence: float = 0.6):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
        )
        self._drawer = mp.solutions.drawing_utils
        self._connections = mp.solutions.hands.HAND_CONNECTIONS

    def close(self) -> None:
        self._hands.close()

    def extract(self, bgr_image: np.ndarray) -> tuple[np.ndarray | None, str | None, float]:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None, None, 0.0

        hand = result.multi_hand_landmarks[0]
        landmarks = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float64)
        handedness = result.multi_handedness[0].classification[0]
        return landmarks, handedness.label, float(handedness.score)

    def draw(self, bgr_image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        annotated = bgr_image.copy()
        proto = landmark_pb2.NormalizedLandmarkList()
        for x, y, z in landmarks:
            proto.landmark.add(x=float(x), y=float(y), z=float(z))
        self._drawer.draw_landmarks(
            annotated,
            proto,
            self._connections,
            self._drawer.DrawingSpec(color=(38, 224, 183), thickness=3, circle_radius=3),
            self._drawer.DrawingSpec(color=(255, 255, 255), thickness=2),
        )
        return annotated

    @staticmethod
    def normalize_shape(points: np.ndarray) -> np.ndarray:
        """Translation/scale-normalize while preserving pose geometry."""
        centered = points - points[0]
        palm_scale = np.linalg.norm(centered[9])
        if palm_scale < 1e-8:
            raise ValueError("Invalid hand geometry: palm scale is zero")
        return centered / palm_scale

    @staticmethod
    def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        ba, bc = a - b, c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom < 1e-8:
            return 0.0
        cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))

    @classmethod
    def finger_angles(cls, points: np.ndarray) -> dict[str, list[float]]:
        angles: dict[str, list[float]] = {}
        wrist = points[0]
        for name, (mcp, pip, dip, tip) in FINGER_CHAINS.items():
            angles[name] = [
                cls._angle(wrist, points[mcp], points[pip]),
                cls._angle(points[mcp], points[pip], points[dip]),
                cls._angle(points[pip], points[dip], points[tip]),
            ]
        return angles

    @staticmethod
    def palm_normal(points: np.ndarray) -> np.ndarray:
        a = points[5] - points[0]
        b = points[17] - points[0]
        normal = np.cross(a, b)
        norm = np.linalg.norm(normal)
        return normal / norm if norm > 1e-8 else np.zeros(3)

    @staticmethod
    def _score(error: float, tolerance: float) -> float:
        return round(float(100.0 * np.exp(-((error / tolerance) ** 2))), 1)

    def evaluate(self, attempt: np.ndarray, reference: np.ndarray, detection_confidence: float) -> EvaluationResult:
        a_norm = self.normalize_shape(attempt)
        r_norm = self.normalize_shape(reference)

        point_errors = np.linalg.norm(a_norm - r_norm, axis=1)
        shape_error = float(np.mean(point_errors))
        shape_score = self._score(shape_error, 0.42)

        a_angles = self.finger_angles(a_norm)
        r_angles = self.finger_angles(r_norm)
        finger_errors = {
            name: float(np.mean(np.abs(np.array(a_angles[name]) - np.array(r_angles[name]))))
            for name in FINGER_CHAINS
        }
        angle_error = float(np.mean(list(finger_errors.values())))
        angle_score = self._score(angle_error, 32.0)

        dot = float(np.clip(np.dot(self.palm_normal(a_norm), self.palm_normal(r_norm)), -1.0, 1.0))
        palm_error = float(np.degrees(np.arccos(dot)))
        palm_score = self._score(palm_error, 48.0)

        position_error = float(np.linalg.norm(attempt[0, :2] - reference[0, :2]))
        position_score = self._score(position_error, 0.28)

        component_scores = {
            "Hand shape": shape_score,
            "Finger angles": angle_score,
            "Palm orientation": palm_score,
            "Hand position": position_score,
        }
        component_weights = {
            "Hand shape": 0.38,
            "Finger angles": 0.32,
            "Palm orientation": 0.20,
            "Hand position": 0.10,
        }
        weighted_contributions = {
            name: round(component_scores[name] * weight, 2)
            for name, weight in component_weights.items()
        }
        overall = round(sum(weighted_contributions.values()), 1)
        worst_feature = min(component_scores, key=component_scores.get)
        worst_finger = max(finger_errors, key=finger_errors.get)
        feedback = self._feedback(component_scores, worst_finger, finger_errors[worst_finger], overall)

        return EvaluationResult(
            overall_score=overall,
            hand_shape_score=shape_score,
            finger_angles_score=angle_score,
            palm_orientation_score=palm_score,
            hand_position_score=position_score,
            confidence=round(detection_confidence * 100.0, 1),
            feedback=feedback,
            worst_feature=worst_feature,
            raw_errors={
                "Hand shape": round(shape_error, 4),
                "Finger angles": round(angle_error, 2),
                "Palm orientation": round(palm_error, 2),
                "Hand position": round(position_error, 4),
            },
            component_weights=component_weights,
            weighted_contributions=weighted_contributions,
        )

    @staticmethod
    def _feedback(scores: dict[str, float], worst_finger: str, finger_error: float, overall: float) -> list[str]:
        messages: list[str] = []
        if scores["Palm orientation"] < 75:
            messages.append("دوّر راحة اليد لتطابق اتجاه المرجع.")
        if scores["Finger angles"] < 75:
            finger_ar = {"Thumb": "الإبهام", "Index": "السبابة", "Middle": "الوسطى", "Ring": "البنصر", "Pinky": "الخنصر"}
            messages.append(f"عدّل إصبع {finger_ar[worst_finger]}؛ متوسط خطأ الزاوية {finger_error:.1f}°.")
        if scores["Hand position"] < 75:
            messages.append("حرّك يدك لتقترب من موضع المرجع داخل إطار الكاميرا.")
        if scores["Hand shape"] < 75:
            messages.append("طابق تباعد الأصابع وامتدادها مع المرجع بصورة أدق.")
        if overall >= 85:
            messages = ["تطابق ممتاز. ثبّت الإشارة لإكمال الدرس."]
        return messages or ["محاولة جيدة. أجرِ تعديلًا بسيطًا ثم أعد المحاولة."]


def ghost_overlay(image: np.ndarray, reference: np.ndarray, alpha: float = 0.38) -> np.ndarray:
    """Draw a transparent reference skeleton aligned to the attempt frame."""
    overlay = image.copy()
    h, w = image.shape[:2]
    points = np.column_stack((reference[:, 0] * w, reference[:, 1] * h)).astype(int)
    for start, end in mp.solutions.hands.HAND_CONNECTIONS:
        cv2.line(overlay, tuple(points[start]), tuple(points[end]), (255, 80, 200), 4)
    for x, y in points:
        cv2.circle(overlay, (int(x), int(y)), 5, (255, 80, 200), -1)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
