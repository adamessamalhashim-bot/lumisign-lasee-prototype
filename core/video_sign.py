from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from core.dataset import normalized_features
from core.hand_analyzer import HandAnalyzer


def temporal_features(landmark_frames: list[np.ndarray]) -> np.ndarray:
    """Build hand-shape and wrist-trajectory features for a dynamic sign."""
    if len(landmark_frames) < 4:
        raise ValueError("يلزم اكتشاف اليد في أربع لقطات على الأقل.")
    points = np.asarray(landmark_frames, dtype=np.float64)
    shapes = np.asarray([normalized_features(frame) for frame in points])
    wrists = points[:, 0, :2]
    palm_scales = np.linalg.norm(points[:, 9, :2] - points[:, 0, :2], axis=1)
    scale = max(float(np.median(palm_scales)), 1e-6)
    trajectory = np.clip((wrists - wrists[0]) / scale, -6.0, 6.0) * 0.35
    return np.column_stack((shapes, trajectory))


def extract_video_sequence(
    video_path: str | Path,
    sample_fps: float = 10.0,
    max_seconds: float = 20.0,
) -> tuple[np.ndarray, dict]:
    """Extract privacy-preserving temporal features without retaining raw frames."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("تعذّر فتح ملف الفيديو.")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round(fps / sample_fps)))
    max_frame = int(max_seconds * fps)
    analyzer = HandAnalyzer(min_detection_confidence=0.5)
    detected: list[np.ndarray] = []
    checked = 0
    frame_index = 0
    try:
        while frame_index < max_frame:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % step == 0:
                checked += 1
                landmarks, _, _ = analyzer.extract(frame)
                if landmarks is not None:
                    detected.append(landmarks)
            frame_index += 1
    finally:
        capture.release()
        analyzer.close()
    if len(detected) < 12:
        raise ValueError("لم تُكتشف اليد في عدد كافٍ من لقطات الفيديو.")
    metadata = {
        "checked_frames": checked,
        "detected_frames": len(detected),
        "detection_rate": round(100.0 * len(detected) / max(checked, 1), 1),
        "duration_seconds": round(frame_index / fps, 2),
    }
    return temporal_features(detected), metadata


def dtw_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Compute a speed-tolerant dynamic-time-warping distance."""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("تسلسلا الحركة غير متوافقين.")
    costs = np.full((len(a) + 1, len(b) + 1), np.inf)
    costs[0, 0] = 0.0
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            local = float(np.sqrt(np.mean((a[i - 1] - b[j - 1]) ** 2)))
            costs[i, j] = local + min(
                costs[i - 1, j], costs[i, j - 1], costs[i - 1, j - 1]
            )
    return float(costs[-1, -1] / max(len(a), len(b)))


def match_sequence(attempt: np.ndarray, reference: np.ndarray) -> dict:
    distance = dtw_distance(attempt, reference)
    score = round(float(100.0 * np.exp(-((distance / 0.55) ** 2))), 1)
    return {
        "sign_name": "السلام عليكم",
        "score": score,
        "distance": round(distance, 4),
        "status": "مطابقة أولية" if score >= 65.0 else "تحتاج إعادة المحاولة",
        "threshold": 65.0,
    }


def match_reference_bank(attempt: np.ndarray, references: list[np.ndarray]) -> dict:
    if not references:
        raise ValueError("لا توجد مراجع فيديو متاحة لهذه الإشارة.")
    results = [match_sequence(attempt, reference) for reference in references]
    best = max(results, key=lambda item: item["score"])
    best["reference_count"] = len(references)
    best["all_scores"] = [item["score"] for item in results]
    return best


def save_reference(path: str | Path, features: np.ndarray, metadata: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sign_name": "السلام عليكم",
        "language": "Saudi Sign Language",
        "source_type": "qualified_translator_video",
        "model_status": "preliminary_single_reference",
        "metadata": metadata,
        "features": np.asarray(features, dtype=float).tolist(),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return output


def load_reference(path: str | Path) -> tuple[np.ndarray, dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return np.asarray(payload["features"], dtype=np.float64), payload
