from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from core.hand_analyzer import HandAnalyzer


@dataclass
class ParticipantHoldoutReport:
    language: str
    sign_name: str
    method: str
    reference_participant: str
    test_participant: str
    reference_samples: int
    test_samples: int
    acceptance_threshold: float
    accepted_samples: int
    acceptance_rate: float
    mean_overall_score: float
    median_overall_score: float
    minimum_overall_score: float
    maximum_overall_score: float
    mean_hand_shape_score: float
    mean_finger_angles_score: float
    mean_palm_orientation_score: float
    mean_hand_position_score: float
    mean_detection_confidence: float
    test_results: list[dict]
    claim_scope: str

    def to_dict(self) -> dict:
        return asdict(self)


def validate_single_sign_participant_holdout(
    records: list[dict],
    acceptance_threshold: float = 85.0,
) -> ParticipantHoldoutReport:
    """Validate one static sign without leaking test-participant samples.

    The first participant supplies a bank of valid reference poses. Each sample
    from the second participant is compared with every reference, and the best
    LASEE score is retained. This measures cross-participant consistency for one
    sign; it is deliberately not presented as multi-class recognition accuracy.
    """
    if not records:
        raise ValueError("No Saudi Sign Language samples are available")

    signs = sorted({str(row["sign_name"]) for row in records})
    if len(signs) != 1:
        raise ValueError("Single-sign validation requires exactly one sign label")

    participants = sorted({str(row["participant_id"]) for row in records})
    if len(participants) < 2:
        raise ValueError("At least two participant codes are required for an independent holdout test")

    reference_participant, test_participant = participants[:2]
    references = [row for row in records if str(row["participant_id"]) == reference_participant]
    tests = [row for row in records if str(row["participant_id"]) == test_participant]
    if not references or not tests:
        raise ValueError("Both reference and test participants must contain samples")

    analyzer = HandAnalyzer.__new__(HandAnalyzer)
    test_results: list[dict] = []
    for row in tests:
        attempt = np.asarray(row["landmarks"], dtype=np.float64)
        confidence = float(row.get("detection_confidence", 0.0)) / 100.0
        candidates = [
            analyzer.evaluate(attempt, np.asarray(reference["landmarks"], dtype=np.float64), confidence)
            for reference in references
        ]
        best_index = int(np.argmax([candidate.overall_score for candidate in candidates]))
        best = candidates[best_index]
        test_results.append(
            {
                "sample_id": row.get("sample_id", ""),
                "matched_reference_sample_id": references[best_index].get("sample_id", ""),
                "overall_score": best.overall_score,
                "hand_shape_score": best.hand_shape_score,
                "finger_angles_score": best.finger_angles_score,
                "palm_orientation_score": best.palm_orientation_score,
                "hand_position_score": best.hand_position_score,
                "detection_confidence": float(row.get("detection_confidence", 0.0)),
                "accepted": best.overall_score >= acceptance_threshold,
            }
        )

    def mean(field: str) -> float:
        return round(float(np.mean([row[field] for row in test_results])), 1)

    scores = [row["overall_score"] for row in test_results]
    accepted = sum(bool(row["accepted"]) for row in test_results)
    all_confidences = [float(row.get("detection_confidence", 0.0)) for row in records]
    return ParticipantHoldoutReport(
        language="Saudi Sign Language",
        sign_name=signs[0],
        method="participant_holdout_multi_reference_lasee",
        reference_participant=reference_participant,
        test_participant=test_participant,
        reference_samples=len(references),
        test_samples=len(tests),
        acceptance_threshold=float(acceptance_threshold),
        accepted_samples=accepted,
        acceptance_rate=round(100.0 * accepted / len(tests), 1),
        mean_overall_score=round(float(np.mean(scores)), 1),
        median_overall_score=round(float(np.median(scores)), 1),
        minimum_overall_score=round(float(np.min(scores)), 1),
        maximum_overall_score=round(float(np.max(scores)), 1),
        mean_hand_shape_score=mean("hand_shape_score"),
        mean_finger_angles_score=mean("finger_angles_score"),
        mean_palm_orientation_score=mean("palm_orientation_score"),
        mean_hand_position_score=mean("hand_position_score"),
        mean_detection_confidence=round(float(np.mean(all_confidences)), 1),
        test_results=test_results,
        claim_scope=(
            "Positive cross-participant consistency validation for one static Saudi Sign Language sign; "
            "not multi-class recognition accuracy."
        ),
    )


def save_validation_report(report: ParticipantHoldoutReport, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return output
