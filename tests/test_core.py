import json

import numpy as np

from core.audit import HashChainAuditLog
from core.classifier import SaudiSignClassifier
from core.dataset import SaudiSignDataset, normalized_features
from core.evidence import build_evidence_manifest, evidence_html, sha256_file
from core.hand_analyzer import HandAnalyzer
from core.validation import validate_single_sign_participant_holdout
from core.video_sign import dtw_distance, match_sequence, temporal_features


def sample_hand():
    # Non-degenerate synthetic geometry sufficient for deterministic math tests.
    return np.array([[0.40 + (i % 5) * 0.025, 0.70 - (i // 5) * 0.08, i * 0.001] for i in range(21)])


def test_identical_hand_scores_near_100():
    analyzer = HandAnalyzer.__new__(HandAnalyzer)
    points = sample_hand()
    result = analyzer.evaluate(points, points.copy(), 0.97)
    assert result.overall_score >= 99.0
    assert result.confidence == 97.0
    assert round(sum(result.weighted_contributions.values()), 1) == result.overall_score
    assert result.raw_errors["Hand shape"] == 0.0


def test_normalization_is_translation_invariant():
    points = sample_hand()
    shifted = points + np.array([0.2, -0.1, 0.05])
    assert np.allclose(HandAnalyzer.normalize_shape(points), HandAnalyzer.normalize_shape(shifted))


def test_audit_chain_detects_tampering(tmp_path):
    log = HashChainAuditLog(tmp_path / "audit.jsonl")
    log.append("one", {"score": 90})
    log.append("two", {"score": 85})
    assert log.verify() == (True, 2)

    lines = log.path.read_text(encoding="utf-8").splitlines()
    event = json.loads(lines[0])
    event["data"]["score"] = 1
    lines[0] = json.dumps(event)
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert log.verify()[0] is False


def test_saudi_dataset_preserves_input_provenance(tmp_path):
    dataset = SaudiSignDataset(tmp_path / "samples.jsonl")
    record = dataset.append(
        "الرقم 5",
        sample_hand(),
        "P01",
        "https://sshi.sa/numbers",
        "Right",
        0.958,
    )
    assert record["language"] == "Saudi Sign Language"
    assert record["official_source_url"] == "https://sshi.sa/numbers"
    assert len(record["features"]) == 63
    assert dataset.summary()["signs"]["الرقم 5"] == 1


def test_random_forest_report_uses_held_out_samples(tmp_path):
    base = normalized_features(sample_hand())
    x = []
    y = []
    for index in range(8):
        x.append(base + index * 0.0001)
        y.append("الرقم 5")
        x.append(base + 1.0 + index * 0.0001)
        y.append("الرقم 2")
    classifier = SaudiSignClassifier(tmp_path / "model.joblib")
    report = classifier.train(np.asarray(x), np.asarray(y))
    assert report.train_samples > 0
    assert report.test_samples > 0
    assert report.split_strategy == "stratified_sample_holdout"
    assert report.accuracy == 100.0
    label, confidence = classifier.predict(np.asarray(x[0]))
    assert label == "الرقم 5"
    assert confidence > 50


def test_classifier_can_hold_out_an_unseen_participant(tmp_path):
    base = normalized_features(sample_hand())
    x, y, participants = [], [], []
    for participant in ["P01", "P02"]:
        for index in range(4):
            x.extend([base + index * 0.0001, base + 1.0 + index * 0.0001])
            y.extend(["الرقم 5", "الرقم 2"])
            participants.extend([participant, participant])
    classifier = SaudiSignClassifier(tmp_path / "participant_model.joblib")
    report = classifier.train(np.asarray(x), np.asarray(y), np.asarray(participants))
    assert report.split_strategy == "participant_holdout"
    assert report.test_participants == ["P02"]


def test_classifier_exposes_uncertainty_and_human_review_gate(tmp_path):
    base = normalized_features(sample_hand())
    x, y = [], []
    for index in range(8):
        x.extend([base + index * 0.0001, base + 1.0 + index * 0.0001])
        y.extend(["الرقم 5", "الرقم 2"])
    classifier = SaudiSignClassifier(tmp_path / "model.joblib")
    classifier.train(np.asarray(x), np.asarray(y))
    details = classifier.predict_details(np.asarray(x[0]), confidence_threshold=101.0)
    assert set(details["class_probabilities"]) == {"الرقم 5", "الرقم 2"}
    assert details["decision_status"] == "needs_human_review"
    assert details["confidence_margin"] >= 0


def test_single_sign_validation_never_puts_test_participant_in_reference_bank():
    rows = []
    for participant, offset in [("P01", 0.0), ("P02", 0.002)]:
        for index in range(3):
            points = sample_hand() + np.array([offset + index * 0.0001, 0.0, 0.0])
            rows.append(
                {
                    "sample_id": f"{participant}-{index}",
                    "language": "Saudi Sign Language",
                    "sign_name": "الرقم 5",
                    "participant_id": participant,
                    "detection_confidence": 97.0,
                    "landmarks": points.tolist(),
                }
            )
    report = validate_single_sign_participant_holdout(rows)
    assert report.reference_participant == "P01"
    assert report.test_participant == "P02"
    assert report.reference_samples == 3
    assert report.test_samples == 3
    assert report.accepted_samples == 3
    assert all(row["sample_id"].startswith("P02-") for row in report.test_results)


def test_evidence_manifest_hashes_real_artifacts(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    sample_file = data / "saudi_sign_samples.jsonl"
    sample_file.write_text('{"sample_id":"one"}\n', encoding="utf-8")
    manifest = build_evidence_manifest(
        {"total_samples": 1, "signs": {"الرقم 5": 1}, "participant_count": 1},
        True,
        3,
        tmp_path,
    )
    assert manifest["artifact_integrity"][0]["sha256"] == sha256_file(sample_file)
    assert manifest["audit"]["valid"] is True
    assert b"LumiSign" in evidence_html(manifest)


def test_dynamic_sequence_identical_reference_scores_100():
    frames = [sample_hand() + np.array([index * 0.002, 0.0, 0.0]) for index in range(8)]
    features = temporal_features(frames)
    assert dtw_distance(features, features) == 0.0
    assert match_sequence(features, features)["score"] == 100.0
