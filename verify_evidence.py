from __future__ import annotations

import json
from pathlib import Path

from core.audit import HashChainAuditLog
from core.dataset import SaudiSignDataset
from core.evidence import build_evidence_manifest, evidence_html


def main() -> int:
    dataset = SaudiSignDataset()
    records = dataset.records()
    summary = dataset.summary()
    audit_valid, audit_events = HashChainAuditLog().verify()

    schema_valid = bool(records) and all(
        len(row.get("landmarks", [])) == 21
        and all(len(point) == 3 for point in row.get("landmarks", []))
        and len(row.get("features", [])) == 63
        and bool(row.get("official_source_url"))
        and bool(row.get("participant_id"))
        for row in records
    )
    model_report_path = Path("data/saudi_sign_classifier.report.json")
    validation_path = Path("data/participant_holdout_validation.json")
    model_report = json.loads(model_report_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    checks = {
        "dataset_schema_21x3_and_63_features": schema_valid,
        "two_or_more_signs": len(summary["signs"]) >= 2,
        "two_or_more_participants": summary["participant_count"] >= 2,
        "participant_holdout_model_test": model_report.get("split_strategy") == "participant_holdout",
        "participant_holdout_lasee_test": validation.get("method") == "participant_holdout_multi_reference_lasee",
        "sha256_audit_chain": audit_valid,
    }
    manifest = build_evidence_manifest(summary, audit_valid, audit_events)
    manifest["automated_verification"] = checks
    manifest_path = Path("data/lumisign_technical_evidence_manifest.json")
    html_path = Path("data/lumisign_technical_evidence_report.html")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_bytes(evidence_html(manifest))

    print("LumiSign Technical Evidence Verification")
    print("=" * 42)
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print("-" * 42)
    print(f"Samples: {summary['total_samples']}")
    print(f"Signs: {len(summary['signs'])}")
    print(f"Participants: {summary['participant_count']}")
    print(f"Random Forest accuracy: {model_report.get('accuracy')}%")
    print(f"LASEE mean: {validation.get('mean_overall_score')}%")
    print(f"Audit events: {audit_events} ({'VALID' if audit_valid else 'TAMPERED'})")
    print(f"Manifest: {manifest_path}")
    print(f"HTML report: {html_path}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
