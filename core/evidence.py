from __future__ import annotations

import hashlib
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_or_none(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_evidence_manifest(
    dataset_summary: dict[str, Any],
    audit_valid: bool,
    audit_events: int,
    base_directory: str | Path = ".",
) -> dict[str, Any]:
    """Build a reproducible evidence manifest from the actual local artifacts."""
    base = Path(base_directory)
    artifacts = []
    for relative in [
        "data/saudi_sign_samples.jsonl",
        "data/saudi_sign_classifier.joblib",
        "data/saudi_sign_classifier.report.json",
        "data/participant_holdout_validation.json",
        "data/audit_log.jsonl",
    ]:
        path = base / relative
        if path.exists():
            artifacts.append(
                {
                    "path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    return {
        "solution": "LumiSign Connect",
        "engine": "LASEE — LumiSign Adaptive Sign Evaluation Engine",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "language_scope": "Saudi Sign Language",
        "implemented_pipeline": [
            "Camera or uploaded image",
            "MediaPipe extraction of 21 three-dimensional hand landmarks",
            "63 translation- and scale-normalized numeric features",
            "Random Forest classification with participant holdout",
            "LASEE geometric scoring and Arabic corrective feedback",
            "Ghost Hand reference overlay",
            "SHA-256 linked audit trail",
        ],
        "responsible_ai_controls": [
            "Participant-level test separation",
            "Classifier confidence threshold and human-review status",
            "Component-level geometric explanation",
            "Raw camera image is not stored in the training dataset",
            "Tamper-evident decision log",
        ],
        "dataset_card": dataset_summary,
        "model_card": _json_or_none(base / "data/saudi_sign_classifier.report.json"),
        "participant_holdout_validation": _json_or_none(base / "data/participant_holdout_validation.json"),
        "audit": {"valid": bool(audit_valid), "events": int(audit_events), "algorithm": "SHA-256"},
        "artifact_integrity": artifacts,
        "claim_scope": (
            "Validated prototype for two static Saudi Sign Language signs and two pseudonymous participants; "
            "not evidence of full-vocabulary or dynamic-sign generalization."
        ),
    }


def evidence_html(manifest: dict[str, Any]) -> bytes:
    dataset = manifest.get("dataset_card", {})
    model = manifest.get("model_card") or {}
    validation = manifest.get("participant_holdout_validation") or {}
    rows = "".join(
        f"<tr><td>{html.escape(item['path'])}</td><td>{item['size_bytes']}</td><td><code>{item['sha256']}</code></td></tr>"
        for item in manifest.get("artifact_integrity", [])
    )
    document = f"""<!doctype html><html lang='ar' dir='rtl'><head><meta charset='utf-8'>
<title>LumiSign Technical Evidence Report</title><style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:40px auto;color:#10263b;line-height:1.7}}
h1,h2{{color:#087f72}}.metric{{display:inline-block;border:1px solid #9ec9c4;padding:12px 18px;margin:6px;border-radius:10px}}
table{{border-collapse:collapse;width:100%;direction:ltr}}th,td{{border:1px solid #ccd8df;padding:8px;text-align:left}}code{{font-size:11px;word-break:break-all}}
</style></head><body>
<h1>LumiSign · LASEE — تقرير الإثبات التقني</h1>
<p>تقرير مولّد من ملفات التشغيل الفعلية بتاريخ {html.escape(manifest['generated_at'])}.</p>
<div class='metric'><b>العينات</b><br>{dataset.get('total_samples', 0)}</div>
<div class='metric'><b>الإشارات</b><br>{len(dataset.get('signs', {}))}</div>
<div class='metric'><b>المشاركون</b><br>{dataset.get('participant_count', 0)}</div>
<div class='metric'><b>دقة المصنف</b><br>{model.get('accuracy', 'غير متاح')}%</div>
<div class='metric'><b>متوسط LASEE</b><br>{validation.get('mean_overall_score', 'غير متاح')}%</div>
<div class='metric'><b>سجل التدقيق</b><br>{'VALID' if manifest['audit']['valid'] else 'TAMPERED'}</div>
<h2>حدود الادعاء</h2><p>{html.escape(manifest['claim_scope'])}</p>
<h2>بصمات سلامة الأدلة</h2><table><tr><th>Artifact</th><th>Bytes</th><th>SHA-256</th></tr>{rows}</table>
</body></html>"""
    return document.encode("utf-8")
