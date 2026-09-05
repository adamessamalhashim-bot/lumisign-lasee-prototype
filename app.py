from __future__ import annotations

import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.audit import HashChainAuditLog
from core.classifier import SaudiSignClassifier
from core.dataset import SaudiSignDataset, normalized_features
from core.evidence import build_evidence_manifest, evidence_html
from core.hand_analyzer import HandAnalyzer, ghost_overlay
from core.reference_store import ReferenceStore
from core.validation import save_validation_report, validate_single_sign_participant_holdout


st.set_page_config(page_title="LumiSign LASEE Demo", page_icon="🤟", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: radial-gradient(circle at top right, #13283b, #07131f 48%, #050b12); color:#eef7ff}
    [data-testid="stMetric"] {background:#0d2233;border:1px solid #1d5260;border-radius:14px;padding:14px}
    .hero {padding:20px 24px;border-radius:18px;background:linear-gradient(115deg,#0e3340,#14213b);border:1px solid #22606b;margin-bottom:18px}
    .hero h1 {margin:0;color:#58f0ce;font-size:2.25rem}.hero p{margin:6px 0 0;color:#c8d8e7}
    .proof {font-size:.88rem;color:#a8bdca;border-left:3px solid #58f0ce;padding-left:12px}
    </style>
    <div class="hero"><h1>LumiSign · LASEE</h1><p>Explainable AI sign evaluation, corrective feedback and Ghost Hand guidance</p></div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def analyzer() -> HandAnalyzer:
    return HandAnalyzer()


store = ReferenceStore()
audit = HashChainAuditLog()
dataset = SaudiSignDataset()
classifier = SaudiSignClassifier()


def decode(upload) -> np.ndarray:
    pil = Image.open(io.BytesIO(upload.getvalue())).convert("RGB")
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def image_input(key: str):
    source = st.radio("Image source", ["Camera", "Upload"], horizontal=True, key=f"source_{key}")
    if source == "Camera":
        return st.camera_input("Place one hand clearly inside the frame", key=f"camera_{key}")
    return st.file_uploader("Upload a clear hand image", type=["jpg", "jpeg", "png"], key=f"upload_{key}")


tab_evaluate, tab_reference, tab_dataset, tab_model, tab_governance = st.tabs(
    ["🎯 تقييم إشارة", "➕ تسجيل مرجع", "📥 جمع بيانات", "🧠 تدريب النموذج", "🔐 دليل الحوكمة"]
)

with tab_reference:
    st.subheader("إنشاء مرجع سعودي موثّق")
    st.caption("سجّل الإشارة الصحيحة؛ يحفظ LumiSign عدد 21 معلمًا ثلاثي الأبعاد كأساس للتقييم.")
    sign_name = st.text_input("اسم الإشارة", placeholder="مثال: الرقم 5 - لغة الإشارة السعودية")
    reference_source = st.text_input("رابط المصدر السعودي الرسمي", value="https://sshi.sa/numbers")
    reference_input = image_input("reference")
    if reference_input and st.button("Analyze and save reference", type="primary"):
        frame = decode(reference_input)
        points, handedness, confidence = analyzer().extract(frame)
        if points is None:
            st.error("لم يتم اكتشاف اليد. حسّن الإضاءة وأظهر اليد كاملة.")
        elif not sign_name.strip():
            st.error("أدخل اسم الإشارة أولًا.")
        else:
            path = store.save(sign_name, points, handedness or "Unknown", reference_source)
            event = audit.append("reference_created", {"sign_name": sign_name, "handedness": handedness, "official_source_url": reference_source})
            annotated = analyzer().draw(frame, points)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="21 MediaPipe landmarks detected")
            st.success(f"Reference saved: {path.name} · Detection confidence {confidence * 100:.1f}%")
            st.code(event["event_hash"], language=None)

with tab_dataset:
    st.subheader("جمع بيانات تدريب موثقة")
    st.caption("الهدف المقترح: 3 إشارات ثابتة × 20 عينة × مشاركين اثنين على الأقل.")
    data_sign = st.text_input(
        "اسم الإشارة السعودية",
        value="الرقم 5 - لغة الإشارة السعودية",
        key="data_sign",
    )
    participant = st.text_input("رمز المشارك", value="P01", key="participant")
    official_url = st.text_input("رابط المرجع الرسمي", value="https://sshi.sa/numbers", key="official_url")
    sample_input = image_input("dataset")
    if sample_input and st.button("استخراج المعالم وحفظ العينة", type="primary"):
        frame = decode(sample_input)
        points, handedness, confidence = analyzer().extract(frame)
        if points is None:
            st.error("لم يتم اكتشاف اليد. حسّن الإضاءة وأظهر اليد كاملة.")
        else:
            try:
                saved_sign = str(st.session_state.get("data_sign") or data_sign or "الرقم 5 - لغة الإشارة السعودية").strip()
                saved_participant = str(st.session_state.get("participant") or participant or "P01").strip()
                saved_source = str(st.session_state.get("official_url") or official_url or "https://sshi.sa/numbers").strip()
                record = dataset.append(saved_sign, points, saved_participant, saved_source, handedness or "Unknown", confidence)
                audit.append("training_sample_created", {"sample_id": record["sample_id"], "sign_name": saved_sign, "participant_id": saved_participant})
                st.image(cv2.cvtColor(analyzer().draw(frame, points), cv2.COLOR_BGR2RGB), caption="تم استخراج 21 معلمًا وحفظها كمدخل تدريب")
                st.success(f"حُفظت العينة: {record['sample_id']} · ثقة الكشف {record['detection_confidence']:.1f}%")
            except ValueError as exc:
                st.error(str(exc))
    summary = dataset.summary()
    st.json(summary)

with tab_model:
    st.subheader("تدريب واختبار تقنيات الإشارات السعودية")
    st.caption("يفصل التطبيق بيانات مشارك كامل للاختبار؛ فلا تدخل محاولاته في المراجع أو التدريب.")
    model_summary = dataset.summary()
    if model_summary["total_samples"]:
        st.dataframe(pd.DataFrame([{"الإشارة": name, "العينات": count} for name, count in model_summary["signs"].items()]), use_container_width=True)
    if st.button("تدريب النموذج وإنتاج دليل الاختبار", type="primary"):
        x, y, participant_ids = dataset.training_bundle()
        try:
            if len(model_summary["signs"]) == 1:
                report = validate_single_sign_participant_holdout(dataset.records())
                report_path = save_validation_report(report, "data/participant_holdout_validation.json")
                audit.append("participant_holdout_validated", report.to_dict())
                st.info("اختبار ثبات مستقل لإشارة سعودية واحدة — وليس دقة تصنيف عدة إشارات.")
                a, b, c, d = st.columns(4)
                a.metric("متوسط LASEE", f"{report.mean_overall_score:.1f}%")
                b.metric("الوسيط", f"{report.median_overall_score:.1f}%")
                c.metric("اجتاز معيار 85%", f"{report.accepted_samples}/{report.test_samples}")
                d.metric("نسبة الاجتياز", f"{report.acceptance_rate:.1f}%")
                st.caption(
                    f"بنك المراجع: {report.reference_participant} ({report.reference_samples} عينة) · "
                    f"اختبار مستقل: {report.test_participant} ({report.test_samples} عينة) · "
                    f"متوسط ثقة MediaPipe: {report.mean_detection_confidence:.1f}%"
                )
                result_rows = pd.DataFrame(report.test_results)
                st.bar_chart(result_rows.set_index("sample_id")[["overall_score"]], horizontal=True)
                st.download_button(
                    "تنزيل تقرير الاختبار المستقل JSON",
                    report_path.read_bytes(),
                    file_name="lumisign_participant_holdout_validation.json",
                    mime="application/json",
                )
                st.success("تم إنتاج دليل اختبار مستقل قابل للتدقيق من بيانات مشارك لم يدخل في بنك المراجع.")
            else:
                report = classifier.train(x, y, participant_ids)
                audit.append("model_trained", report.to_dict())
                st.metric("دقة مجموعة الاختبار", f"{report.accuracy:.1f}%")
                if report.test_participants:
                    st.caption(f"اختبار على مشارك لم يدخل في التدريب: {', '.join(report.test_participants)}")
                st.write("مصفوفة الالتباس")
                st.dataframe(pd.DataFrame(report.confusion_matrix, index=report.labels, columns=report.labels), use_container_width=True)
                st.success("تم حفظ النموذج وتقرير الاختبار ببذرة ثابتة رقم 42.")
        except ValueError as exc:
            st.error(str(exc))

with tab_evaluate:
    signs = store.list_signs()
    if not signs:
        st.info("Record at least one correct reference sample in the second tab, then return here.")
    else:
        target = st.selectbox("Target sign", signs)
        attempt_input = image_input("attempt")
        use_ghost = st.toggle("Show Ghost Hand reference overlay", value=True)
        if attempt_input and st.button("Run LASEE evaluation", type="primary"):
            frame = decode(attempt_input)
            attempt, handedness, confidence = analyzer().extract(frame)
            if attempt is None:
                st.error("No hand was detected. Improve lighting and show the full hand.")
            else:
                reference, metadata = store.load(target)
                result = analyzer().evaluate(attempt, reference, confidence)
                visual = analyzer().draw(frame, attempt)
                if use_ghost:
                    visual = ghost_overlay(visual, reference)

                left, right = st.columns([1.1, 1])
                with left:
                    st.image(
                        cv2.cvtColor(visual, cv2.COLOR_BGR2RGB),
                        caption="Green: detected attempt · Pink: Ghost Hand reference",
                    )
                with right:
                    st.metric("Overall similarity", f"{result.overall_score:.1f}%")
                    a, b = st.columns(2)
                    a.metric("Hand shape", f"{result.hand_shape_score:.1f}%")
                    b.metric("Finger angles", f"{result.finger_angles_score:.1f}%")
                    a.metric("Palm orientation", f"{result.palm_orientation_score:.1f}%")
                    b.metric("Hand position", f"{result.hand_position_score:.1f}%")
                    st.progress(int(result.overall_score))
                    st.caption(f"MediaPipe detection confidence: {result.confidence:.1f}%")

                with st.expander("كيف حُسبت درجة LASEE؟", expanded=True):
                    explanation_rows = []
                    error_units = {
                        "Hand shape": "normalized distance",
                        "Finger angles": "degrees",
                        "Palm orientation": "degrees",
                        "Hand position": "normalized distance",
                    }
                    score_values = {
                        "Hand shape": result.hand_shape_score,
                        "Finger angles": result.finger_angles_score,
                        "Palm orientation": result.palm_orientation_score,
                        "Hand position": result.hand_position_score,
                    }
                    for component, weight in result.component_weights.items():
                        explanation_rows.append(
                            {
                                "المكوّن": component,
                                "الخطأ المقاس": result.raw_errors[component],
                                "الوحدة": error_units[component],
                                "الدرجة": score_values[component],
                                "الوزن": f"{weight * 100:.0f}%",
                                "المساهمة": result.weighted_contributions[component],
                            }
                        )
                    st.dataframe(pd.DataFrame(explanation_rows), use_container_width=True, hide_index=True)
                    st.code(
                        "Overall = 0.38×HandShape + 0.32×FingerAngles + "
                        "0.20×PalmOrientation + 0.10×HandPosition",
                        language=None,
                    )
                    st.caption(f"أضعف مكوّن في هذه المحاولة: {result.worst_feature}")

                if classifier.load():
                    prediction = classifier.predict_details(normalized_features(attempt))
                    st.subheader("التعرّف الآلي على الإشارة")
                    p1, p2, p3 = st.columns(3)
                    p1.metric("الإشارة المتوقعة", prediction["predicted_sign"])
                    p2.metric("ثقة المصنّف", f"{prediction['confidence']:.1f}%")
                    p3.metric("هامش القرار", f"{prediction['confidence_margin']:.1f}%")
                    probability_frame = pd.DataFrame(
                        {
                            "الإشارة": list(prediction["class_probabilities"].keys()),
                            "الاحتمال": list(prediction["class_probabilities"].values()),
                        }
                    ).set_index("الإشارة")
                    st.bar_chart(probability_frame)
                    if prediction["decision_status"] == "accepted":
                        st.success("قرار المصنّف تجاوز بوابة الثقة المحددة ويمكن عرضه للمتعلم.")
                    else:
                        st.warning("الثقة أقل من الحد المحدد؛ تُحال النتيجة للمراجعة بدل تقديمها كقرار مؤكد.")
                else:
                    prediction = None

                st.subheader("تغذية راجعة تصحيحية قابلة للتفسير")
                for item in result.feedback:
                    if result.overall_score < 85:
                        st.warning(item)
                    else:
                        st.success(item)
                event = audit.append(
                    "sign_evaluated",
                    {"target_sign": target, "handedness": handedness, **result.to_dict()},
                )
                st.markdown(
                    f'<p class="proof">Audit evidence · SHA-256 event hash: <code>{event["event_hash"]}</code></p>',
                    unsafe_allow_html=True,
                )
                evaluation_evidence = {
                    "solution": "LumiSign Connect",
                    "engine": "LASEE",
                    "target_sign": target,
                    "reference_metadata": metadata,
                    "evaluation": result.to_dict(),
                    "classifier_prediction": prediction,
                    "audit_event": event,
                    "claim_scope": "Evaluation of one captured static hand pose against a recorded reference.",
                }
                st.download_button(
                    "تنزيل تقرير هذه المحاولة",
                    json.dumps(evaluation_evidence, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="lumisign_attempt_evidence.json",
                    mime="application/json",
                )

with tab_governance:
    valid, count = audit.verify()
    dataset_card = dataset.summary()
    manifest = build_evidence_manifest(dataset_card, valid, count)
    a, b, c, d = st.columns(4)
    a.metric("العينات الموثقة", dataset_card["total_samples"])
    b.metric("الإشارات", len(dataset_card["signs"]))
    c.metric("قرارات السجل", count)
    d.metric("سلامة السجل", "VALID" if valid else "TAMPERED")

    st.subheader("مركز الإثبات التقني والحوكمة")
    st.markdown(
        """
        - **الرؤية الحاسوبية:** تستخرج MediaPipe عدد 21 معلمًا ثلاثي الأبعاد من صورة حقيقية.
        - **ذكاء اصطناعي قابل للتفسير:** تنتج الأخطاء الهندسية درجات منفصلة للشكل والزوايا والاتجاه والموقع.
        - **توجيه بصري معزز:** يعرض Ghost Hand المرجع الموثق فوق محاولة المتعلم.
        - **ذكاء اصطناعي مسؤول:** تمنع بوابة الثقة عرض التصنيف غير الحاسم كقرار مؤكد.
        - **خصوصية حسب التصميم:** لا تُحفظ صورة الكاميرا الخام في بيانات التدريب؛ تُحفظ المعالم والخصائص الهندسية فقط.
        - **حوكمة رقمية:** يرتبط كل حدث بما قبله داخل سلسلة تدقيق SHA-256 تكشف العبث.
        """
    )

    model_card = manifest.get("model_card") or {}
    validation_card = manifest.get("participant_holdout_validation") or {}
    st.subheader("بطاقات الدليل المقاس")
    rows = [
        {"الدليل": "Dataset Card", "القيمة": f"{dataset_card['total_samples']} عينة / {len(dataset_card['signs'])} إشارتان / {dataset_card['participant_count']} مشاركين"},
        {"الدليل": "Model Card", "القيمة": f"دقة {model_card.get('accuracy', 'غير متاح')}% · {model_card.get('split_strategy', 'غير متاح')}"},
        {"الدليل": "LASEE Holdout", "القيمة": f"متوسط {validation_card.get('mean_overall_score', 'غير متاح')}% · اجتياز {validation_card.get('accepted_samples', 'غير متاح')}/{validation_card.get('test_samples', 'غير متاح')}"},
        {"الدليل": "Audit Chain", "القيمة": f"{count} حدثًا · SHA-256 · {'VALID' if valid else 'TAMPERED'}"},
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("بصمات سلامة ملفات الإثبات")
    integrity_rows = [
        {"الملف": item["path"], "الحجم": item["size_bytes"], "SHA-256": item["sha256"]}
        for item in manifest["artifact_integrity"]
    ]
    st.dataframe(pd.DataFrame(integrity_rows), use_container_width=True, hide_index=True)
    st.caption(manifest["claim_scope"])

    if Path("data/audit_log.jsonl").exists():
        st.download_button(
            "Download audit evidence",
            Path("data/audit_log.jsonl").read_bytes(),
            file_name="lumisign_audit_evidence.jsonl",
            mime="application/jsonl",
        )
    left_download, right_download = st.columns(2)
    left_download.download_button(
        "تحميل Manifest الإثبات JSON",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="lumisign_technical_evidence_manifest.json",
        mime="application/json",
        use_container_width=True,
    )
    right_download.download_button(
        "تحميل تقرير الإثبات HTML",
        evidence_html(manifest),
        file_name="lumisign_technical_evidence_report.html",
        mime="text/html",
        use_container_width=True,
    )
