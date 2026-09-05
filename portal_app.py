from __future__ import annotations

import io
import json
from html import escape
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


ADMIN_EMAIL = "adamessam.alhashim@gmail.com"

st.set_page_config(
    page_title="LumiSign | منصة تعلم الإشارة",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def apply_style() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stHeader"], [data-testid="stToolbar"],
        [data-testid="stDecoration"], #MainMenu, footer {display:none !important;}
        [data-testid="stSidebar"] {display:none !important;}
        .block-container {max-width:1100px;padding-top:1.25rem;padding-bottom:2rem;}
        .stApp {background:radial-gradient(circle at top right,#13283b,#07131f 48%,#050b12);color:#eef7ff;}
        .hero {padding:22px 26px;border-radius:20px;background:linear-gradient(115deg,#0e3340,#14213b);border:1px solid #22606b;margin-bottom:16px;direction:rtl;text-align:right;}
        .hero h1 {margin:0;color:#58f0ce;font-size:2.2rem;}
        .hero p {margin:8px 0 0;color:#c8d8e7;font-size:1.05rem;}
        .badge {display:inline-block;margin-top:12px;padding:5px 11px;border-radius:999px;background:#123f3b;color:#79f5d9;font-size:.84rem;}
        .privacy {direction:rtl;text-align:right;color:#9fb3c2;font-size:.82rem;margin:4px 0 16px;}
        .login-card {max-width:560px;margin:10vh auto 0;padding:34px;border-radius:24px;background:linear-gradient(145deg,#0e3340,#14213b);border:1px solid #286675;text-align:center;direction:rtl;}
        .login-card h1 {color:#58f0ce;margin:0 0 10px;}
        .login-card p {color:#c8d8e7;}
        .account {padding:10px 14px;border:1px solid #22505e;border-radius:14px;background:#0b1c2b;direction:rtl;text-align:right;}
        [data-testid="stMetric"] {background:#0d2233;border:1px solid #1d5260;border-radius:14px;padding:14px;}
        h1,h2,h3,p,label,[data-testid="stMarkdownContainer"] {font-family:"Segoe UI",Tahoma,Arial,sans-serif;}
        @media (max-width: 640px) {
          .block-container {padding-left:.8rem;padding-right:.8rem;padding-top:.7rem;}
          .hero {padding:18px;}
          .hero h1 {font-size:1.85rem;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


apply_style()


def login_page() -> None:
    st.markdown(
        """
        <div class="login-card">
          <h1>🤟 LumiSign</h1>
          <h3>منصة تعلم لغة الإشارة السعودية</h3>
          <p>سجّل الدخول بحساب Google للوصول إلى تجربة تحليل الإشارة.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, center, right = st.columns([1, 1.2, 1])
    with center:
        if st.button("تسجيل الدخول باستخدام Google", type="primary", use_container_width=True):
            st.login()


if not st.user.is_logged_in:
    login_page()
    st.stop()


user_email = str(getattr(st.user, "email", "")).strip().lower()
user_name = str(getattr(st.user, "name", "مستخدم LumiSign"))
is_admin = user_email == ADMIN_EMAIL

account_col, logout_col = st.columns([5, 1])
with account_col:
    role_label = "مدير النظام" if is_admin else "مستخدم"
    st.markdown(
        f'<div class="account">مرحبًا، {escape(user_name)} · {role_label}</div>',
        unsafe_allow_html=True,
    )
with logout_col:
    if st.button("تسجيل الخروج", use_container_width=True):
        st.logout()


@st.cache_resource
def get_analyzer() -> HandAnalyzer:
    return HandAnalyzer()


@st.cache_resource
def get_classifier() -> SaudiSignClassifier | None:
    model = SaudiSignClassifier()
    return model if model.load() else None


@st.cache_data
def reference_bank() -> dict[str, np.ndarray]:
    """Build a deterministic representative reference from the included P01 samples."""
    rows = SaudiSignDataset().records()
    bank: dict[str, np.ndarray] = {}
    for sign_name in sorted({str(row["sign_name"]) for row in rows}):
        sign_rows = [row for row in rows if str(row["sign_name"]) == sign_name]
        participants = sorted({str(row["participant_id"]) for row in sign_rows})
        if not participants:
            continue
        reference_rows = [
            row for row in sign_rows if str(row["participant_id"]) == participants[0]
        ]
        landmarks = [np.asarray(row["landmarks"], dtype=np.float64) for row in reference_rows]
        features = np.asarray([normalized_features(points) for points in landmarks])
        distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
        medoid_index = int(np.argmin(distances.mean(axis=1)))
        bank[sign_name] = landmarks[medoid_index]
    return bank


def decode(upload) -> np.ndarray:
    image = Image.open(io.BytesIO(upload.getvalue())).convert("RGB")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def analysis_page() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🤟 LumiSign</h1>
          <p>تعلّم الإشارة السعودية بتقييم فوري وتوجيه بصري قابل للتفسير</p>
          <span class="badge">نموذج أولي تشغيلي</span>
        </div>
        <div class="privacy">تُحلَّل الصورة لحظيًا لاستخراج معالم اليد، ولا تُحفظ صورة الكاميرا الخام.</div>
        """,
        unsafe_allow_html=True,
    )

    bank = reference_bank()
    if not bank:
        st.error("تعذّر تحميل بنك الإشارات التجريبي. يرجى المحاولة لاحقًا.")
        return

    st.markdown("### جرّب الإشارة الآن")
    target = st.selectbox("اختر الإشارة المطلوب تنفيذها", list(bank))
    source = st.radio("مصدر الصورة", ["الكاميرا", "رفع صورة"], horizontal=True)

    if source == "الكاميرا":
        attempt_input = st.camera_input("ضع يدًا واحدة كاملة وواضحة داخل الإطار")
    else:
        attempt_input = st.file_uploader("ارفع صورة واضحة لليد", type=["jpg", "jpeg", "png"])

    show_ghost = st.toggle("إظهار اليد الشبحية للتوجيه", value=True)

    if attempt_input is None or not st.button(
        "تحليل الإشارة", type="primary", use_container_width=True
    ):
        return

    frame = decode(attempt_input)
    attempt, _, detection_confidence = get_analyzer().extract(frame)

    if attempt is None:
        st.error("لم تُكتشف اليد. حسّن الإضاءة وأظهر اليد كاملة ثم حاول مجددًا.")
        return

    reference = bank[target]
    result = get_analyzer().evaluate(attempt, reference, detection_confidence)
    visual = get_analyzer().draw(frame, attempt)
    if show_ghost:
        visual = ghost_overlay(visual, reference)

    image_column, score_column = st.columns([1.15, 1])
    with image_column:
        st.image(
            cv2.cvtColor(visual, cv2.COLOR_BGR2RGB),
            caption="الأخضر: محاولتك · الوردي: المرجع البصري",
            use_container_width=True,
        )
    with score_column:
        st.metric("درجة التطابق", f"{result.overall_score:.1f}%")
        st.progress(int(result.overall_score))
        first, second = st.columns(2)
        first.metric("شكل اليد", f"{result.hand_shape_score:.1f}%")
        second.metric("زوايا الأصابع", f"{result.finger_angles_score:.1f}%")
        first.metric("اتجاه الكف", f"{result.palm_orientation_score:.1f}%")
        second.metric("موضع اليد", f"{result.hand_position_score:.1f}%")
        st.caption(f"ثقة اكتشاف اليد: {result.confidence:.1f}%")

    st.markdown("### التوجيه التصحيحي")
    for message in result.feedback:
        if result.overall_score >= 85:
            st.success(message)
        else:
            st.warning(message)

    classifier = get_classifier()
    if classifier is not None:
        prediction = classifier.predict_details(normalized_features(attempt))
        st.markdown("### التعرّف الآلي")
        prediction_columns = st.columns(3)
        prediction_columns[0].metric("الإشارة المتوقعة", prediction["predicted_sign"])
        prediction_columns[1].metric("الثقة", f"{prediction['confidence']:.1f}%")
        prediction_columns[2].metric("هامش القرار", f"{prediction['confidence_margin']:.1f}%")
        if prediction["decision_status"] == "needs_human_review":
            st.info("الثقة غير كافية لإصدار قرار مؤكد؛ جرّب صورة أوضح.")
        probability_frame = pd.DataFrame(
            {
                "الإشارة": list(prediction["class_probabilities"].keys()),
                "الاحتمال": list(prediction["class_probabilities"].values()),
            }
        ).set_index("الإشارة")
        st.bar_chart(probability_frame)

    st.divider()
    st.caption("LumiSign LASEE · نموذج أولي لتقييم الإشارات السعودية الثابتة")


def load_json(path: str) -> dict:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def admin_dashboard() -> None:
    st.markdown("## لوحة إدارة LumiSign")
    st.caption("هذه الصفحة لا تظهر إلا لحساب الأدمن المعتمد.")

    dataset = SaudiSignDataset()
    summary = dataset.summary()
    model_report = load_json("data/saudi_sign_classifier.report.json")
    validation = load_json("data/participant_holdout_validation.json")
    audit_valid, audit_events = HashChainAuditLog().verify()

    metrics = st.columns(5)
    metrics[0].metric("العينات", summary.get("total_samples", 0))
    metrics[1].metric("الإشارات", len(summary.get("signs", {})))
    metrics[2].metric("المشاركون", summary.get("participant_count", 0))
    metrics[3].metric("دقة النموذج", f"{model_report.get('accuracy', 0):.1f}%")
    metrics[4].metric("متوسط LASEE", f"{validation.get('mean_overall_score', 0):.1f}%")

    left, right = st.columns(2)
    with left:
        st.markdown("### توزيع العينات")
        sign_counts = summary.get("signs", {})
        if sign_counts:
            st.bar_chart(pd.DataFrame.from_dict(sign_counts, orient="index", columns=["العينات"]))
        else:
            st.info("لا توجد بيانات عينات.")

    with right:
        st.markdown("### حالة التحقق")
        st.success(f"سلسلة التدقيق سليمة · {audit_events} حدث") if audit_valid else st.error(
            "فشل التحقق من سلسلة التدقيق"
        )
        st.write(f"استراتيجية الاختبار: `{model_report.get('split_strategy', 'غير متوفر')}`")
        st.write(f"مشارك الاختبار: `{', '.join(model_report.get('test_participants', [])) or 'غير متوفر'}`")
        st.write(f"معدل اجتياز LASEE: `{validation.get('acceptance_rate', 0)}%`")

    st.markdown("### تفاصيل النموذج")
    model_rows = {
        "عينات التدريب": model_report.get("train_samples", 0),
        "عينات الاختبار": model_report.get("test_samples", 0),
        "أقل درجة LASEE": validation.get("minimum_overall_score", 0),
        "أعلى درجة LASEE": validation.get("maximum_overall_score", 0),
        "عتبة القبول": validation.get("acceptance_threshold", 0),
    }
    st.dataframe(
        pd.DataFrame(model_rows.items(), columns=["المؤشر", "القيمة"]),
        hide_index=True,
        use_container_width=True,
    )


def admin_image_input(key: str):
    source = st.radio(
        "مصدر الصورة",
        ["الكاميرا", "رفع صورة"],
        horizontal=True,
        key=f"admin_source_{key}",
    )
    if source == "الكاميرا":
        return st.camera_input("ضع يدًا واحدة كاملة داخل الإطار", key=f"admin_camera_{key}")
    return st.file_uploader(
        "ارفع صورة واضحة لليد",
        type=["jpg", "jpeg", "png"],
        key=f"admin_upload_{key}",
    )


def admin_references() -> None:
    st.markdown("## تسجيل مرجع جديد")
    st.caption("يُحفظ 21 معلمًا ثلاثي الأبعاد مع المصدر الرسمي للإشارة.")
    store = ReferenceStore()
    audit = HashChainAuditLog()
    sign_name = st.text_input(
        "اسم الإشارة",
        placeholder="مثال: الرقم 5 - لغة الإشارة السعودية",
        key="admin_reference_sign",
    )
    source_url = st.text_input(
        "رابط المصدر السعودي الرسمي",
        value="https://sshi.sa/numbers",
        key="admin_reference_url",
    )
    upload = admin_image_input("reference")
    if upload is not None and st.button("تحليل المرجع وحفظه", type="primary"):
        frame = decode(upload)
        points, handedness, confidence = get_analyzer().extract(frame)
        if points is None:
            st.error("لم يتم اكتشاف اليد. حسّن الإضاءة وأظهر اليد كاملة.")
        elif not sign_name.strip():
            st.error("أدخل اسم الإشارة أولًا.")
        else:
            path = store.save(sign_name, points, handedness or "Unknown", source_url)
            audit.append(
                "reference_created",
                {
                    "sign_name": sign_name,
                    "handedness": handedness,
                    "official_source_url": source_url,
                },
            )
            st.image(
                cv2.cvtColor(get_analyzer().draw(frame, points), cv2.COLOR_BGR2RGB),
                caption="تم اكتشاف 21 معلمًا لليد",
            )
            st.success(f"حُفظ المرجع: {path.name} · ثقة الكشف {confidence * 100:.1f}%")

    signs = store.list_signs()
    st.markdown("### المراجع المسجلة")
    st.write(signs if signs else "لا توجد مراجع مضافة يدويًا حتى الآن.")


def admin_dataset() -> None:
    st.markdown("## جمع بيانات التدريب")
    st.caption("لا تُحفظ الصورة الخام؛ تُحفظ معالم اليد والخصائص الهندسية فقط.")
    dataset = SaudiSignDataset()
    audit = HashChainAuditLog()
    sign_name = st.text_input(
        "اسم الإشارة السعودية",
        value="الرقم 5 - لغة الإشارة السعودية",
        key="admin_data_sign",
    )
    participant = st.text_input("رمز المشارك", value="P01", key="admin_participant")
    source_url = st.text_input(
        "رابط المرجع الرسمي",
        value="https://sshi.sa/numbers",
        key="admin_data_url",
    )
    upload = admin_image_input("dataset")
    if upload is not None and st.button("استخراج المعالم وحفظ العينة", type="primary"):
        frame = decode(upload)
        points, handedness, confidence = get_analyzer().extract(frame)
        if points is None:
            st.error("لم يتم اكتشاف اليد. حسّن الإضاءة وأظهر اليد كاملة.")
        else:
            try:
                record = dataset.append(
                    sign_name,
                    points,
                    participant,
                    source_url,
                    handedness or "Unknown",
                    confidence,
                )
                audit.append(
                    "training_sample_created",
                    {
                        "sample_id": record["sample_id"],
                        "sign_name": sign_name,
                        "participant_id": participant,
                    },
                )
                st.success(f"حُفظت العينة: {record['sample_id']}")
            except ValueError as exc:
                st.error(str(exc))
    st.markdown("### ملخص البيانات")
    st.json(dataset.summary())


def admin_training() -> None:
    st.markdown("## تدريب النموذج واختباره")
    st.caption("يُفصل مشارك كامل للاختبار المستقل كلما توفرت بيانات مشاركين.")
    dataset = SaudiSignDataset()
    classifier = SaudiSignClassifier()
    audit = HashChainAuditLog()
    summary = dataset.summary()
    if summary["total_samples"]:
        st.dataframe(
            pd.DataFrame(
                [{"الإشارة": name, "العينات": count} for name, count in summary["signs"].items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
    if not st.button("تدريب النموذج وإنتاج دليل الاختبار", type="primary"):
        return
    x, y, participant_ids = dataset.training_bundle()
    try:
        if len(summary["signs"]) == 1:
            report = validate_single_sign_participant_holdout(dataset.records())
            report_path = save_validation_report(report, "data/participant_holdout_validation.json")
            audit.append("participant_holdout_validated", report.to_dict())
            columns = st.columns(4)
            columns[0].metric("متوسط LASEE", f"{report.mean_overall_score:.1f}%")
            columns[1].metric("الوسيط", f"{report.median_overall_score:.1f}%")
            columns[2].metric("عينات الاختبار", report.test_samples)
            columns[3].metric("نسبة الاجتياز", f"{report.acceptance_rate:.1f}%")
            st.download_button(
                "تنزيل تقرير الاختبار JSON",
                report_path.read_bytes(),
                file_name="lumisign_participant_holdout_validation.json",
                mime="application/json",
            )
        else:
            report = classifier.train(x, y, participant_ids)
            audit.append("model_trained", report.to_dict())
            st.metric("دقة مجموعة الاختبار", f"{report.accuracy:.1f}%")
            st.dataframe(
                pd.DataFrame(report.confusion_matrix, index=report.labels, columns=report.labels),
                use_container_width=True,
            )
            st.success("تم حفظ النموذج وتقرير الاختبار.")
    except ValueError as exc:
        st.error(str(exc))


def admin_governance() -> None:
    st.markdown("## دليل الحوكمة والإثبات التقني")
    dataset = SaudiSignDataset()
    audit = HashChainAuditLog()
    valid, count = audit.verify()
    dataset_card = dataset.summary()
    manifest = build_evidence_manifest(dataset_card, valid, count)

    columns = st.columns(4)
    columns[0].metric("العينات الموثقة", dataset_card["total_samples"])
    columns[1].metric("الإشارات", len(dataset_card["signs"]))
    columns[2].metric("أحداث السجل", count)
    columns[3].metric("سلامة السجل", "VALID" if valid else "TAMPERED")

    st.markdown(
        """
        - **رؤية حاسوبية:** استخراج 21 معلمًا ثلاثي الأبعاد من صورة حقيقية.
        - **ذكاء اصطناعي قابل للتفسير:** درجات منفصلة للشكل والزوايا والاتجاه والموقع.
        - **خصوصية حسب التصميم:** لا تُحفظ صورة الكاميرا الخام في بيانات التدريب.
        - **حوكمة رقمية:** سلسلة تدقيق SHA-256 مترابطة تكشف العبث.
        """
    )

    integrity_rows = [
        {"الملف": item["path"], "الحجم": item["size_bytes"], "SHA-256": item["sha256"]}
        for item in manifest.get("artifact_integrity", [])
    ]
    if integrity_rows:
        st.dataframe(pd.DataFrame(integrity_rows), hide_index=True, use_container_width=True)

    left, right = st.columns(2)
    left.download_button(
        "تحميل Manifest الإثبات JSON",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="lumisign_technical_evidence_manifest.json",
        mime="application/json",
        use_container_width=True,
    )
    right.download_button(
        "تحميل تقرير الإثبات HTML",
        evidence_html(manifest),
        file_name="lumisign_technical_evidence_report.html",
        mime="text/html",
        use_container_width=True,
    )


if is_admin:
    trial_tab, dashboard_tab, reference_tab, dataset_tab, training_tab, governance_tab = st.tabs(
        [
            "التجربة والتحليل",
            "لوحة الإدارة",
            "تسجيل المراجع",
            "جمع البيانات",
            "تدريب النموذج",
            "الحوكمة والأدلة",
        ]
    )
    with trial_tab:
        analysis_page()
    with dashboard_tab:
        admin_dashboard()
    with reference_tab:
        admin_references()
    with dataset_tab:
        admin_dataset()
    with training_tab:
        admin_training()
    with governance_tab:
        admin_governance()
else:
    analysis_page()
