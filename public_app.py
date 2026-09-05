from __future__ import annotations

import io

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from core.classifier import SaudiSignClassifier
from core.dataset import SaudiSignDataset, normalized_features
from core.hand_analyzer import HandAnalyzer, ghost_overlay


st.set_page_config(
    page_title="LumiSign | جرّب الإشارة",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
    [data-testid="stMetric"] {background:#0d2233;border:1px solid #1d5260;border-radius:14px;padding:14px;}
    h1,h2,h3,p,label,[data-testid="stMarkdownContainer"] {font-family:"Segoe UI",Tahoma,Arial,sans-serif;}
    </style>
    <div class="hero">
      <h1>🤟 LumiSign</h1>
      <p>تعلّم الإشارة السعودية بتقييم فوري وتوجيه بصري قابل للتفسير</p>
      <span class="badge">نموذج أولي تشغيلي</span>
    </div>
    <div class="privacy">تُحلَّل الصورة لحظيًا لاستخراج معالم اليد، ولا تُحفظ صورة الكاميرا الخام.</div>
    """,
    unsafe_allow_html=True,
)


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


bank = reference_bank()
if not bank:
    st.error("تعذّر تحميل بنك الإشارات التجريبي. يرجى المحاولة لاحقًا.")
    st.stop()

st.markdown("### جرّب الإشارة الآن")
target = st.selectbox("اختر الإشارة المطلوب تنفيذها", list(bank))
source = st.radio("مصدر الصورة", ["الكاميرا", "رفع صورة"], horizontal=True)

if source == "الكاميرا":
    attempt_input = st.camera_input("ضع يدًا واحدة كاملة وواضحة داخل الإطار")
else:
    attempt_input = st.file_uploader(
        "ارفع صورة واضحة لليد",
        type=["jpg", "jpeg", "png"],
    )

show_ghost = st.toggle("إظهار اليد الشبحية للتوجيه", value=True)

if attempt_input is not None and st.button("تحليل الإشارة", type="primary", use_container_width=True):
    frame = decode(attempt_input)
    attempt, _, detection_confidence = get_analyzer().extract(frame)

    if attempt is None:
        st.error("لم تُكتشف اليد. حسّن الإضاءة وأظهر اليد كاملة ثم حاول مجددًا.")
    else:
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
