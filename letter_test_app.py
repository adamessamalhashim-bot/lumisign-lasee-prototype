from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st


# Each reference is the mean of five verified training images.
REFERENCE_CENTROIDS = {
    "حرف الألف": [0.0,0.0,0.0,0.09029095,-0.78776628,-0.09702585,0.28147069,-1.65284114,-0.21915831,0.3534492,-2.29433256,-0.32109853,0.34791923,-2.82516192,-0.39728615,0.43681259,-1.32947776,-0.39243131,0.95808376,-1.21887863,-0.53442711,0.80033497,-1.14873723,-0.56550997,0.59550455,-1.18584832,-0.57330726,0.42869883,-0.75345812,-0.41500522,0.98620676,-0.68103963,-0.50193482,0.80099812,-0.69067917,-0.44448287,0.58105605,-0.71640835,-0.42808049,0.43359329,-0.23241503,-0.43125004,0.93748461,-0.21794181,-0.47236471,0.75855948,-0.2453192,-0.35368112,0.55107871,-0.28033579,-0.30172767,0.43771967,0.20477028,-0.45430881,0.80941013,0.15033151,-0.44841718,0.65470715,0.1133185,-0.34791483,0.47092879,0.07522559,-0.29813108],
    "حرف ب": [0.0,0.0,0.0,0.14416221,-0.16692453,-0.12447436,0.22341722,-0.4420496,-0.21464728,0.17520936,-0.57538611,-0.30151926,0.07277176,-0.57014559,-0.38821461,0.03032514,-1.06678042,-0.17229724,0.00141736,-1.51798387,-0.29392931,-0.01858761,-1.8206375,-0.36491899,-0.03852387,-2.06931373,-0.41434511,-0.13012652,-0.97062748,-0.18240252,-0.0990875,-0.93828649,-0.37206356,-0.00916401,-0.59769656,-0.4187498,0.05263371,-0.41058273,-0.40954468,-0.24187763,-0.7690303,-0.20294494,-0.17507438,-0.61426121,-0.38018008,-0.0631105,-0.34402168,-0.36452979,0.00223118,-0.23245961,-0.31061532,-0.31688123,-0.52849753,-0.23427902,-0.25087463,-0.41014233,-0.35643905,-0.14492075,-0.22544832,-0.33928436,-0.07940217,-0.16341868,-0.2940816],
    "حرف ت": [0.0,0.0,0.0,0.15796431,-0.1030248,-0.10709944,0.24885152,-0.40984744,-0.15723902,0.1923782,-0.61720592,-0.20708031,0.04079467,-0.67397497,-0.25460295,0.12427135,-0.99757426,-0.07548289,0.11590339,-1.40101362,-0.15476302,0.10272217,-1.66264227,-0.20932901,0.08009305,-1.89829515,-0.24637716,-0.00900935,-0.99498585,-0.09236792,-0.0198262,-1.42771084,-0.18580146,-0.01428898,-1.71497375,-0.25091968,-0.02452907,-1.96886899,-0.28407909,-0.12233068,-0.86777834,-0.12239313,-0.16889598,-1.12786058,-0.27666688,-0.07128251,-0.8744449,-0.3162075,-0.00212841,-0.67230991,-0.29915254,-0.21915949,-0.645923,-0.1590509,-0.2113247,-0.76521635,-0.29838433,-0.1118032,-0.59666616,-0.3214794,-0.03727599,-0.44789469,-0.30622673],
}


def normalize_landmarks(points: np.ndarray) -> np.ndarray:
    centered = points - points[0]
    palm_scale = float(np.linalg.norm(centered[9]))
    if palm_scale < 1e-8:
        raise ValueError("تعذر قياس شكل اليد")
    return (centered / palm_scale).reshape(-1)


def classify(features: np.ndarray) -> tuple[str, float, dict[str, float]]:
    # Compare both horizontal orientations so either hand can be used.
    mirrored = features.reshape(21, 3).copy()
    mirrored[:, 0] *= -1
    mirrored = mirrored.reshape(-1)

    distances = {}
    for label, centroid in REFERENCE_CENTROIDS.items():
        reference = np.asarray(centroid, dtype=np.float64)
        distances[label] = min(
            float(np.linalg.norm(features - reference)),
            float(np.linalg.norm(mirrored - reference)),
        )

    labels = list(distances)
    raw = np.asarray([-distances[label] for label in labels], dtype=np.float64)
    raw -= raw.max()
    probabilities = np.exp(raw) / np.exp(raw).sum()
    scores = {label: round(float(p) * 100.0, 1) for label, p in zip(labels, probabilities)}
    predicted = max(scores, key=scores.get)
    return predicted, scores[predicted], scores


def analyze_image(image_bytes: bytes):
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("تعذر قراءة الصورة")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.35,
    ) as hands:
        result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return bgr, None

    hand = result.multi_hand_landmarks[0]
    points = np.asarray([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float64)
    detection = float(result.multi_handedness[0].classification[0].score) * 100.0

    annotated = bgr.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        annotated,
        hand,
        mp.solutions.hands.HAND_CONNECTIONS,
        mp.solutions.drawing_utils.DrawingSpec(color=(38, 224, 183), thickness=3, circle_radius=3),
        mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2),
    )
    predicted, confidence, scores = classify(normalize_landmarks(points))
    return annotated, {
        "predicted": predicted,
        "confidence": confidence,
        "detection": round(detection, 1),
        "scores": scores,
    }


def main() -> None:
    st.set_page_config(page_title="LumiSign — اختبار الحروف", page_icon="🤟", layout="centered")
    st.markdown(
        """
        <style>
        .stApp {direction: rtl; text-align: right; background:#071827; color:#fff}
        h1, h2, h3 {color:#42e6c4 !important}
        [data-testid="stMetric"] {background:#0d2d3d; border:1px solid #1c6070; padding:16px; border-radius:14px}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("LumiSign 🤟 — اختبار الحروف")
    st.write("اختبار أولي للحروف: **الألف، ب، ت** باستخدام 5 صور تدريبية موثقة لكل حرف.")
    st.info("للاختبار الحقيقي استخدم صورة جديدة لم تدخل ضمن صور التدريب، واجعل اليد كاملة وواضحة.")

    source = st.radio("مصدر الصورة", ["رفع صورة", "التقاط بالكاميرا"], horizontal=True)
    if source == "رفع صورة":
        image_file = st.file_uploader("اختر صورة جديدة", type=["jpg", "jpeg", "png"])
    else:
        image_file = st.camera_input("التقط صورة للحرف")

    if image_file is None:
        return

    if st.button("تحليل الحرف", type="primary", width="stretch"):
        try:
            with st.spinner("جارٍ تحليل شكل اليد..."):
                annotated, result = analyze_image(image_file.getvalue())
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="نقاط اليد المكتشفة", width="stretch")
            if result is None:
                st.error("لم يتم اكتشاف اليد. قرّب اليد، وحسّن الإضاءة، وأظهر الأصابع كاملة.")
                return

            col1, col2 = st.columns(2)
            col1.metric("الحرف المتوقع", result["predicted"])
            col2.metric("نسبة التمييز الأولية", f'{result["confidence"]:.1f}%')
            st.progress(int(round(result["confidence"])))
            st.caption(f'جودة اكتشاف اليد: {result["detection"]:.1f}%')

            if result["confidence"] >= 65:
                st.success(f'تم التعرّف مبدئيًا على {result["predicted"]}.')
            else:
                st.warning("النتيجة غير حاسمة وتحتاج مراجعة بشرية أو صورة أوضح.")

            with st.expander("تفاصيل مقارنة الحروف"):
                for label, score in sorted(result["scores"].items(), key=lambda item: item[1], reverse=True):
                    st.write(f"{label}: {score:.1f}%")
        except Exception as exc:
            st.error(f"تعذر تحليل الصورة: {exc}")

    st.divider()
    st.caption("نموذج أولي بحثي — النتائج الحالية لا تمثل اعتمادًا نهائيًا للدقة.")


if __name__ == "__main__":
    main()
