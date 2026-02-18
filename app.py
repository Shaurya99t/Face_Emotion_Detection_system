import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Face Emotion Detector",
    page_icon="😊",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #4CAF50;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #666;
    margin-bottom: 30px;
}
.stButton > button {
    width: 100%;
    height: 3em;
    font-size: 18px;
    border-radius: 12px;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background-color: #f9f9f9;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
classes = model.names

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">Face Emotion Detection </div>', unsafe_allow_html=True)
st.markdown('<div class="main-title">By Shaurya Seth </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Powered by YOLOv8 • Fast • Accurate</div>', unsafe_allow_html=True)

# ---------------- MODE STATE ----------------
if "mode" not in st.session_state:
    st.session_state.mode = None

# ---------------- BUTTONS ----------------
col1, col2 = st.columns(2)

with col1:
    if st.button("📤 Upload Image"):
        st.session_state.mode = "image"

with col2:
    if st.button("📷 Use Webcam"):
        st.session_state.mode = "webcam"

st.markdown("---")

# ---------------- IMAGE MODE ----------------
if st.session_state.mode == "image":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    img_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

    if img_file:
        image = Image.open(img_file).convert("RGB")
        st.image(image, use_column_width=True)

        result = model(image)[0]
        emotion = classes[result.probs.top1]
        confidence = result.probs.top1conf.item() * 100

        st.success(f"🎯 Emotion: **{emotion}**")
        st.info(f"🔍 Confidence: **{confidence:.2f}%**")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- WEBCAM MODE ----------------
elif st.session_state.mode == "webcam":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.warning("Click STOP in Streamlit to turn off webcam")

    cap = cv2.VideoCapture(0)
    frame_area = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam not accessible")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = model(rgb)[0]

        emotion = classes[result.probs.top1]
        conf = result.probs.top1conf.item()

        cv2.putText(
            frame,
            f"{emotion} ({conf:.2f})",
            (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        frame_area.image(frame, channels="BGR")

    cap.release()
    st.markdown('</div>', unsafe_allow_html=True)
