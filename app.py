"""Streamlit image-inspection interface."""

from pathlib import Path

import streamlit as st
from PIL import Image

from flange_inspector.detector import FlangeDetector


@st.cache_resource
def load_detector(weights: str) -> FlangeDetector:
    return FlangeDetector(weights)


st.set_page_config(page_title="Flange Defect Inspector", page_icon="🔩", layout="wide")
st.title("Flange Defect Inspector")
st.write("Upload an image to detect flange defects with a trained YOLO model.")

with st.sidebar:
    st.header("Detection settings")
    weights = st.text_input("Model weights", "runs/detect/flange/weights/best.pt")
    confidence = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
    image_size = st.selectbox("Inference image size", [320, 480, 640, 800, 1024], index=2)

uploaded_image = st.file_uploader(
    "Choose a flange image", type=("jpg", "jpeg", "png", "bmp", "webp")
)

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    left, right = st.columns(2)
    left.subheader("Input image")
    left.image(image, use_container_width=True)

    if st.button("Detect defects", type="primary"):
        if not Path(weights).expanduser().is_file():
            st.error(f"Model weights not found: {weights}")
        else:
            with st.spinner("Running YOLO inference..."):
                detector = load_detector(weights)
                results = detector.predict(
                    image,
                    confidence=confidence,
                    image_size=image_size,
                )
            result = results[0]
            annotated = result.plot()
            right.subheader("Detection result")
            right.image(annotated[:, :, ::-1], use_container_width=True)

            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    detections.append(
                        {
                            "Defect": result.names[class_id],
                            "Confidence": round(float(box.conf.item()), 4),
                            "Box (xyxy)": [
                                round(float(value), 1)
                                for value in box.xyxy[0].tolist()
                            ],
                        }
                    )

            st.subheader(f"Detections ({len(detections)})")
            if detections:
                st.dataframe(detections, use_container_width=True, hide_index=True)
            else:
                st.info("No defects detected above the selected confidence threshold.")
