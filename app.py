import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

st.set_page_config(
    page_title="YOLOv8 Medical Imaging",
    page_icon="🏥",
    layout="wide"
)

# Model paths
CLS_MODEL = os.path.join(os.path.dirname(__file__), 'models', 'covid_cls.pt')
DET_MODEL = os.path.join(os.path.dirname(__file__), 'models', 'bloodcells_det.pt')
SEG_MODEL = os.path.join(os.path.dirname(__file__), 'models', 'breast_seg.pt')

@st.cache_resource
def load_model(path):
    return YOLO(path)

# Sidebar
with st.sidebar:
    st.title("YOLOv8 Medical Imaging")
    st.markdown("---")
    mode = st.selectbox("Choose App Mode", [
        "About App",
        "COVID-19 Classification",
        "Blood Cell Detection",
        "Breast Ultrasound Segmentation"
    ])
    st.markdown("---")
    st.caption("Submitted to: Mam Samia Kiran")
    st.caption("By: Hermen Khan, Babar Ali, Muhammad Taqui")

# About page
if mode == "About App":
    st.title("YOLOv8 for Medical Imaging")
    st.markdown("""
    This application demonstrates YOLOv8-based solutions for medical imaging analysis.

    ### Capabilities
    - **COVID-19 Classification** — Classifies chest X-rays as Normal, COVID-19, or Pneumonia
    - **Blood Cell Detection** — Detects and localizes RBC and WBC in microscopy images
    - **Breast Ultrasound Segmentation** — Segments normal, benign and malignant masses

    ### Models
    | Task | Dataset | Metric | Score |
    |------|---------|--------|-------|
    | Classification | COVID-19 X-rays | Accuracy | 96.5% |
    | Detection | Blood Cells | mAP50 | 99.3% |
    | Segmentation | Breast Ultrasound | mAP50 | 72.3% |

    ### How to Use
    Select a mode from the sidebar, upload an image and view the results.
    """)

# Classification
elif mode == "COVID-19 Classification":
    st.title("COVID-19 Classification with YOLOv8")
    st.markdown("Upload a chest X-ray to classify it as Normal, COVID-19, or Pneumonia.")

    uploaded = st.file_uploader("Upload Chest X-ray", type=['jpg', 'jpeg', 'png'])

    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Classification Result")
            with st.spinner("Classifying..."):
                model = load_model(CLS_MODEL)
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    image.save(f.name)
                    results = model(f.name)
                    os.unlink(f.name)

                probs = results[0].probs
                names = results[0].names
                top1 = probs.top1
                top1_conf = float(probs.top1conf)

                label = names[top1]
                st.metric("Prediction", label)
                st.metric("Confidence", f"{top1_conf:.2%}")

                st.markdown("#### All Class Probabilities")
                for i, name in names.items():
                    conf = float(probs.data[i])
                    st.progress(conf, text=f"{name}: {conf:.2%}")

# Detection
elif mode == "Blood Cell Detection":
    st.title("Blood Cell Detection with YOLOv8")
    st.markdown("Upload a blood cell microscopy image to detect RBC and WBC.")

    uploaded = st.file_uploader("Upload Blood Cell Image", type=['jpg', 'jpeg', 'png'])

    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Detection Result")
            with st.spinner("Detecting..."):
                model = load_model(DET_MODEL)
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    image.save(f.name)
                    results = model(f.name)
                    result_img = results[0].plot()
                    os.unlink(f.name)

                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, use_container_width=True)

                boxes = results[0].boxes
                if boxes is not None:
                    names = results[0].names
                    from collections import Counter
                    counts = Counter([names[int(c)] for c in boxes.cls])
                    st.markdown("#### Detected Cells")
                    for cell, count in counts.items():
                        st.metric(cell.upper(), count)

# Segmentation
elif mode == "Breast Ultrasound Segmentation":
    st.title("Breast Ultrasound Segmentation with YOLOv8")
    st.markdown("Upload a breast ultrasound image to segment masses.")

    uploaded = st.file_uploader("Upload Ultrasound Image", type=['jpg', 'jpeg', 'png'])

    if uploaded:
        image = Image.open(uploaded).convert('RGB')
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with col2:
            st.subheader("Segmentation Result")
            with st.spinner("Segmenting..."):
                model = load_model(SEG_MODEL)
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
                    image.save(f.name)
                    results = model(f.name)
                    result_img = results[0].plot()
                    os.unlink(f.name)

                result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, use_container_width=True)

                boxes = results[0].boxes
                if boxes is not None and len(boxes):
                    names = results[0].names
                    st.markdown("#### Detected Regions")
                    for i, box in enumerate(boxes):
                        label = names[int(box.cls)]
                        conf = float(box.conf)
                        st.write(f"Region {i+1}: **{label}** ({conf:.2%})")
                else:
                    st.info("No regions detected.")