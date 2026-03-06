import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from ultralytics import YOLO
import timm
import os
import urllib.request

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CliniScan - Lung Abnormality Detection",
    layout="wide"
)

# --------------------------------------------------
# HuggingFace Model URLs
# --------------------------------------------------

DET_URL = "https://huggingface.co/vasuchakravarthi/cliniscan-models/resolve/main/best.pt"
CLF_URL = "https://huggingface.co/vasuchakravarthi/cliniscan-models/resolve/main/best_clf_model.pth"

# --------------------------------------------------
# Download Models
# --------------------------------------------------

@st.cache_resource
def download_models():

    os.makedirs("models/detection", exist_ok=True)
    os.makedirs("models/classification", exist_ok=True)

    det_path = "models/detection/best.pt"
    clf_path = "models/classification/best_clf_model.pth"

    if not os.path.exists(det_path):
        with st.spinner("Downloading detection model..."):
            urllib.request.urlretrieve(DET_URL, det_path)

    if not os.path.exists(clf_path):
        with st.spinner("Downloading classification model..."):
            urllib.request.urlretrieve(CLF_URL, clf_path)

    return det_path, clf_path


det_path, clf_path = download_models()

# --------------------------------------------------
# Classification Model
# --------------------------------------------------

class EfficientNetClassifier(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.model = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


@st.cache_resource
def load_classifier():

    device = torch.device("cpu")

    model = EfficientNetClassifier()

    checkpoint = torch.load(
        clf_path,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


# --------------------------------------------------
# Detection Model
# --------------------------------------------------

@st.cache_resource
def load_detector():

    model = YOLO(det_path)

    return model


clf_model = load_classifier()
det_model = load_detector()

# --------------------------------------------------
# Image Preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# --------------------------------------------------
# UI
# --------------------------------------------------

st.title("🩻 CliniScan: AI Lung Abnormality Detection")

st.markdown(
"""
Upload a **Chest X-ray image** to:

• Detect **14 lung abnormalities** using YOLOv8  
• Classify image as **Normal / Abnormal** using EfficientNet-B3
"""
)

uploaded_file = st.file_uploader(
    "Upload Chest X-ray",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    img_tensor = transform(image)

    # --------------------------------------------------
    # Classification
    # --------------------------------------------------

    st.subheader("Classification Result")

    with torch.no_grad():

        preds = clf_model(img_tensor.unsqueeze(0))

        probs = torch.nn.functional.softmax(preds, dim=1)

        pred_class = torch.argmax(probs).item()

    class_names = ["Abnormal", "Normal"]

    st.write(
        f"Prediction: **{class_names[pred_class]}**"
    )

    st.write(
        f"Confidence: **{probs[0][pred_class].item():.2%}**"
    )

    # --------------------------------------------------
    # Detection
    # --------------------------------------------------

    st.subheader("Detected Abnormalities")

    with st.spinner("Running detection..."):

        results = det_model.predict(
            source=np.array(image),
            conf=0.25,
            verbose=False
        )

    result_img = results[0].plot()

    st.image(
        result_img,
        caption="Detection Results",
        use_container_width=True
    )

    if results[0].boxes is not None:

        boxes = results[0].boxes

        st.write("Total detections:", len(boxes))

        for i in range(len(boxes)):

            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])

            st.write(
                f"{det_model.names[cls_id]} - {conf:.2%}"
            )

    else:

        st.success("No abnormalities detected")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.markdown(
"""
⚠️ **Disclaimer**

This system is for **educational and research purposes only**.  
It should **not be used for clinical diagnosis**.

Developer: **Vasu Chakravarthi**  
SRKR Engineering College
"""
)
