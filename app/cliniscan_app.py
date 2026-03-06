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
import gdown

st.set_page_config(
    page_title="CliniScan - Lung Detection",
    layout="wide"
)

# ----------------------------
# GOOGLE DRIVE MODEL IDS
# ----------------------------

DETECTION_MODEL_ID = "1RN903UCBYkkY9JftW9NauOZbCdFTLc1a"
CLASSIFICATION_MODEL_ID = "1e2xHBMKshkPcaUDJSLLF-dJe2ohQIhk_"

# ----------------------------
# DOWNLOAD MODELS
# ----------------------------

@st.cache_resource
def download_models():

    os.makedirs("models/detection", exist_ok=True)
    os.makedirs("models/classification", exist_ok=True)

    det_path = "models/detection/best.pt"
    clf_path = "models/classification/best_clf_model.pth"

    if not os.path.exists(det_path):
        url = f"https://drive.google.com/uc?export=download&id={DETECTION_MODEL_ID}"
        gdown.download(url, det_path, quiet=False)

    if not os.path.exists(clf_path):
        url = f"https://drive.google.com/uc?export=download&id={CLASSIFICATION_MODEL_ID}"
        gdown.download(url, clf_path, quiet=False)

    return det_path, clf_path

det_path, clf_path = download_models()

# ----------------------------
# CLASSIFIER MODEL
# ----------------------------

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

    checkpoint = torch.load(clf_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model

# ----------------------------
# DETECTION MODEL
# ----------------------------

@st.cache_resource
def load_detector():
    model = YOLO(det_path)
    return model

clf_model = load_classifier()
det_model = load_detector()

# ----------------------------
# IMAGE TRANSFORM
# ----------------------------

transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# ----------------------------
# UI
# ----------------------------

st.title("🩻 CliniScan: Lung Abnormality Detection")

st.write(
"""
Upload a **Chest X-ray image** to detect lung abnormalities using AI.
"""
)

uploaded_file = st.file_uploader(
    "Upload X-ray",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, use_container_width=True)

    img_tensor = transform(image)

    with torch.no_grad():

        preds = clf_model(img_tensor.unsqueeze(0))

        probs = torch.nn.functional.softmax(preds, dim=1)

        pred_class = torch.argmax(probs).item()

    class_names = ["Abnormal","Normal"]

    st.subheader("Classification Result")

    st.write("Prediction:", class_names[pred_class])
    st.write("Confidence:", float(probs[0][pred_class]))

    st.subheader("Detection")

    results = det_model.predict(
        source=np.array(image),
        conf=0.25,
        verbose=False
    )

    result_img = results[0].plot()

    st.image(result_img, use_container_width=True)

    if results[0].boxes is not None:

        boxes = results[0].boxes

        st.write("Detections:", len(boxes))

    else:

        st.success("No abnormalities detected")

st.markdown("---")

st.write("Developer: Vasu Chakravarthi")
