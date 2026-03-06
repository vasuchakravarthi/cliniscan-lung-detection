import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from ultralytics import YOLO
import timm
from torchvision.models.feature_extraction import create_feature_extractor
import os
import urllib.request

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="🩻 CliniScan - Lung Detection",
    layout="wide"
)

# --------------------------------------------------
# Navigation state
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "login"


def go_to(page):
    st.session_state.page = page
    st.rerun()


# --------------------------------------------------
# Model URLs (HuggingFace)
# --------------------------------------------------

DET_URL = "https://huggingface.co/vasuchakravarthi/cliniscan-models/resolve/main/best1.pt"
CLF_URL = "https://huggingface.co/vasuchakravarthi/cliniscan-models/resolve/main/best_clf_model.pth"


# --------------------------------------------------
# Download Models
# --------------------------------------------------

@st.cache_resource
def download_models():

    os.makedirs("models", exist_ok=True)

    det_path = "models/best.pt"
    clf_path = "models/best_clf_model.pth"

    if not os.path.exists(det_path):
        urllib.request.urlretrieve(DET_URL, det_path)

    if not os.path.exists(clf_path):
        urllib.request.urlretrieve(CLF_URL, clf_path)

    return det_path, clf_path


det_path, clf_path = download_models()


# --------------------------------------------------
# Classification Model
# --------------------------------------------------

class EfficientNetClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=2
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
    return YOLO(det_path)


clf_model = load_classifier()
det_model = load_detector()


# --------------------------------------------------
# GradCAM
# --------------------------------------------------

def generate_gradcam(model, img_tensor):

    feature_extractor = create_feature_extractor(
        model.model,
        {"conv_head": "feat"}
    )

    with torch.no_grad():
        img_tensor = img_tensor.unsqueeze(0)
        features = feature_extractor(img_tensor)

    feat_map = features["feat"].squeeze().mean(dim=0).cpu().numpy()

    heatmap = cv2.resize(feat_map, (512,512))
    heatmap = np.maximum(heatmap,0)

    if heatmap.max() != 0:
        heatmap /= heatmap.max()

    return heatmap


# --------------------------------------------------
# Image transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

def login_page():

    st.title("🩻 CliniScan AI")

    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Login"):

            if username == "admin" and password == "cliniscan":
                go_to("dashboard")
            else:
                st.error("Invalid credentials")

    with col2:
        if st.button("Free Trial"):
            go_to("trial")


# --------------------------------------------------
# FREE TRIAL PAGE
# --------------------------------------------------

def trial_page():

    st.title("🧪 Free Trial")

    st.info("You can test CliniScan with one X-ray image.")

    if st.button("Start Trial"):
        go_to("dashboard")

    if st.button("Back"):
        go_to("login")


# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

def dashboard_page():

    st.title("🩻 CliniScan: AI Lung Abnormality Detection")

    with st.sidebar:

        st.header("Detected Abnormalities")

        abnormalities = [
            "Aortic enlargement",
            "Atelectasis",
            "Calcification",
            "Cardiomegaly",
            "Consolidation",
            "ILD",
            "Infiltration",
            "Lung Opacity",
            "Nodule/Mass",
            "Other lesion",
            "Pleural effusion",
            "Pleural thickening",
            "Pneumothorax",
            "Pulmonary fibrosis"
        ]

        for a in abnormalities:
            st.write("•", a)

        st.markdown("---")

        if st.button("Logout"):
            go_to("login")

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray",
        type=["jpg","png","jpeg"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        img_tensor = transform(image)

        col1, col2 = st.columns(2)

        # -------------------------
        # Classification
        # -------------------------

        with col1:

            st.subheader("Classification")

            with torch.no_grad():

                preds = clf_model(img_tensor.unsqueeze(0))
                probs = torch.nn.functional.softmax(preds, dim=1)

                pred_class = torch.argmax(probs).item()

            classes = ["Abnormal","Normal"]

            st.write("Prediction:", classes[pred_class])
            st.write("Confidence:", f"{probs[0][pred_class]:.2%}")

            # GradCAM

            st.subheader("Grad-CAM")

            heatmap = generate_gradcam(clf_model, img_tensor)

            heatmap = cv2.applyColorMap(
                np.uint8(255*heatmap),
                cv2.COLORMAP_JET
            )

            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            original = np.array(image.resize((512,512)))

            overlay = cv2.addWeighted(original,0.6,heatmap,0.4,0)

            st.image(overlay)

        # -------------------------
        # Detection
        # -------------------------

        with col2:

            st.subheader("Detection")

            results = det_model.predict(
                source=np.array(image),
                conf=0.25,
                verbose=False
            )

            res_img = results[0].plot()

            st.image(res_img, use_container_width=True)

            if results[0].boxes is not None:

                boxes = results[0].boxes

                st.write("Total detections:", len(boxes))

                for i in range(len(boxes)):

                    cls = int(boxes.cls[i])
                    conf = float(boxes.conf[i])

                    st.write(
                        f"{det_model.names[cls]} - {conf:.2%}"
                    )

            else:
                st.success("No abnormalities detected")


# --------------------------------------------------
# ROUTER
# --------------------------------------------------

if st.session_state.page == "login":
    login_page()

elif st.session_state.page == "trial":
    trial_page()

elif st.session_state.page == "dashboard":
    dashboard_page()
