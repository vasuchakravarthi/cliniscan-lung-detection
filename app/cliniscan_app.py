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

# Page configuration
st.set_page_config(
    page_title="🩻 CliniScan - Lung Abnormality Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# NAVIGATION STATE
# -----------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to(page):
    st.session_state.page = page
    st.rerun()


# -----------------------------------------------------------------------------
# HEADER & SIDEBAR
# -----------------------------------------------------------------------------

def show_header():
    st.title("🩻 CliniScan: AI-Powered Lung Abnormality Detection")

    st.markdown("""
    Upload a **Chest X-ray** image to:
    - 🎯 Detect **14 lung abnormalities** with bounding boxes (YOLOv8-M, mAP: 0.4305)
    - 📊 Get **overall classification**: Abnormal vs Normal (EfficientNet-B3, Acc: 95.20%)
    - 🧠 View **Grad-CAM heatmap** showing model focus areas

    **Note**: Classification trained on 512×½12 images, optimized for chest X-ray analysis.
    """)

    with st.sidebar:
        st.header("ℹ️ About CliniScan")
        st.markdown("""
        **14 Detectable Abnormalities**:
        1. Aortic enlargement
        2. Atelectasis
        3. Calcification
        4. Cardiomegaly
        5. Consolidation
        6. ILD
        7. Infiltration
        8. Lung Opacity
        9. Nodule/Mass
        10. Other lesion
        11. Pleural effusion
        12. Pleural thickening
        13. Pneumothorax
        14. Pulmonary fibrosis
        
        **Classification Classes**:
        - Abnormal (Class 0)
        - Normal (Class 1)
        
        **⚠️ Disclaimer**: Educational purposes only.
        """)

        st.markdown("---")
        st.markdown("**Developer**: Vasu Chakravarthi")
        st.markdown("**Institution**: SRKR Engineering College")
        st.markdown("[GitHub Repository](https://github.com/vasuchakravarthi/cliniscan-lung-detection)")


def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    <p><strong>⚠️ DISCLAIMER</strong></p>
    <p>This system is for <strong>educational and research purposes only</strong>.</p>
    <p>It should NOT be used for clinical diagnosis or medical decision-making.</p>
    <p>Always consult a qualified radiologist for medical interpretation of chest X-rays.</p>
    <hr>
    <p><strong>Vasu Chakravarthi</strong> | SRKR Engineering College | BTech AIML 2025</p>
    <p><a href='https://github.com/vasuchakravarthi/cliniscan-lung-detection'>GitHub Repository</a></p>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MODEL URLS (Hugging Face)
# -----------------------------------------------------------------------------

DET_URL = (
    "https://huggingface.co/vasuchakravarthi/cliniscan-models/"
    "resolve/main/best1.pt?download=true"
)

CLF_URL = (
    "https://huggingface.co/vasuchakravarthi/cliniscan-models/"
    "resolve/main/best_clf_model.pth?download=true"
)


# -----------------------------------------------------------------------------
# DOWNLOAD MODELS
# -----------------------------------------------------------------------------

@st.cache_resource
def download_models():
    """Download models from Hugging Face if not present"""
    os.makedirs("models/detection", exist_ok=True)
    os.makedirs("models/classification", exist_ok=True)

    det_path = "models/detection/best.pt"
    clf_path = "models/classification/best_clf_model.pth"

    try:
        if not os.path.exists(det_path):
            urllib.request.urlretrieve(DET_URL, det_path)

        if not os.path.exists(clf_path):
            urllib.request.urlretrieve(CLF_URL, clf_path)
    except Exception as exc:
        raise RuntimeError(
            f"Model download failed. Check the Hugging Face URLs. Details: {exc}"
        ) from exc

    if not os.path.exists(det_path) or os.path.getsize(det_path) == 0:
        raise RuntimeError(f"Detection model is missing or empty: {det_path}")

    if not os.path.exists(clf_path) or os.path.getsize(clf_path) == 0:
        raise RuntimeError(f"Classification model is missing or empty: {clf_path}")

    return det_path, clf_path


# -----------------------------------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------------------------------

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.model = timm.create_model(
            "efficientnet_b3",
            pretrained=False,
            num_classes=num_classes,
            drop_rate=dropout
        )

    def forward(self, x):
        return self.model(x)


@st.cache_resource
def load_models():
    """Load classification and detection models"""
    det_path, clf_path = download_models()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Classification model
    clf_model = EfficientNetClassifier(num_classes=2, dropout=0.3).to(device)

    checkpoint = torch.load(clf_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
        acc_val = checkpoint.get("acc", 0.0)
        st.success(f"✅ Classification model loaded! Accuracy: {acc_val * 100:.2f}%")
    else:
        state_dict = checkpoint

    # Remove possible 'module.' prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    clf_model.load_state_dict(state_dict, strict=True)
    clf_model.eval()

    # Detection model
    det_model = YOLO(det_path)

    return clf_model, det_model


# -----------------------------------------------------------------------------
# GRAD-CAM (same logic as your working version)
# -----------------------------------------------------------------------------

def generate_gradcam(model, img_tensor):
    """Generate Grad-CAM heatmap for EfficientNet-B3"""
    if model is None:
        return None, None

    try:
        device = next(model.parameters()).device
        model.eval()

        # Use conv_head feature from timm EfficientNet-B3
        feature_extractor = create_feature_extractor(
            model.model,
            {"conv_head": "feat"}
        )

        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(device)
            out = feature_extractor(img_tensor)
            preds = model(img_tensor)
            pred_class = preds.argmax(dim=1).item()

        feat_map = out["feat"].squeeze().detach().mean(dim=0).cpu().numpy()
        heatmap = cv2.resize(feat_map, (512, 512))
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)

        return heatmap, pred_class
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")
        return None, None


# -----------------------------------------------------------------------------
# PAGES
# -----------------------------------------------------------------------------

def home_page():
    show_header()

    st.subheader("🏠 Welcome to CliniScan")

    st.write("""
    **CliniScan** is an AI-powered chest X-ray analysis system designed to assist
    in detecting lung abnormalities.

    The system integrates:
    - YOLOv8 detection for lung abnormalities
    - EfficientNet classification
    - Grad-CAM explainability
    """)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔐 Login"):
            go_to("login")

    with col2:
        if st.button("🧪 Free Trial"):
            go_to("trial")

    show_footer()


def login_page():
    show_header()

    st.subheader("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "cliniscan":
            go_to("dashboard")
        else:
            st.error("Invalid credentials")

    if st.button("⬅ Back"):
        go_to("home")

    show_footer()


def trial_page():
    show_header()

    st.subheader("🧪 Free Trial")

    st.info("Upload a chest X-ray to test the AI system.")

    if st.button("Start Trial"):
        go_to("dashboard")

    if st.button("⬅ Back"):
        go_to("home")

    show_footer()


def dashboard_page():
    show_header()

    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-ray (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.subheader("📷 Uploaded X-ray")
        st.image(image, use_column_width=True)

        with st.spinner("Loading AI models (first time may take a minute)..."):
            clf_model, det_model = load_models()

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 Classification")

            transform = transforms.Compose([
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            img_tensor = transform(image)
            device = next(clf_model.parameters()).device
            img_tensor = img_tensor.to(device)

            with torch.no_grad():
                preds = clf_model(img_tensor.unsqueeze(0))
                probs = torch.nn.functional.softmax(preds, dim=1)
                pred_class = torch.argmax(probs).item()

            class_names = ["Abnormal", "Normal"]

            st.markdown(f"### Predicted: **{class_names[pred_class]}**")
            st.markdown(f"### Confidence: **{probs[0][pred_class]:.2%}**")

            st.markdown("#### Probabilities:")
            for i, name in enumerate(class_names):
                st.write(f"{name}: {probs[0][i].item():.2%}")
                st.progress(float(probs[0][i].item()))

            st.markdown("---")
            st.subheader("🧠 Grad-CAM")
            st.markdown("*Red/yellow areas show where the model focused for classification*")

            heatmap, _ = generate_gradcam(clf_model, img_tensor)

            if heatmap is not None:
                heatmap_colored = cv2.applyColorMap(
                    np.uint8(255 * heatmap),
                    cv2.COLORMAP_JET
                )
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                original_resized = np.array(image.resize((512, 512)))
                overlay = cv2.addWeighted(
                    original_resized, 0.6,
                    heatmap_colored, 0.4,
                    0
                )
                st.image(overlay, caption="Grad-CAM: Model Focus Areas", use_column_width=True)
            else:
                st.warning("Could not generate Grad-CAM")

        with col2:
            st.subheader("📦 Detection: 14 Abnormalities")

            with st.spinner("Detecting abnormalities..."):
                results = det_model.predict(np.array(image), conf=0.25, verbose=False)

            res_img = results[0].plot()
            st.image(res_img, caption="Detected Abnormalities with Bounding Boxes", use_column_width=True)

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                st.markdown("#### 🎯 Detected Abnormalities:")

                for i in range(min(5, len(boxes))):
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    st.write(f"**{i+1}. {det_model.names[cls_id]}**")
                    st.progress(conf)
                    st.write(f"Confidence: {conf:.2%}\n")

                st.markdown(f"**Total Detections**: {len(boxes)}")
                st.markdown(f"**Average Confidence**: {float(boxes.conf.mean()):.2%}")
            else:
                st.success("✅ No abnormalities detected")
                st.info("This X-ray appears normal based on the detection model.")

    if st.button("Logout"):
        go_to("home")

    show_footer()


# -----------------------------------------------------------------------------
# ROUTER
# -----------------------------------------------------------------------------

if st.session_state.page == "home":
    home_page()

elif st.session_state.page == "login":
    login_page()

elif st.session_state.page == "trial":
    trial_page()

elif st.session_state.page == "dashboard":
    dashboard_page()
