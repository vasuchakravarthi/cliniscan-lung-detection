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
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="🩻 CliniScan - Lung Detection",
    layout="wide"
)

# --------------------------------------------------
# NAVIGATION STATE
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"


def go_to(page):
    st.session_state.page = page
    st.rerun()


# --------------------------------------------------
# HEADER (TITLE + SIDEBAR)
# --------------------------------------------------

def show_header():

    st.title("🩻 CliniScan: AI-Powered Lung Abnormality Detection")

    st.markdown("""
    Upload a **Chest X-ray** image to:

    - 🎯 Detect **14 lung abnormalities** with bounding boxes (YOLOv8-M, mAP: 0.4305)
    - 📊 Get **overall classification**: Abnormal vs Normal (EfficientNet-B3, Acc: 95.20%)
    - 🧠 View **Grad-CAM heatmap** showing model focus areas

    **Note**: Classification trained on 512×512 images, optimized for chest X-ray analysis.
    """)

    with st.sidebar:

        st.header("ℹ️ About CliniScan")

        st.markdown("""
        **14 Detectable Abnormalities**

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

        **Classification Classes**

        - Abnormal (Class 0)
        - Normal (Class 1)

        **⚠️ Disclaimer**: Educational purposes only.
        """)

        st.markdown("---")
        st.markdown("**Developer**: Vasu Chakravarthi")
        st.markdown("**Institution**: SRKR Engineering College")

        st.markdown(
            "[GitHub Repository](https://github.com/vasuchakravarthi/cliniscan-lung-detection)"
        )


# --------------------------------------------------
# FOOTER (DISCLAIMER)
# --------------------------------------------------

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

    <p>
    <a href='https://github.com/vasuchakravarthi/cliniscan-lung-detection'>
    GitHub Repository
    </a>
    </p>

    </div>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# MODEL URLS
# --------------------------------------------------

DET_URL = "https://huggingface.co/vasuchakravarthi/cliniscan-models/resolve/main/best1.pt"
CLF_URL = "https://huggingface.co/vasuchakravarthi/cliniscan-models/resolve/main/best_clf_model.pth"


# --------------------------------------------------
# DOWNLOAD MODELS
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
# CLASSIFICATION MODEL
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

    model = EfficientNetClassifier()

    checkpoint = torch.load(
        clf_path,
        map_location="cpu",
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    return model


# --------------------------------------------------
# DETECTION MODEL
# --------------------------------------------------

@st.cache_resource
def load_detector():
    return YOLO(det_path)


clf_model = load_classifier()
det_model = load_detector()


# --------------------------------------------------
# GRADCAM
# --------------------------------------------------

def generate_gradcam(model, img_tensor):

    extractor = create_feature_extractor(
        model.model,
        {"conv_head": "feat"}
    )

    with torch.no_grad():

        img_tensor = img_tensor.unsqueeze(0)

        features = extractor(img_tensor)

    fmap = features["feat"].squeeze().mean(dim=0).cpu().numpy()

    heatmap = cv2.resize(fmap, (512,512))
    heatmap = np.maximum(heatmap,0)

    if heatmap.max()!=0:
        heatmap /= heatmap.max()

    return heatmap


# --------------------------------------------------
# IMAGE TRANSFORM
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
# HOME PAGE
# --------------------------------------------------

def home_page():

    show_header()

    st.subheader("🏠 Welcome to CliniScan")

    st.write("""
    **CliniScan** is an AI-powered chest X-ray analysis system designed to assist
    in detecting lung abnormalities.

    The system integrates:

    • YOLOv8 detection for lung abnormalities  
    • EfficientNet classification  
    • Grad-CAM explainability
    """)

    col1,col2 = st.columns(2)

    with col1:
        if st.button("🔐 Login"):
            go_to("login")

    with col2:
        if st.button("🧪 Free Trial"):
            go_to("trial")

    show_footer()


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

def login_page():

    show_header()

    st.subheader("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username=="admin" and password=="cliniscan":
            go_to("dashboard")
        else:
            st.error("Invalid credentials")

    if st.button("⬅ Back"):
        go_to("home")

    show_footer()


# --------------------------------------------------
# TRIAL PAGE
# --------------------------------------------------

def trial_page():

    show_header()

    st.subheader("🧪 Free Trial")

    st.info("Upload a chest X-ray to test the AI system.")

    if st.button("Start Trial"):
        go_to("dashboard")

    if st.button("⬅ Back"):
        go_to("home")

    show_footer()


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

def dashboard_page():

    show_header()

    uploaded_file = st.file_uploader(
        "Upload Chest X-ray",
        type=["jpg","jpeg","png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        img_tensor = transform(image)

        col1,col2 = st.columns(2)

        # Classification

        with col1:

            st.subheader("Classification")

            with torch.no_grad():

                preds = clf_model(img_tensor.unsqueeze(0))
                probs = torch.nn.functional.softmax(preds,dim=1)

                pred_class = torch.argmax(probs).item()

            classes=["Abnormal","Normal"]

            st.write("Prediction:",classes[pred_class])
            st.write("Confidence:",f"{probs[0][pred_class]:.2%}")

            st.subheader("Grad-CAM")

            heatmap = generate_gradcam(clf_model,img_tensor)

            heatmap=cv2.applyColorMap(
                np.uint8(255*heatmap),
                cv2.COLORMAP_JET
            )

            heatmap=cv2.cvtColor(heatmap,cv2.COLOR_BGR2RGB)

            original=np.array(image.resize((512,512)))

            overlay=cv2.addWeighted(original,0.6,heatmap,0.4,0)

            st.image(overlay)

        # Detection

        with col2:

            st.subheader("Detection")

            results = det_model.predict(
                source=np.array(image),
                conf=0.25,
                verbose=False
            )

            res_img = results[0].plot()

            st.image(res_img,use_container_width=True)

            if results[0].boxes is not None:

                boxes=results[0].boxes

                st.write("Total detections:",len(boxes))

                for i in range(len(boxes)):

                    cls=int(boxes.cls[i])
                    conf=float(boxes.conf[i])

                    st.write(
                        f"{det_model.names[cls]} - {conf:.2%}"
                    )

    if st.button("Logout"):
        go_to("home")

    show_footer()


# --------------------------------------------------
# ROUTER
# --------------------------------------------------

if st.session_state.page=="home":
    home_page()

elif st.session_state.page=="login":
    login_page()


elif st.session_state.page=="trial":
    trial_page()

elif st.session_state.page=="dashboard":
    dashboard_page()
    
#can we make these more interactive website it is hosted from streamlit
