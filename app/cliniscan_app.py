import streamlit as st
import pandas as pd
import altair as alt
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
from ultralytics import YOLO
import timm
from torchvision.models.feature_extraction import create_feature_extractor
import os
import gdown

# -----------------------------------------------------------------------------
# 🎨 PAGE CONFIGURATION & CSS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CliniScan AI | Lung Diagnostics",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/vasuchakravarthi/cliniscan-lung-detection',
        'Report a bug': "https://github.com/vasuchakravarthi/cliniscan-lung-detection/issues",
        'About': "CliniScan is an AI-powered tool for lung abnormality detection."
    }
)

# Custom CSS for polished "Medical Dashboard" look
st.markdown("""
<style>
    /* Main container padding */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] { font-size: 2rem; }
    
    /* Custom headers */
    h1 { color: #2C3E50; font-weight: 700; }
    h2, h3 { color: #34495E; }
    
    /* Dark mode adjustments (automatic based on theme) */
    @media (prefers-color-scheme: dark) {
        h1, h2, h3 { color: #ECF0F1; }
    }
    
    /* Info box styling */
    .info-box {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f0f2f6;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1rem;
    }
    
    /* Warning box */
    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        color: #856404;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 🔽 GOOGLE DRIVE MODEL DOWNLOAD
# -----------------------------------------------------------------------------

DETECTION_MODEL_ID = "1RN903UCBYkkY9JftW9NauOZbCdFTLc1a"
CLASSIFICATION_MODEL_ID = "1e2xHBMKshkPcaUDJSLLF-dJe2ohQIhk_"

@st.cache_resource
def download_models():
    """Download models from Google Drive if not present"""
    os.makedirs("models/detection", exist_ok=True)
    os.makedirs("models/classification", exist_ok=True)
    
    det_path = "models/detection/best.pt"
    clf_path = "models/classification/best_clf_model.pth"
    
    if not os.path.exists(det_path):
        with st.spinner("⏳ Downloading detection model (52 MB)... First run only."):
            try:
                url = f"https://drive.google.com/uc?id={DETECTION_MODEL_ID}"
                gdown.download(url, det_path, quiet=False)
            except Exception as e:
                st.error(f"❌ Error downloading detection model: {e}")
                return False, False
    
    if not os.path.exists(clf_path):
        with st.spinner("⏳ Downloading classification model (129 MB)..."):
            try:
                url = f"https://drive.google.com/uc?id={CLASSIFICATION_MODEL_ID}"
                gdown.download(url, clf_path, quiet=False)
            except Exception as e:
                st.error(f"❌ Error downloading classification model: {e}")
                return True, False
    
    return True, True

det_ready, clf_ready = download_models()

# -----------------------------------------------------------------------------
# 1️⃣ LOAD MODELS
# -----------------------------------------------------------------------------

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes, drop_rate=dropout)
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_classification_model():
    if not clf_ready: return None
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EfficientNetClassifier(num_classes=2, dropout=0.3).to(device)
        model_path = "models/classification/best_clf_model.pth"
        
        if not os.path.exists(model_path): return None
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return None

@st.cache_resource
def load_detection_model():
    if not det_ready: return None
    try:
        model_path = "models/detection/best.pt"
        if not os.path.exists(model_path): return None
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading detection model: {e}")
        return None

clf_model = load_classification_model()
det_model = load_detection_model()

# -----------------------------------------------------------------------------
# 2️⃣ GRAD-CAM UTILS
# -----------------------------------------------------------------------------

def generate_gradcam(model, img_tensor):
    if model is None: return None, None
    try:
        device = next(model.parameters()).device
        model.eval()
        feature_extractor = create_feature_extractor(model.model, {"conv_head": "feat"})
        
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(device)
            out = feature_extractor(img_tensor)
            preds = model(img_tensor)
            pred_class = preds.argmax(dim=1).item()
        
        feat_map = out["feat"].squeeze().detach().mean(dim=0).cpu().numpy()
        heatmap = cv2.resize(feat_map, (512, 512))
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0: heatmap /= np.max(heatmap)
        
        return heatmap, pred_class
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")
        return None, None

# -----------------------------------------------------------------------------
# 3️⃣ SIDEBAR & UI
# -----------------------------------------------------------------------------

with st.sidebar:
    st.image("https://img.icons8.com/color/96/lungs.png", width=80)
    st.title("CliniScan AI")
    st.caption("v1.0.0 | Beta")
    
    st.markdown("---")
    
    st.subheader("⚙️ Settings")
    conf_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.25, 0.05, help="Lower values detect more objects but may increase false positives.")
    
    st.markdown("---")
    
    with st.expander("ℹ️ About the Models"):
        st.markdown("""
        **Detection (YOLOv8-M)**
        - Trained on VinDr-CXR
        - Detects 14 classes (Aortic enlargement, Cardiomegaly, etc.)
        
        **Classification (EfficientNet-B3)**
        - Binary: Normal vs Abnormal
        - Accuracy: 95.20%
        """)
    
    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.8rem; color: gray;">
    <b>Developer:</b> Vasu Chakravarthi<br>
    SRKR Engineering College
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4️⃣ MAIN CONTENT
# -----------------------------------------------------------------------------

st.title("🩻 Lung Abnormality Analysis")
st.markdown("Upload a Chest X-Ray to detect abnormalities using **YOLOv8** and **EfficientNet**.")

if clf_model is None or det_model is None:
    st.error("⚠️ Models are missing. Please check the logs.")
    st.stop()

uploaded_file = st.file_uploader("Drop X-Ray Image Here", type=["jpg", "jpeg", "png"], help="Supported formats: JPG, PNG")

if uploaded_file:
    # --- PREPROCESSING ---
    image = Image.open(uploaded_file).convert("RGB")
    
    # Classification Transform
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Run Inference
    with st.spinner("🤖 analyzing image..."):
        # 1. Classification
        img_tensor = transform(image)
        device = next(clf_model.parameters()).device
        img_tensor = img_tensor.to(device)
        
        with torch.no_grad():
            preds = clf_model(img_tensor.unsqueeze(0))
            probs = torch.nn.functional.softmax(preds, dim=1)
            pred_class = torch.argmax(probs).item()
            confidence = probs[0][pred_class].item()
        
        # 2. Detection
        det_results = det_model.predict(np.array(image), conf=conf_threshold, verbose=False)
        det_img = det_results[0].plot()
        boxes = det_results[0].boxes

    # --- UI LAYOUT: TABS ---
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔍 Detailed Analysis", "📝 Model Details"])

    # === TAB 1: DASHBOARD (Summary) ===
    with tab1:
        col_summary, col_img = st.columns([1, 1])
        
        with col_summary:
            st.subheader("Diagnostic Summary")
            
            # Dynamic coloring based on result
            if pred_class == 0: # Abnormal
                st.error("🚨 **ABNORMALITY DETECTED**")
                metric_color = "normal" # Red in Streamlit usually implies attention
                delta_color = "inverse"
            else:
                st.success("✅ **NORMAL**")
                metric_color = "off"
                delta_color = "normal"

            # Metrics Row
            m1, m2 = st.columns(2)
            m1.metric("Classification", "Abnormal" if pred_class == 0 else "Normal", delta_color=delta_color)
            m2.metric("Confidence", f"{confidence:.1%}")
            
            st.divider()
            
            # Findings List
            st.markdown("#### 🩺 Key Findings")
            if len(boxes) > 0:
                unique_labels = set()
                for box in boxes:
                    cls_id = int(box.cls[0])
                    unique_labels.add(det_model.names[cls_id])
                
                for label in unique_labels:
                    st.markdown(f"- 🔴 **{label}**")
            else:
                if pred_class == 0:
                    st.info("No specific bounding boxes found despite 'Abnormal' classification. Check Heatmap.")
                else:
                    st.markdown("- No visible abnormalities detected.")

        with col_img:
            st.image(image, caption="Uploaded X-Ray", use_column_width=True, channels="RGB")

    # === TAB 2: DETAILED ANALYSIS (Visuals) ===
    with tab2:
        col_det, col_cam = st.columns(2)
        
        with col_det:
            st.subheader("🎯 Object Detection (YOLOv8)")
            st.image(det_img, caption=f"Detections (Threshold: {conf_threshold})", use_column_width=True)
            
            if len(boxes) > 0:
                with st.expander("View Detection Coordinates"):
                    for i, box in enumerate(boxes):
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        st.text(f"{i+1}. {det_model.names[cls_id]} | Conf: {conf:.1%}")

        with col_cam:
            st.subheader("🧠 Model Attention (Grad-CAM)")
            heatmap, _ = generate_gradcam(clf_model, img_tensor)
            
            if heatmap is not None:
                heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
                heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
                original_resized = np.array(image.resize((512, 512)))
                overlay = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)
                st.image(overlay, caption="EfficientNet Focus Areas (Red = High Attention)", use_column_width=True)
            else:
                st.warning("Grad-CAM generation failed.")

# === TAB 3: MODEL DETAILS ===
    with tab3:
        st.markdown("### Class Probabilities")
        
        # Create a specific DataFrame for the chart
        df_chart = pd.DataFrame({
            "Class": ["Abnormal", "Normal"],
            "Probability": [probs[0][0].item(), probs[0][1].item()]
        })

        # Create a custom chart with Red/Green colors
        chart = alt.Chart(df_chart).mark_bar().encode(
            x=alt.X('Class', sort=None),
            y=alt.Y('Probability', title='Probability'),
            # Define specific colors: Abnormal=Red, Normal=Green
            color=alt.Color('Class', 
                            scale=alt.Scale(domain=['Abnormal', 'Normal'], range=['#FF4B4B', '#00CC96']),
                            legend=None),
            tooltip=['Class', alt.Tooltip('Probability', format='.1%')]
        ).properties(
            height=300
        )

        st.altair_chart(chart, use_container_width=True)

        st.markdown("### System Disclaimer")
        st.warning("""
        **Disclaimer:** This tool is for educational and research purposes only (BTech Project). 
        It is **not** a substitute for professional medical advice, diagnosis, or treatment. 
        False positives/negatives may occur.
        """)
# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888;">
    CliniScan AI | Built with Streamlit, PyTorch & YOLOv8
</div>
""", unsafe_allow_html=True)
