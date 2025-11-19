import streamlit as st
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
import pandas as pd

# -----------------------------------------------------------------------------
# ⚙️ CONFIGURATION & STYLING
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="CliniScan AI | Radiologist Assistant",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Medical UI
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Header Styling */
    h1 {
        color: #2c3e50;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
    }
    h2, h3 {
        color: #34495e;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    /* Custom Cards */
    .css-1r6slb0 {
        background-color: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
    }
    
    /* Metric Styling */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #2980b9;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #2c3e50;
    }
    [data-testid="stSidebar"] * {
        color: #ecf0f1 !important;
    }
    
    /* Custom Alert Boxes */
    .diagnosis-box-normal {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        text-align: center;
        margin-bottom: 10px;
    }
    .diagnosis-box-abnormal {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        text-align: center;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 🔽 GOOGLE DRIVE MODEL DOWNLOAD
# -----------------------------------------------------------------------------

# Replace these with your actual Google Drive file IDs
DETECTION_MODEL_ID = "1RN903UCBYkkY9JftW9NauOZbCdFTLc1a"
CLASSIFICATION_MODEL_ID = "1e2xHBMKshkPcaUDJSLLF-dJe2ohQIhk_"

@st.cache_resource
def download_models():
    """Download models from Google Drive if not present"""
    os.makedirs("models/detection", exist_ok=True)
    os.makedirs("models/classification", exist_ok=True)
    
    det_path = "models/detection/best.pt"
    clf_path = "models/classification/best_clf_model.pth"
    
    # Download detection model
    if not os.path.exists(det_path):
        with st.spinner("⏳ Initializing System: Downloading detection model (52 MB)..."):
            try:
                url = f"https://drive.google.com/uc?id={DETECTION_MODEL_ID}"
                gdown.download(url, det_path, quiet=False)
            except Exception as e:
                st.error(f"❌ Detection Model Error: {e}")
                return False, False
    
    # Download classification model
    if not os.path.exists(clf_path):
        with st.spinner("⏳ Initializing System: Downloading classification model (129 MB)..."):
            try:
                url = f"https://drive.google.com/uc?id={CLASSIFICATION_MODEL_ID}"
                gdown.download(url, clf_path, quiet=False)
            except Exception as e:
                st.error(f"❌ Classification Model Error: {e}")
                return True, False
    
    return True, True

det_ready, clf_ready = download_models()

# -----------------------------------------------------------------------------
# 1️⃣ Load Models
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
# 2️⃣ Logic Functions
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
    except Exception:
        return None, None

# -----------------------------------------------------------------------------
# 3️⃣ UI Layout
# -----------------------------------------------------------------------------

# Sidebar
with st.sidebar:
    st.title("🩻 CliniScan AI")
    st.caption("v2.1.0 | Pulmonary Analysis")
    st.markdown("---")
    
    st.markdown("#### ⚙️ Model Specifications")
    st.info(
        """
        **Binary Classifier:**
        EfficientNet-B3 (Acc: 95.20%)
        
        **Object Detector:**
        YOLOv8-Medium (mAP: 0.43)
        """
    )
    
    st.markdown("#### 📋 Detectable Pathologies")
    with st.expander("View Full List"):
        st.markdown("""
        - Aortic enlargement
        - Atelectasis
        - Calcification
        - Cardiomegaly
        - Consolidation
        - ILD
        - Infiltration
        - Lung Opacity
        - Nodule/Mass
        - Pleural effusion
        - Pleural thickening
        - Pneumothorax
        - Pulmonary fibrosis
        """)
    
    st.markdown("---")
    st.markdown("© 2024 SRKR Engineering College")
    st.caption("Developer: Vasu Chakravarthi")

# Main Content
st.markdown("# 🫁 Chest X-Ray Diagnostic Assistant")
st.markdown("### Upload patient radiograph for automated screening")

# Upload Section Container
with st.container():
    uploaded_file = st.file_uploader(
        "", 
        type=["jpg", "jpeg", "png"], 
        help="Supported formats: JPG, PNG. Max size 10MB."
    )

if uploaded_file:
    if clf_model is None or det_model is None:
        st.error("⚠️ System Error: Models failed to initialize.")
        st.stop()

    # Process Image
    image = Image.open(uploaded_file).convert("RGB")
    
    # UI Layout for Analysis
    st.markdown("---")
    st.markdown("## 📊 Analysis Report")
    
    # Create a layout with columns
    col_left, col_right = st.columns([1, 1.5], gap="large")

    # --- LEFT COLUMN: CLASSIFICATION & GRAD-CAM ---
    with col_left:
        st.markdown("### 🔎 Preliminary Screening")
        
        # Preprocessing
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
        
        # 0=Abnormal, 1=Normal
        class_names = ["Abnormal", "Normal"]
        confidence = probs[0][pred_class].item()
        
        # Display Diagnosis Badge
        if pred_class == 0: # Abnormal
            st.markdown(f"""
                <div class="diagnosis-box-abnormal">
                    <h3>DETECTED: ABNORMAL</h3>
                    <p>Confidence: {confidence:.2%}</p>
                </div>
            """, unsafe_allow_html=True)
        else: # Normal
            st.markdown(f"""
                <div class="diagnosis-box-normal">
                    <h3>RESULT: NORMAL</h3>
                    <p>Confidence: {confidence:.2%}</p>
                </div>
            """, unsafe_allow_html=True)

        # Probability Metrics
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Prob. Normal", f"{probs[0][1].item():.1%}")
        with m2:
            st.metric("Prob. Abnormal", f"{probs[0][0].item():.1%}")
            
        st.markdown("#### 🧠 AI Focus Area (Grad-CAM)")
        heatmap, _ = generate_gradcam(clf_model, img_tensor)
        
        if heatmap is not None:
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            original_resized = np.array(image.resize((512, 512)))
            overlay = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)
            st.image(overlay, caption="Heatmap indicates regions influencing the diagnosis", use_column_width=True)
        else:
            st.warning("Grad-CAM generation unavailable.")

    # --- RIGHT COLUMN: DETECTION ---
    with col_right:
        st.markdown("### 📍 Localized Findings")
        
        with st.spinner("Running object detection..."):
            results = det_model.predict(np.array(image), conf=0.25, verbose=False)
        
        res_img = results[0].plot()
        st.image(res_img, caption="Bounding Box Analysis", use_column_width=True)
        
        # Detailed Findings Table
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            # Prepare data for DataFrame
            findings = []
            for i in range(len(boxes)):
                findings.append({
                    "Pathology": det_model.names[int(boxes.cls[i])],
                    "Confidence": f"{float(boxes.conf[i]):.2%}",
                    "Location": "See Image" 
                })
            
            df = pd.DataFrame(findings)
            st.markdown("#### Detection Summary")
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No specific localized abnormalities detected by the YOLO model.")

else:
    # Placeholder when no image is uploaded
    st.markdown(
        """
        <div style='padding: 50px; text-align: center; background-color: white; border-radius: 10px; border: 1px dashed #ccc; color: #666;'>
            <h3>👋 Welcome to CliniScan</h3>
            <p>Please upload a chest X-ray image from the sidebar or above to begin analysis.</p>
            <small>Supported formats: JPG, PNG</small>
        </div>
        """, 
        unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; font-size: 12px; color: #95a5a6;'>
        <p><strong>⚠️ MEDICAL DISCLAIMER</strong></p>
        <p>This tool is a Clinical Decision Support System (CDSS) prototype intended for <strong>research and educational use only</strong>.</p>
        <p>Results should not be used for primary diagnosis. Always consult a certified radiologist.</p>
    </div>
    """, 
    unsafe_allow_html=True
)
