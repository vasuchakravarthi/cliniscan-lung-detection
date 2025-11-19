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

# -----------------------------------------------------------------------------
# 🎨 CUSTOM STYLING
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="🩻 CliniScan - AI Lung Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for vibrant, modern medical UI
st.markdown("""
<style>
    /* Main background gradient - Soft Clinical White with hint of blue */
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #e8f4f8 50%, #f0f7ff 100%);
    }
    
    /* Header styling - Medical Blue gradient */
    .main-header {
        background: linear-gradient(135deg, #0077be 0%, #005a8d 50%, #003d5c 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,119,190,0.3);
        margin-bottom: 2rem;
    }
    
    /* Stats cards - Distinct professional colors */
    .stat-card {
        background: linear-gradient(135deg, #0077be 0%, #005a8d 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,119,190,0.25);
        transition: transform 0.3s ease;
        margin: 0.5rem 0;
    }
    
    .stat-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,119,190,0.35);
    }
    
    .stat-card.purple {
        background: linear-gradient(135deg, #6c63ff 0%, #5a52d5 100%);
        box-shadow: 0 8px 20px rgba(108,99,255,0.25);
    }
    
    .stat-card.cyan {
        background: linear-gradient(135deg, #00b4d8 0%, #0096c7 100%);
        box-shadow: 0 8px 20px rgba(0,180,216,0.25);
    }
    
    .stat-card.green {
        background: linear-gradient(135deg, #06d6a0 0%, #04b589 100%);
        box-shadow: 0 8px 20px rgba(6,214,160,0.25);
    }
    
    /* Alert boxes */
    .info-alert {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 5px solid #0077be;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,119,190,0.15);
    }
    
    .warning-alert {
        background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%);
        border-left: 5px solid #ff9800;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(255,152,0,0.15);
    }
    
    /* Result cards */
    .result-card {
        background: white;
        padding: 2rem;
        border-radius: 1rem;
        box-shadow: 0 6px 24px rgba(0,0,0,0.08);
        margin: 1rem 0;
        border: 2px solid #e0e7ff;
    }
    
    .result-card.classification {
        border-color: #6c63ff;
        background: linear-gradient(135deg, #ffffff 0%, #f3f2ff 100%);
    }
    
    .result-card.detection {
        border-color: #00b4d8;
        background: linear-gradient(135deg, #ffffff 0%, #e8f8fb 100%);
    }
    
    .result-card h2 {
        color: #0077be;
        margin-top: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Badges - Modern medical status colors */
    .badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    
    .badge.normal {
        background: linear-gradient(135deg, #06d6a0 0%, #04b589 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(6,214,160,0.3);
    }
    
    .badge.abnormal {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(255,107,107,0.3);
    }
    
    /* Confidence display */
    .confidence-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
        border: 2px solid #0077be;
    }
    
    .confidence-box .value {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #0077be 0%, #005a8d 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Detection list */
    .detection-item {
        background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 0.75rem 0;
        border: 2px solid #00b4d8;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 12px rgba(0,180,216,0.12);
    }
    
    .detection-item:hover {
        border-color: #0096c7;
        box-shadow: 0 6px 18px rgba(0,180,216,0.2);
        transform: translateY(-2px);
        transition: all 0.3s ease;
    }
    
    .detection-number {
        background: linear-gradient(135deg, #00b4d8 0%, #0096c7 100%);
        color: white;
        width: 2.5rem;
        height: 2.5rem;
        border-radius: 0.5rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        margin-right: 1rem;
    }
    
    /* Disclaimer - Medical Warning Red */
    .disclaimer {
        background: linear-gradient(135deg, #ffebee 0%, #ffe5e5 100%);
        border-left: 8px solid #ff6b6b;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 6px 24px rgba(255,107,107,0.15);
    }
    
    .disclaimer .icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f7ff 100%);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #0077be 0%, #005a8d 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin-bottom: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,119,190,0.2);
    }
    
    .sidebar-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border: 2px solid #00b4d8;
        box-shadow: 0 4px 12px rgba(0,180,216,0.1);
    }
    
    .sidebar-card h3 {
        color: #0077be;
        margin-top: 0;
    }
    
    /* Gradcam section */
    .gradcam-section {
        background: linear-gradient(135deg, #ffffff 0%, #f3f2ff 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 2px solid #6c63ff;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(108,99,255,0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #0077be;
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
    
    if not os.path.exists(det_path):
        with st.spinner("⏳ Downloading detection model (52 MB)... First run only."):
            try:
                url = f"https://drive.google.com/uc?id={DETECTION_MODEL_ID}"
                gdown.download(url, det_path, quiet=False)
                st.success("✅ Detection model downloaded!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return False, False
    
    if not os.path.exists(clf_path):
        with st.spinner("⏳ Downloading classification model (129 MB)..."):
            try:
                url = f"https://drive.google.com/uc?id={CLASSIFICATION_MODEL_ID}"
                gdown.download(url, clf_path, quiet=False)
                st.success("✅ Classification model downloaded!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return True, False
    
    return True, True

det_ready, clf_ready = download_models()

# -----------------------------------------------------------------------------
# 🤖 MODEL DEFINITIONS
# -----------------------------------------------------------------------------

class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes, drop_rate=dropout)
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_classification_model():
    if not clf_ready:
        return None
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = EfficientNetClassifier(num_classes=2, dropout=0.3).to(device)
        model_path = "models/classification/best_clf_model.pth"
        
        if not os.path.exists(model_path):
            st.error("⚠️ Model file not found")
            return None
        
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
    if not det_ready:
        return None
    
    try:
        model_path = "models/detection/best.pt"
        if not os.path.exists(model_path):
            st.error("⚠️ Model file not found")
            return None
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error: {e}")
        return None

clf_model = load_classification_model()
det_model = load_detection_model()

# -----------------------------------------------------------------------------
# 🔥 GRAD-CAM
# -----------------------------------------------------------------------------

def generate_gradcam(model, img_tensor):
    if model is None:
        return None, None
    
    try:
        device = next(model.parameters()).device
        model.eval()
        
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
# 🎨 UI COMPONENTS
# -----------------------------------------------------------------------------

# Header
st.markdown("""
<div class="main-header">
    <h1>🩻 CliniScan: AI-Powered Lung Abnormality Detection</h1>
    <p>Upload a <strong>Chest X-ray</strong> image to detect abnormalities and classify diagnosis</p>
</div>
""", unsafe_allow_html=True)

# Stats Cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="stat-card cyan">
        <div class="emoji">🎯</div>
        <h3>14 Abnormalities</h3>
        <p>YOLOv8-M Detection (mAP: 0.4305)</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="emoji">📊</div>
        <h3>95.20% Accuracy</h3>
        <p>EfficientNet-B3 Classification</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card purple">
        <div class="emoji">🧠</div>
        <h3>Grad-CAM</h3>
        <p>Visual AI Focus Areas</p>
    </div>
    """, unsafe_allow_html=True)

# Info Alert
st.markdown("""
<div class="info-alert">
    <strong>ℹ️ Note:</strong> Classification trained on 512×512 images, optimized for chest X-ray analysis.
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2>ℹ️ About CliniScan</h2>
        <p>Advanced AI diagnostic assistance</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card">
        <h3>🩺 14 Detectable Abnormalities</h3>
        <ol style="font-size: 0.9rem; line-height: 1.8;">
            <li>Aortic enlargement</li>
            <li>Atelectasis</li>
            <li>Calcification</li>
            <li>Cardiomegaly</li>
            <li>Consolidation</li>
            <li>ILD</li>
            <li>Infiltration</li>
            <li>Lung Opacity</li>
            <li>Nodule/Mass</li>
            <li>Other lesion</li>
            <li>Pleural effusion</li>
            <li>Pleural thickening</li>
            <li>Pneumothorax</li>
            <li>Pulmonary fibrosis</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="sidebar-card">
        <h3>📋 Classification Classes</h3>
        <div style="margin: 1rem 0;">
            <div style="display: flex; align-items: center; gap: 0.5rem; margin: 0.5rem 0; padding: 0.5rem; background: #ffebee; border-radius: 0.5rem; border: 1px solid #ef5350;">
                <div style="width: 12px; height: 12px; background: #ef5350; border-radius: 50%;"></div>
                <strong>Abnormal</strong> (Class 0)
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem; margin: 0.5rem 0; padding: 0.5rem; background: #e8f5e9; border-radius: 0.5rem; border: 1px solid #66bb6a;">
                <div style="width: 12px; height: 12px; background: #66bb6a; border-radius: 50%;"></div>
                <strong>Normal</strong> (Class 1)
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="warning-alert">
        <strong>⚠️ Disclaimer:</strong> Educational purposes only.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("Developer: Vasu Chakravarthi")
    st.markdown("Institution: SRKR Engineering College")
    st.markdown("[🔗 GitHub Repository](https://github.com/vasuchakravarthi/cliniscan-lung-detection)")

# File Upload
st.markdown("<br>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload Chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    st.markdown("<h2 style='color: #1976d2;'>📷 Uploaded X-ray</h2>", unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    
    if clf_model is None or det_model is None:
        st.error("⚠️ Models not loaded. Check Google Drive file IDs.")
        st.stop()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # === CLASSIFICATION ===
    with col1:
        st.markdown("""
        <div class="result-card classification">
            <h2>🧠 AI Classification</h2>
        </div>
        """, unsafe_allow_html=True)
        
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
        badge_class = "normal" if pred_class == 1 else "abnormal"
        
        st.markdown(f"""
        <div style="text-align: center; margin: 1.5rem 0;">
            <p style="margin: 0; color: #666;">Predicted Class:</p>
            <div class="badge {badge_class}">{class_names[pred_class]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="confidence-box">
            <p style="margin: 0; color: #1976d2; font-weight: 600;">Confidence Score</p>
            <div class="value">{probs[0][pred_class]:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 style='color: #1976d2; margin-top: 2rem;'>📊 Probability Distribution</h3>", unsafe_allow_html=True)
        for i, name in enumerate(class_names):
            prob_val = float(probs[0][i].item())
            color = "#66bb6a" if i == 1 else "#ef5350"
            st.markdown(f"""
            <div style="margin: 1rem 0;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 12px; height: 12px; background: {color}; border-radius: 50%;"></div>
                        <strong>{name}</strong>
                    </span>
                    <span style="background: #f5f5f5; padding: 0.25rem 0.75rem; border-radius: 1rem; font-weight: 600;">{prob_val:.2%}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob_val)
        
        # Grad-CAM
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="gradcam-section">
            <h3 style='color: #673ab7; margin-top: 0;'>👁️ Grad-CAM Visualization</h3>
            <p style='color: #666; font-style: italic; font-size: 0.9rem;'>🔥 Heatmap highlights critical regions the AI model analyzed for diagnosis</p>
        </div>
        """, unsafe_allow_html=True)
        
        heatmap, _ = generate_gradcam(clf_model, img_tensor)
        
        if heatmap is not None:
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            original_resized = np.array(image.resize((512, 512)))
            overlay = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)
            st.image(overlay, caption="Grad-CAM: Model Focus Areas", use_container_width=True)
        else:
            st.warning("Could not generate Grad-CAM")
    
    # === DETECTION ===
    with col2:
        st.markdown("""
        <div class="result-card detection">
            <h2>🎯 Object Detection</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🔍 Detecting abnormalities..."):
            results = det_model.predict(np.array(image), conf=0.25, verbose=False)
        
        res_img = results[0].plot()
        st.image(res_img, caption="Detected Abnormalities with Bounding Boxes", use_container_width=True)
        
        st.markdown("<p style='text-align: center; color: #666; font-size: 0.85rem; margin: 1rem 0;'>📦 YOLOv8-M Detection Model (mAP: 0.4305)</p>", unsafe_allow_html=True)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); padding: 1rem; border-radius: 0.75rem; border-left: 5px solid #f57c00; margin: 1.5rem 0;">
                <h3 style="color: #e65100; margin: 0;">🎯 Detected Abnormalities</h3>
            </div>
            """, unsafe_allow_html=True)
            
            for i in range(min(5, len(boxes))):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                
                st.markdown(f"""
                <div class="detection-item">
                    <div style="display: flex; align-items: center; flex: 1;">
                        <div class="detection-number">{i+1}</div>
                        <strong>{det_model.names[cls_id]}</strong>
                    </div>
                    <div style="background: white; padding: 0.5rem 1rem; border-radius: 0.5rem; font-weight: 600; border: 1px solid #00bcd4;">
                        {conf:.2%}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(conf)
            
            # Summary metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ede7f6 0%, #d1c4e9 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; border: 2px solid #673ab7;">
                    <p style="margin: 0; color: #4a148c; font-size: 0.85rem;">Total Detections</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700; color: #673ab7;">{len(boxes)}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col_b:
                avg_conf = float(boxes.conf.mean())
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; border: 2px solid #00bcd4;">
                    <p style="margin: 0; color: #006064; font-size: 0.85rem;">Avg Confidence</p>
                    <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700; color: #00acc1;">{avg_conf:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 2rem; border-radius: 1rem; text-align: center; border: 2px solid #66bb6a; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
                <div style="font-size: 3rem; margin-bottom: 1rem;">✅</div>
                <h3 style="color: #2e7d32; margin: 0.5rem 0;">No abnormalities detected</h3>
                <p style="color: #388e3c; margin: 0;">This X-ray appears normal based on the detection model. All systems clear!</p>
            </div>
            """, unsafe_allow_html=True)

# Disclaimer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="disclaimer">
    <div class="icon">⚠️</div>
    <h2 style="color: #c62828; margin: 1rem 0;">MEDICAL DISCLAIMER</h2>
    <p style="color: #d32f2f; font-size: 1.1rem; margin: 1rem 0;">
        This system is for <strong>educational and research purposes only</strong>.
    </p>
    <p style="color: #e53935; margin: 1rem 0;">
        It should NOT be used for clinical diagnosis or medical decision-making.<br>
        Always consult a qualified radiologist for medical interpretation of chest X-rays.
    </p>
    <hr style="border-color: #ef9a9a; margin: 2rem 0;">
    <p style="color: #424242; margin: 0.5rem 0;"><strong>Vasu Chakravarthi</strong> | SRKR Engineering College | BTech AIML 2025</p>
    <p style="margin: 0.5rem 0;">
        <a href="https://github.com/vasuchakravarthi/cliniscan-lung-detection" 
           style="color: #1976d2; text-decoration: none; font-weight: 600;"
           target="_blank">
            🔗 GitHub Repository →
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
