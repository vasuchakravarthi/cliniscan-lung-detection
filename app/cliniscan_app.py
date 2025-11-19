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
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #e8f4f8 50%, #f0f7ff 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #0077be 0%, #005a8d 50%, #003d5c 100%);
        padding: 2rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,119,190,0.3);
        margin-bottom: 2rem;
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        margin: 0;
        font-weight: 700;
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Stats cards */
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
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
        box-shadow: 0 8px 20px rgba(124,58,237,0.25);
    }
    
    .stat-card.cyan {
        background: linear-gradient(135deg, #00b4d8 0%, #0096c7 100%);
        box-shadow: 0 8px 20px rgba(0,180,216,0.25);
    }
    
    .stat-card .emoji {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    .stat-card h3 {
        font-size: 1.3rem;
        margin: 0.5rem 0;
        font-weight: 600;
    }
    
    .stat-card p {
        font-size: 0.9rem;
        margin: 0;
    }
    
    /* Alerts */
    .info-alert {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
        border-left: 5px solid #0077be;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,119,190,0.15);
        color: #1e3a8a;
    }
    
    .info-alert strong {
        color: #1e40af;
    }
    
    .warning-alert {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-left: 5px solid #f59e0b;
        padding: 1.5rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(245,158,11,0.15);
        color: #78350f;
    }
    
    .warning-alert strong {
        color: #92400e;
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
        border-color: #7c3aed;
        background: linear-gradient(135deg, #ffffff 0%, #f5f3ff 100%);
    }
    
    .result-card.detection {
        border-color: #00b4d8;
        background: linear-gradient(135deg, #ffffff 0%, #ecfeff 100%);
    }
    
    .result-card h2 {
        color: #0077be;
        margin-top: 0;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.5rem 1.5rem;
        border-radius: 2rem;
        font-weight: 600;
        font-size: 1.2rem;
        margin: 1rem 0;
    }
    
    .badge.normal {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(16,185,129,0.3);
    }
    
    .badge.abnormal {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(239,68,68,0.3);
    }
    
    /* Confidence box */
    .confidence-box {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
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
    
    /* Detection items - FIXED */
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
    
    .detection-item strong {
        color: #1e293b !important;
        font-size: 1rem;
        font-weight: 700;
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
    
    /* Disclaimer */
    .disclaimer {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 8px solid #ef4444;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 6px 24px rgba(239,68,68,0.15);
    }
    
    .disclaimer .icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f0f9ff 100%);
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
        font-weight: 600;
    }
    
    .sidebar-card ol {
        color: #1e293b;
        font-size: 0.9rem;
        line-height: 1.8;
    }
    
    .sidebar-card li {
        color: #1e293b;
    }
    
    /* Classification Classes - FIXED */
    .class-item {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0.5rem 0;
        padding: 0.75rem;
        border-radius: 0.5rem;
    }
    
    .class-item.abnormal-class {
        background: #fee2e2;
        border: 2px solid #ef4444;
    }
    
    .class-item.abnormal-class strong {
        color: #7f1d1d !important;
        font-weight: 700;
    }
    
    .class-item.abnormal-class span {
        color: #991b1b;
    }
    
    .class-item.normal-class {
        background: #d1fae5;
        border: 2px solid #10b981;
    }
    
    .class-item.normal-class strong {
        color: #064e3b !important;
        font-weight: 700;
    }
    
    .class-item.normal-class span {
        color: #065f46;
    }
    
    .class-dot {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    
    /* Gradcam */
    .gradcam-section {
        background: linear-gradient(135deg, #ffffff 0%, #f5f3ff 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        border: 2px solid #7c3aed;
        margin-top: 1.5rem;
        box-shadow: 0 4px 12px rgba(124,58,237,0.1);
    }
    
    .gradcam-section h3 {
        color: #6d28d9;
        margin-top: 0;
    }
    
    .gradcam-section p {
        color: #4c1d95;
    }
    
    /* Hide branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
        
        if not os.path.exists(model_
