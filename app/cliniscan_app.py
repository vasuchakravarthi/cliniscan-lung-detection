import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from torchvision.models import resnet50
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
import os
import gdown

# Page configuration
st.set_page_config(
    page_title="🩻 CliniScan - Lung Abnormality Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 🔽 GOOGLE DRIVE MODEL DOWNLOAD
# -----------------------------------------------------------------------------

# Replace these with your actual Google Drive file IDs
DETECTION_MODEL_ID = "1RN903UCBYkkY9JftW9NauOZbCdFTLc1a"  # Get from Drive sharing link
CLASSIFICATION_MODEL_ID = "1e2xHBMKshkPcaUDJSLLF-dJe2ohQIhk_"  # Get from Drive sharing link

@st.cache_resource
def download_models():
    """Download models from Google Drive if not present (runs once)"""
    os.makedirs("models/detection", exist_ok=True)
    os.makedirs("models/classification", exist_ok=True)
    
    det_path = "models/detection/best.pt"
    clf_path = "models/classification/best_clf_model.pth"
    
    # Download detection model (148.6 MB)
    if not os.path.exists(det_path):
        with st.spinner("⏳ Downloading detection model (148 MB)... First run only, please wait."):
            try:
                url = f"https://drive.google.com/uc?id={DETECTION_MODEL_ID}"
                gdown.download(url, det_path, quiet=False)
                st.success("✅ Detection model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Error downloading detection model: {e}")
                st.info("💡 Please check the file ID and sharing settings on Google Drive")
                return False, False
    
    # Download classification model (123.3 MB)
    if not os.path.exists(clf_path):
        with st.spinner("⏳ Downloading classification model (123 MB)..."):
            try:
                url = f"https://drive.google.com/uc?id={CLASSIFICATION_MODEL_ID}"
                gdown.download(url, clf_path, quiet=False)
                st.success("✅ Classification model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Error downloading classification model: {e}")
                st.info("💡 Please check the file ID and sharing settings on Google Drive")
                return True, False
    
    return True, True

# Download models on first run
det_ready, clf_ready = download_models()

# -----------------------------------------------------------------------------
# 1️⃣ Load Models
# -----------------------------------------------------------------------------

@st.cache_resource
def load_classification_model():
    """Load ResNet50 classification model"""
    if not clf_ready:
        st.error("⚠️ Classification model not ready")
        return None
    
    try:
        model = resnet50(weights=None)  

