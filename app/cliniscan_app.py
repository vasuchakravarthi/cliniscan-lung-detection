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
        with st.spinner("⏳ Downloading detection model (148 MB)... First run only."):
            try:
                url = f"https://drive.google.com/uc?id={DETECTION_MODEL_ID}"
                gdown.download(url, det_path, quiet=False)
                st.success("✅ Detection model downloaded!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return False, False
    
    # Download classification model
    if not os.path.exists(clf_path):
        with st.spinner("⏳ Downloading classification model (123 MB)..."):
            try:
                url = f"https://drive.google.com/uc?id={CLASSIFICATION_MODEL_ID}"
                gdown.download(url, clf_path, quiet=False)
                st.success("✅ Classification model downloaded!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return True, False
    
    return True, True

# Download models
det_ready, clf_ready = download_models()

# -----------------------------------------------------------------------------
# 1️⃣ Load Models
# -----------------------------------------------------------------------------

@st.cache_resource
def load_classification_model():
    """Load ResNet50 classification model"""
    if not clf_ready:
        return None
    
    try:
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)
        model_path = "models/classification/best_clf_model.pth"
        
        if not os.path.exists(model_path):
            st.error("⚠️ Model file not found")
            return None
        
        # Fix: Add weights_only=False for PyTorch 2.6+
        model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=False))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error: {e}")
        return None


@st.cache_resource
def load_detection_model():
    """Load YOLOv8 detection model"""
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

# Load models
clf_model = load_classification_model()
det_model = load_detection_model()

# -----------------------------------------------------------------------------
# 2️⃣ Grad-CAM
# -----------------------------------------------------------------------------

def generate_gradcam(model, img_tensor):
    """Generate Grad-CAM heatmap"""
    if model is None:
        return None, None
    
    try:
        model.eval()
        feature_extractor = create_feature_extractor(model, {"layer4.2": "feat"})
        
        with torch.no_grad():
            out = feature_extractor(img_tensor.unsqueeze(0))
            preds = model(img_tensor.unsqueeze(0))
            pred_class = preds.argmax(dim=1).item()
        
        feat_map = out["feat"].squeeze().detach().mean(dim=0).numpy()
        heatmap = cv2.resize(feat_map, (224, 224))
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        
        return heatmap, pred_class
    except Exception as e:
        st.error(f"Grad-CAM error: {e}")
        return None, None

# -----------------------------------------------------------------------------
# 3️⃣ UI
# -----------------------------------------------------------------------------

st.title("🩻 CliniScan: AI-Powered Lung Abnormality Detection")

st.markdown("""
Upload a **Chest X-ray** image to:
- 🎯 Detect **14 lung abnormalities** with bounding boxes
- 📊 Get **overall classification** (Normal, Abnormal, Severe)
- 🧠 View **Grad-CAM heatmap**

**Models**: YOLOv8-M (mAP: 0.4305) | ResNet50 (Acc: 95.20%)
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
    
    **⚠️ Disclaimer**: Educational purposes only.
    """)
    
    st.markdown("---")
    st.markdown("**Developer**: Vasu Chakravarthi")
    st.markdown("**Institution**: SRKR Engineering College")
    st.markdown("[GitHub Repository](https://github.com/vasuchakravarthi/cliniscan-lung-detection)")

st.markdown("---")
uploaded_file = st.file_uploader("📤 Upload Chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("📷 Uploaded X-ray")
    st.image(image, use_column_width=True)
    
    if clf_model is None or det_model is None:
        st.error("⚠️ Models not loaded. Check Google Drive file IDs.")
        st.stop()
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Classification")
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image)
        
        with torch.no_grad():
            preds = clf_model(img_tensor.unsqueeze(0))
            probs = torch.nn.functional.softmax(preds, dim=1)
            pred_class = torch.argmax(probs).item()
        
        class_names = ["Normal", "Abnormal", "Severe"]
        
        st.markdown(f"### Predicted: **{class_names[pred_class]}**")
        st.markdown(f"### Confidence: **{probs[0][pred_class]:.2%}**")
        
        st.markdown("#### Top 3:")
        top3_probs, top3_indices = torch.topk(probs, 3)
        for i in range(3):
            st.write(f"{i+1}. {class_names[top3_indices[0][i].item()]}: {top3_probs[0][i].item():.2%}")
        
        st.markdown("---")
        st.subheader("🧠 Grad-CAM")
        heatmap, _ = generate_gradcam(clf_model, img_tensor)
        
        if heatmap is not None:
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            original_resized = np.array(image.resize((224, 224)))
            overlay = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)
            st.image(overlay, caption="Model Focus Areas", use_column_width=True)
    
    with col2:
        st.subheader("📦 Detection")
        
        with st.spinner("Detecting..."):
            results = det_model.predict(np.array(image), conf=0.25, verbose=False)
        
        res_img = results[0].plot()
        st.image(res_img, caption="Detected Abnormalities", use_column_width=True)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            st.markdown("#### 🎯 Detected:")
            
            for i in range(min(5, len(boxes))):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                st.write(f"**{i+1}. {det_model.names[cls_id]}**")
                st.progress(conf)
                st.write(f"Confidence: {conf:.2%}\n")
            
            st.markdown(f"**Total**: {len(boxes)}")
        else:
            st.success("✅ No abnormalities detected")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<p><strong>⚠️ DISCLAIMER</strong></p>
<p>Educational purposes only. Not for clinical diagnosis.</p>
<hr>
<p><strong>Vasu Chakravarthi</strong> | SRKR Engineering College | BTech AIML 2025</p>
<p><a href='https://github.com/vasuchakravarthi/cliniscan-lung-detection'>GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
