# app/cliniscan_app.py
import streamlit as st
from streamlit_option_menu import option_menu
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
import gdown

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="CliniScan Pro • AI Chest X-ray Analysis",
    page_icon="Lungs",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main-header {font-size: 3.8rem; font-weight: 800; text-align: center; 
                  background: linear-gradient(90deg, #1e88e5, #42a5f5);
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
    .subtitle {text-align: center; font-size: 1.4rem; color: #555; margin-bottom: 2rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  padding: 1.5rem; border-radius: 15px; color: white; text-align: center;}
    .detection-box {background: #f8f9fa; padding: 1rem; border-radius: 12px; 
                    border-left: 5px solid #1e88e5; margin: 0.5rem 0;}
</style>
""", unsafe_allow_html=True)

# ====================== MODEL DOWNLOAD ======================
DETECTION_MODEL_ID = "1RN903UCBYkkY9JftW9NauOZbCdFTLc1a"
CLASSIFICATION_MODEL_ID = "1e2xHBMKshkPcaUDJSLLF-dJe2ohQIhk_"

@st.cache_resource
def download_models():
    os.makedirs("models/detection", exist_ok=True)
    os.makedirs("models/classification", exist_ok=True)
    
    det_path = "models/detection/best.pt"
    clf_path = "models/classification/best_clf_model.pth"
    
    if not os.path.exists(det_path):
        with st.spinner("Downloading YOLOv8 detection model (~52MB)..."):
            gdown.download(f"https://drive.google.com/uc?id={DETECTION_MODEL_ID}", det_path, quiet=False)
    
    if not os.path.exists(clf_path):
        with st.spinner("Downloading EfficientNet classifier (~129MB)..."):
            gdown.download(f"https://drive.google.com/uc?id={CLASSIFICATION_MODEL_ID}", clf_path, quiet=False)
    
    return True, True

det_ready, clf_ready = download_models()

# ====================== MODEL DEFINITIONS ======================
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes, drop_rate=dropout)
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_classification_model():
    if not clf_ready: return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetClassifier(num_classes=2, dropout=0.3).to(device)
    checkpoint = torch.load("models/classification/best_clf_model.pth", map_location=device)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint)
    model.eval()
    return model

@st.cache_resource
def load_detection_model():
    if not det_ready: return None
    return YOLO("models/detection/best.pt")

clf_model = load_classification_model()
det_model = load_detection_model()

# ====================== GRAD-CAM ======================
def generate_gradcam(model, img_tensor, original_img):
    if model is None: return None
    try:
        device = next(model.parameters()).device
        img_tensor = img_tensor.unsqueeze(0).to(device)
        feats = create_feature_extractor(model.model, {"conv_head": "feat"})(img_tensor)["feat"]
        preds = model(img_tensor)
        pred_class = preds.argmax(dim=1).item()
        
        grad = torch.autograd.grad(preds[0, pred_class], feats)[0]
        weights = grad.mean([2,3], keepdim=True)
        cam = (feats * weights).sum(1).squeeze().cpu().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (original_img.width, original_img.height))
        cam = cam / cam.max()
        cam = np.uint8(255 * cam)
        cam_colored = cv2.applyColorMap(cam, cv2.COLORMAP_INFERNO)
        overlay = cv2.addWeighted(np.array(original_img), 0.6, cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB), 0.4, 0)
        return overlay
    except:
        return None

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/lungs.png")
    st.title("CliniScan Pro")
    st.markdown("**AI-Powered Chest X-ray Analysis**")
    
    choice = option_menu(
        menu_title=None,
        options=["Home", "Analyze X-ray", "About"],
        icons=["house", "file-medical", "info-circle"],
        default_index=0,
        styles={"nav-link-selected": {"background-color": "#1e88e5"}}
    )

# ====================== MAIN APP ======================
if choice == "Home":
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("<h1 class='main-header'>CliniScan</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Advanced AI for Chest X-ray Interpretation</p>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='metric-card'><h2>14</h2><p>Abnormalities Detected</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><h2>95.2%</h2><p>Classification Accuracy</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><h2>Real-time</h2><p>Instant Results</p></div>", unsafe_allow_html=True)
    
    st.markdown("### Upload a Chest X-ray to begin analysis")
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

elif choice == "Analyze X-ray":
    if 'uploaded_file' not in locals() or not uploaded_file:
        st.warning("Please go to **Home** and upload an image first.")
        st.stop()
    
    image = Image.open(uploaded_file).convert("RGB")
    
    if not clf_model or not det_model:
        st.error("Models are still loading. Please wait...")
        st.stop()
    
    progress = st.progress(0)
    status = st.empty()
    
    status.text("Running classification...")
    progress.progress(30)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(next(clf_model.parameters()).device)
    with torch.no_grad():
        pred = torch.softmax(clf_model(tensor), 1)[0]
        label = "Abnormal" if pred[0] > pred[1] else "Normal"
        confidence = max(pred[0], pred[1]).item()
    
    status.text("Running object detection...")
    progress.progress(70)
    results = det_model.predict(np.array(image), conf=0.25, verbose=False)[0]
    
    status.text("Generating heatmap...")
    progress.progress(90)
    heatmap = generate_gradcam(clf_model, transform(image), image)
    progress.progress(100)
    status.empty()
    progress.empty()
    
    st.success("Analysis Complete!")
    
    tab1, tab2, tab3 = st.tabs(["Classification", "Detection", "Attention Map"])
    
    with tab1:
        st.image(image, "Original X-ray", use_column_width=True)
        st.markdown(f"### **Prediction: {label}**")
        st.progress(confidence)
        st.write(f"Confidence: **{confidence:.1%}**")
    
    with tab2:
        annotated = results.plot()
        st.image(annotated, "Detected Abnormalities", use_column_width=True)
        if len(results.boxes) > 0:
            for box in results.boxes:
                name = det_model.names[int(box.cls)]
                conf = box.conf.item()
                st.markdown(f"<div class='detection-box'><strong>{name}</strong> – {conf:.1%}</div>", unsafe_allow_html=True)
        else:
            st.balloons()
            st.success("No abnormalities detected!")
    
    with tab3:
        if heatmap is not None:
            st.image(heatmap, "Where the AI focused (Grad-CAM)", use_column_width=True)
        else:
            st.info("Grad-CAM not available")

elif choice == "About":
    st.markdown("# About CliniScan")
    st.markdown("""
    **Dual-model AI system** combining:
    - YOLOv8 for detecting 14 thoracic abnormalities
    - EfficientNet-B3 for Normal vs Abnormal classification
    - Grad-CAM for explainability
    
    Trained on VinBigData + custom dataset.  
    Accuracy: **95.2%** • mAP@0.5: **0.4305**
    
    **For educational & research use only.**
    """)
    st.markdown("### [GitHub Repository](https://github.com/vasuchakravarthi/cliniscan-lung-detection)")

# Save this file as: app/cliniscan_app.py
# Your requirements.txt is already perfect!
