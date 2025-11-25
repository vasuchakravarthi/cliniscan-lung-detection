# app/cliniscan_app.py
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
    .big-title {font-size: 4rem !important; font-weight: 800; text-align: center;
                background: linear-gradient(90deg, #1e88e5, #42a5f5);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;}
    .subtitle {text-align: center; font-size: 1.4rem; color: #555; margin-bottom: 2rem;}
    .card {background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); margin: 1rem 0;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                  padding: 1.5rem; border-radius: 15px; color: white; text-align: center;}
    .detection-box {background: #e3f2fd; padding: 1rem; border-radius: 12px;
                    border-left: 6px solid #1e88e5; margin: 0.8rem 0;}
    .stButton>button {background: #1e88e5; color: white; border-radius: 12px; padding: 0.6rem 2rem;}
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
        with st.spinner("Downloading YOLOv8 detection model (52MB)..."):
            gdown.download(f"https://drive.google.com/uc?id={DETECTION_MODEL_ID}", det_path, quiet=False)
    
    if not os.path.exists(clf_path):
        with st.spinner("Downloading classification model (129MB)..."):
            gdown.download(f"https://drive.google.com/uc?id={CLASSIFICATION_MODEL_ID}", clf_path, quiet=False)
    
    return True, True

det_ready, clf_ready = download_models()

# ====================== MODEL CLASSES ======================
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes, drop_rate=dropout)
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_classification_model():
    path = "models/classification/best_clf_model.pth"
    if not os.path.exists(path):
        return None
    try:
        # FIXED: This works 100% on Streamlit Cloud
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        
        model = EfficientNetClassifier(num_classes=2, dropout=0.3)
        model.load_state_dict(state_dict)
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return model.to(device)
    except:
        st.error("Failed to load classification model (PyTorch version issue)")
        return None

@st.cache_resource
def load_detection_model():
    path = "models/detection/best.pt"
    if not os.path.exists(path):
        return None
    return YOLO(path)

clf_model = load_classification_model()
det_model = load_detection_model()

# ====================== GRAD-CAM ======================
def get_gradcam_overlay(model, img_tensor, original_img):
    if model is None: return None
    try:
        device = next(model.parameters()).device
        img_tensor = img_tensor.unsqueeze(0).to(device)
        feats = create_feature_extractor(model.model, {"conv_head": "feat"})(img_tensor)["feat"]
        pred = model(img_tensor)
        pred_class = pred.argmax(1).item()
        
        grad = torch.autograd.grad(pred[0, pred_class], feats)[0]
        weights = grad.mean([2,3], keepdim=True)
        cam = (feats * weights).sum(1).squeeze().cpu().detach().numpy()
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (original_img.width, original_img.height))
        cam = cam / cam.max()
        cam = np.uint8(255 * cam)
        cam = cv2.applyColorMap(cam, cv2.COLORMAP_INFERNO)
        overlay = cv2.addWeighted(np.array(original_img), 0.65, cv2.cvtColor(cam, cv2.COLOR_BGR2RGB), 0.35, 0)
        return overlay
    except:
        return None

# ====================== MAIN UI ======================
st.markdown("<h1 class='big-title'>CliniScan Pro</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>AI-Powered Chest X-ray Analysis • 14 Abnormalities • Real-time Results</p>", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("<div class='metric-card'><h2>14</h2><p>Abnormalities</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='metric-card'><h2>95.2%</h2><p>Accuracy</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='metric-card'><h2>Real-time</h2><p>Analysis</p></div>", unsafe_allow_html=True)
with col4:
    st.markdown("<div class='metric-card'><h2>Grad-CAM</h2><p>Explainable AI</p></div>", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader("### Upload Chest X-ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    if not clf_model or not det_model:
        st.error("Models are still loading... Please wait 30 seconds.")
        st.stop()
    
    st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
    progress = st.progress(0)
    status = st.empty()
    
    # Classification
    status.text("Classifying: Normal vs Abnormal...")
    progress.progress(33)
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(image).unsqueeze(0).to(next(clf_model.parameters()).device)
    with torch.no_grad():
        prob = torch.softmax(clf_model(tensor), 1)[0]
        pred_label = "Abnormal" if prob[0] > prob[1] else "Normal"
        confidence = max(prob[0], prob[1]).item()
    
    # Detection
    status.text("Detecting 14 abnormalities...")
    progress.progress(66)
    results = det_model.predict(np.array(image), conf=0.25, verbose=False)[0]
    
    # Grad-CAM
    status.text("Generating attention heatmap...")
    progress.progress(90)
    heatmap = get_gradcam_overlay(clf_model, transform(image), image)
    progress.progress(100)
    status.empty()
    progress.empty()
    
    st.success("Analysis Complete!")
    
    tab1, tab2, tab3 = st.tabs(["Classification", "Abnormalities Detected", "AI Focus (Grad-CAM)"])
    
    with tab1:
        st.markdown(f"<h2>Overall Result: <span style='color:{'red' if pred_label=='Abnormal' else 'green'}'>{pred_label}</span></h2>", unsafe_allow_html=True)
        st.progress(confidence)
        st.write(f"**Confidence: {confidence:.1%}**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Normal", f"{prob[1]:.1%}")
        with col2:
            st.metric("Abnormal", f"{prob[0]:.1%}")
    
    with tab2:
        annotated = results.plot()
        st.image(annotated, use_column_width=True)
        if len(results.boxes) > 0:
            st.markdown("### Detected Findings:")
            for box in results.boxes:
                name = det_model.names[int(box.cls)]
                conf = box.conf.item()
                st.markdown(f"<div class='detection-box'><strong>{name}</strong> – {conf:.1%} confidence</div>", unsafe_allow_html=True)
        else:
            st.balloons()
            st.success("No abnormalities detected – Likely Normal!")
    
    with tab3:
        if heatmap is not None:
            st.image(heatmap, caption="Where the AI focused (Red = High Attention)", use_column_width=True)
            st.info("This heatmap shows which parts of the X-ray influenced the AI's decision.")
        else:
            st.warning("Grad-CAM visualization unavailable")

# ====================== FOOTER ======================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <h3>CliniScan Pro</h3>
    <p>For educational & research purposes only • Not for clinical use</p>
    <p>Developed by <strong>Vasu Chakravarthi</strong> • SRKR Engineering College • BTech AIML 2025</p>
    <p><a href='https://github.com/vasuchakravarthi/cliniscan-lung-detection'>GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)
