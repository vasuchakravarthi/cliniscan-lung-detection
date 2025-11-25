import streamlit as st
from streamlit_option_menu import option_menu
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

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="CliniScan • AI Lung X-ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "# CliniScan v2.0\nAI-Powered Chest X-ray Analysis System\nDeveloped by Vasu Chakravarthi"
    }
)

# ====================== CUSTOM CSS ======================
st.markdown("""
<style>
    .main-header {font-size: 3.5rem; font-weight: 800; text-align: center; background: linear-gradient(90deg, #1e88e5, #42a5f5); 
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;}
    .subtitle {text-align: center; font-size: 1.3rem; color: #666; margin-bottom: 2rem;}
    .metric-card {background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 15px; color: white;}
    .detection-box {background: #f8f9fa; padding: 1rem; border-radius: 12px; border-left: 5px solid #1e88e5; margin: 0.5rem 0;}
    .gradcam-overlay {border: 3px solid #42a5f5; border-radius: 12px;}
    .stApp {background: #f5f7fb;}
    .success-box {background: #d4edda; border-left: 6px solid #28a745; padding: 1rem; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/100/000000/lungs.png")
    st.title("🫁 CliniScan Pro")
    st.markdown("**AI-Powered Chest X-ray Analysis**")
    st.markdown("---")
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "Detection", "Classification", "Heatmap", "About"],
        icons=["house", "search", "activity", "thermometer", "info-circle"],
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"},
            "nav-link-selected": {"background-color": "#1e88e5"},
        }
    )
    
    st.markdown("---")
    st.info("**14 Abnormalities Detected** • YOLOv8 + EfficientNet-B3")
    st.caption("Developed by **Vasu Chakravarthi**  \nSRKR Engineering College • AIML 2025")

# ====================== MODEL DOWNLOAD & LOAD ======================
# [Keep your existing download_models(), load_classification_model(), load_detection_model() functions here]
# ... (same as your original code)

det_ready, clf_ready = download_models()
clf_model = load_classification_model()
det_model = load_detection_model()

# ====================== GRAD-CAM FUNCTION (Improved Overlay) ======================
def generate_gradcam_overlay(model, img_tensor, original_img):
    if model is None:
        return None
    try:
        device = next(model.parameters()).device
        img_tensor = img_tensor.unsqueeze(0).to(device)
        
        feature_extractor = create_feature_extractor(model.model, {"conv_head": "feat"})
        feats = feature_extractor(img_tensor)["feat"]
        preds = model(img_tensor)
        pred_class = preds.argmax(dim=1).item()
        
        # Simple Grad-CAM using average gradients
        grads = torch.autograd.grad(preds[0, pred_class], feats, retain_graph=False)[0]
        weights = grads.mean(dim=[2, 3], keepdim=True)
        cam = (feats * weights).sum(dim=1).squeeze().detach().cpu().numpy()
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (original_img.width, original_img.height))
        cam = cam - cam.min()
        cam = cam / cam.max()
        cam = np.uint8(255 * cam)
        cam_colored = cv2.applyColorMap(cam, cv2.COLORMAP_INFERNO)
        cam_colored = cv2.cvtColor(cam_colored, cv2.COLOR_BGR2RGB)
        
        overlay = cv2.addWeighted(np.array(original_img), 0.6, cam_colored, 0.4, 0)
        return overlay, pred_class
    except:
        return None, None

# ====================== MAIN APP ======================
if selected == "Home" or selected is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 class='main-header'>CliniScan</h1>", unsafe_allow_html=True)
        st.markdown("<p class='subtitle'>Advanced AI for Chest X-ray Interpretation</p>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-card'><h3>14</h3><p>Abnormalities Detected</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card'><h3>95.2%</h3><p>Classification Accuracy</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card'><h3>mAP 0.43</h3><p>Detection Performance</p></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card'><h3>Real-time</h3><p>Instant Results</p></div>", unsafe_allow_html=True)

    st.markdown("### Upload Your Chest X-ray Below")
    uploaded_file = st.file_uploader(
        "Choose a chest X-ray image (JPG, PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

elif selected in ["Detection", "Classification", "Heatmap"]:
    if 'uploaded_file' not in locals() or uploaded_file is None:
        st.warning("Please go to **Home** and upload an X-ray first.")
        st.stop()

# ====================== PROCESSING WHEN IMAGE UPLOADED ======================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    if clf_model is None or det_model is None:
        st.error("Models are still loading or failed to download. Please wait or refresh.")
        st.stop()

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("Preprocessing image...")
    progress_bar.progress(20)
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image)
    
    # Classification
    status_text.text("Running classification model...")
    progress_bar.progress(50)
    with torch.no_grad():
        preds = clf_model(img_tensor.unsqueeze(0).to(next(clf_model.parameters()).device))
        probs = torch.nn.functional.softmax(preds, dim=1)[0]
        pred_class = preds.argmax().item()
        confidence = probs[pred_class].item()
    
    # Detection
    status_text.text("Detecting abnormalities...")
    progress_bar.progress(80)
    results = det_model.predict(np.array(image), conf=0.25, iou=0.45, verbose=False)[0]
    
    # Grad-CAM
    status_text.text("Generating attention heatmap...")
    heatmap_overlay, _ = generate_gradcam_overlay(clf_model, img_tensor, image)
    progress_bar.progress(100)
    status_text.text("Analysis Complete!")
    
    st.success("Analysis completed successfully!")
    progress_bar.empty()
    status_text.empty()

    # ====================== DISPLAY RESULTS ======================
    tab1, tab2, tab3 = st.tabs(["Classification", "Object Detection", "Attention Heatmap"])

    with tab1:
        st.image(image, caption="Original X-ray", use_column_width=True)
        st.markdown(f"### Prediction: **{'Abnormal' if pred_class == 0 else 'Normal'}**")
        st.progress(confidence)
        st.markdown(f"**Confidence**: {confidence:.2%}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Normal Probability", f"{probs[1]:.2%}")
        with col2:
            st.metric("Abnormal Probability", f"{probs[0]:.2%}")

    with tab2:
        annotated = results.plot(line_width=2, font_size=12)
        st.image(annotated, caption="Detected Abnormalities", use_column_width=True)
        
        if len(results.boxes) > 0:
            st.markdown("### Detected Findings:")
            for i, (cls, conf) in enumerate(zip(results.boxes.cls, results.boxes.conf)):
                name = det_model.names[int(cls)]
                st.markdown(f"<div class='detection-box'><strong>{name}</strong> – {conf:.1%} confidence</div>", 
                           unsafe_allow_html=True)
        else:
            st.balloons()
            st.success("No abnormalities detected – Likely Normal X-ray")

    with tab3:
        if heatmap_overlay is not None:
            st.image(heatmap_overlay, caption="Grad-CAM: Where the AI is Looking", 
                    use_column_width=True, clamp=True)
            st.info("Red/Yellow areas show regions the model focused on most when making its decision.")
        else:
            st.warning("Grad-CAM visualization unavailable")

    # About Tab
    if selected == "About":
        st.markdown("""
        ### About CliniScan
        A state-of-the-art dual-model system combining:
        - **YOLOv8** for multi-label object detection (14 thoracic abnormalities)
        - **EfficientNet-B3** for binary classification (Normal vs Abnormal)
        - **Grad-CAM** explainability for clinical trust
        
        Trained on VinBigData + custom annotated dataset.
        
        **For educational & research use only** – not a medical device.
        """)
        st.markdown("### [GitHub Repository](https://github.com/vasuchakravarthi/cliniscan-lung-detection) | [LinkedIn](https://linkedin.com/in/vasuchakravarthi)")
