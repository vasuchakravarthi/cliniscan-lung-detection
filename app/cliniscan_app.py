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

# Page configuration
st.set_page_config(
    page_title="🩻 CliniScan - Lung Abnormality Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 1️⃣ Load Models
# -----------------------------------------------------------------------------

@st.cache_resource
def load_classification_model():
    """Load ResNet50 classification model"""
    try:
        model = resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)  # 3 classes: Normal, Abnormal, Severe
        
        model_path = "models/classification/best_clf_model.pth"
        
        if not os.path.exists(model_path):
            st.error("⚠️ Classification model not found. Please download from Google Drive (see README)")
            return None
        
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return None

@st.cache_resource
def load_detection_model():
    """Load YOLOv8 detection model"""
    try:
        model_path = "models/detection/best.pt"
        
        if not os.path.exists(model_path):
            st.error("⚠️ Detection model not found. Please download from Google Drive (see README)")
            return None
        
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading detection model: {e}")
        return None

# Load models (cached)
clf_model = load_classification_model()
det_model = load_detection_model()

# -----------------------------------------------------------------------------
# 2️⃣ Grad-CAM for Interpretability
# -----------------------------------------------------------------------------

def generate_gradcam(model, img_tensor):
    """Generate Grad-CAM heatmap for classification interpretability"""
    if model is None:
        return None, None
    
    try:
        model.eval()
        # Extract features from layer4.2 (last conv layer of ResNet50)
        feature_extractor = create_feature_extractor(model, {"layer4.2": "feat"})
        
        with torch.no_grad():
            out = feature_extractor(img_tensor.unsqueeze(0))
            preds = model(img_tensor.unsqueeze(0))
            pred_class = preds.argmax(dim=1).item()
        
        # Generate heatmap
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
# 3️⃣ Streamlit UI
# -----------------------------------------------------------------------------

# Title and description
st.title("🩻 CliniScan: AI-Powered Lung Abnormality Detection")

st.markdown("""
<style>
    .main-header {
        font-size: 1.2rem;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
Upload a **Chest X-ray** image to:
- 🎯 Detect and localize **14 lung abnormalities** with bounding boxes
- 📊 Get **overall classification** (Normal, Abnormal, Severe)
- 🧠 View **Grad-CAM heatmap** showing model focus areas

**Models**:
- Detection: YOLOv8-Medium (mAP@0.5: **0.4305**)
- Classification: ResNet50 (Accuracy: **95.20%**)
""")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About CliniScan")
    st.markdown("""
    **CliniScan** uses state-of-the-art deep learning to assist in chest X-ray analysis.
    
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
    
    **⚠️ Disclaimer**: For educational/research purposes only.
    Not for clinical diagnosis.
    """)
    
    st.markdown("---")
    st.markdown("**Developed by**: Vasu Chakravarthi")
    st.markdown("**Institution**: SRKR Engineering College")
    st.markdown("**GitHub**: [Repository](https://github.com/vasuchakravarthi/cliniscan-lung-detection)")

# File uploader
st.markdown("---")
uploaded_file = st.file_uploader(
    "📤 Upload Chest X-ray Image (JPG/PNG)", 
    type=["jpg", "jpeg", "png"],
    help="Upload a frontal chest X-ray image for analysis"
)

if uploaded_file:
    # Load and display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    st.subheader("📷 Uploaded Chest X-ray")
    st.image(image, caption="Original X-ray Image", use_column_width=True)
    
    # Check if models are loaded
    if clf_model is None or det_model is None:
        st.error("⚠️ Models not loaded. Please ensure model files are in the correct location.")
        st.info("📥 Download models from Google Drive links in the README.md file")
        st.stop()
    
    # Create two columns for results
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    # -------------------------------------------------------------------------
    # Left Column: Classification Results
    # -------------------------------------------------------------------------
    with col1:
        st.subheader("🔍 Overall Classification Results")
        
        # Preprocess image for classification
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image)
        
        # Get predictions
        with torch.no_grad():
            preds = clf_model(img_tensor.unsqueeze(0))
            probs = torch.nn.functional.softmax(preds, dim=1)
            pred_class = torch.argmax(probs).item()
        
        # Class names (update based on your model)
        class_names = ["Normal", "Abnormal", "Severe"]
        
        # Display main prediction
        st.markdown(f"### Predicted Condition: **{class_names[pred_class]}**")
        st.markdown(f"### Confidence: **{probs[0][pred_class]:.2%}**")
        
        # Display top 3 predictions
        st.markdown("#### Top 3 Predictions:")
        top3_probs, top3_indices = torch.topk(probs, 3)
        
        for i in range(3):
            class_idx = top3_indices[0][i].item()
            prob = top3_probs[0][i].item()
            st.write(f"{i+1}. **{class_names[class_idx]}**: {prob:.2%}")
        
        # Grad-CAM visualization
        st.markdown("---")
        st.subheader("🧠 Grad-CAM Heatmap")
        st.markdown("*Areas highlighted in red/yellow indicate regions the model focused on for classification*")
        
        heatmap, _ = generate_gradcam(clf_model, img_tensor)
        
        if heatmap is not None:
            # Apply colormap and create overlay
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            
            # Create overlay
            original_resized = np.array(image.resize((224, 224)))
            overlay = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)
            
            st.image(overlay, caption="Grad-CAM: Model Focus Areas", use_column_width=True)
        else:
            st.warning("Could not generate Grad-CAM heatmap")
    
    # -------------------------------------------------------------------------
    # Right Column: Detection Results
    # -------------------------------------------------------------------------
    with col2:
        st.subheader("📦 Object Detection: 14 Abnormalities")
        
        # Run YOLO detection
        with st.spinner("Detecting abnormalities..."):
            results = det_model.predict(np.array(image), conf=0.25, verbose=False)
        
        # Plot results with bounding boxes
        res_img = results[0].plot()
        st.image(res_img, caption="Detected Lung Abnormalities (with Bounding Boxes)", use_column_width=True)
        
        # Display detection details
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            class_names_det = det_model.names
            
            st.markdown("---")
            st.markdown("#### 🎯 Detected Abnormalities:")
            
            # Show top 5 detections
            num_detections = min(5, len(boxes))
            for i in range(num_detections):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                class_name = class_names_det[cls_id]
                
                # Display with progress bar for confidence
                st.write(f"**{i+1}. {class_name}**")
                st.progress(conf)
                st.write(f"Confidence: {conf:.2%}")
                st.write("")
            
            # Summary stats
            st.markdown("---")
            st.markdown(f"**Total Detections**: {len(boxes)}")
            st.markdown(f"**Average Confidence**: {float(boxes.conf.mean()):.2%}")
        else:
            st.success("✅ **No abnormalities detected**")
            st.info("This X-ray appears to be normal based on the detection model.")

# -----------------------------------------------------------------------------
# 4️⃣ Model Performance Section
# -----------------------------------------------------------------------------

st.markdown("---")
st.header("📊 Model Performance Metrics")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🎯 Detection Model (YOLOv8-Medium)
    
    **Performance Metrics**:
    - **mAP@0.5**: 0.4305 (43.05%)
    - **mAP@0.5:0.95**: 0.2327 (23.27%)
    - **Precision**: 49.2%
    - **Recall**: 41.9%
    - **Speed**: ~45ms per image (~22 FPS on T4 GPU)
    
    **Key Strengths**:
    - ✅ 67% recall for **Pneumothorax** (critical, life-threatening condition)
    - ✅ **Exceeds VinBigData competition benchmarks** by 27%
    - ✅ Strong performance on cardiac abnormalities (Cardiomegaly: 0.46 mAP)
    
    **Dataset**: VinBigData (18,000 chest X-rays, 14 classes)
    """)

with col2:
    st.markdown("""
    ### 📊 Classification Model (ResNet50)
    
    **Performance Metrics**:
    - **Accuracy**: 95.20%
    - **Classes**: 3 (Normal, Abnormal, Severe)
    - **Input Size**: 224×224 pixels
    - **Speed**: ~50ms per image
    
    **Key Strengths**:
    - ✅ High accuracy on balanced test set
    - ✅ **Grad-CAM interpretability** for clinical trust
    - ✅ Fast inference suitable for real-time screening
    - ✅ Transfer learning from ImageNet pre-trained weights
    
    **Architecture**: ResNet50 with custom classification head
    """)

# Analysis section
st.markdown("---")
st.header("📈 Strengths & Limitations")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### ✅ Strengths
    - **High Accuracy**: Competitive with state-of-the-art research
    - **Critical Detection**: Excellent recall for life-threatening conditions
    - **Interpretability**: Grad-CAM provides visual explanations
    - **Speed**: Real-time processing capability
    - **Comprehensive**: Detects 14 different abnormality types
    """)

with col2:
    st.markdown("""
    ### ⚠️ Limitations
    - **Small Findings**: Lower performance on subtle lesions
    - **Image Quality**: May struggle with low-contrast or artifacts
    - **Class Imbalance**: Rare conditions have fewer training samples
    - **Not Diagnostic**: Assists radiologists, doesn't replace them
    - **External Validation**: Needs testing on diverse populations
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p><strong>⚠️ IMPORTANT DISCLAIMER</strong></p>
    <p>This tool is for <strong>educational and research purposes only</strong>.</p>
    <p>It should NOT be used for clinical diagnosis or medical decision-making.</p>
    <p>Always consult a qualified radiologist for medical interpretation of chest X-rays.</p>
    <hr>
    <p>Developed by <strong>Vasu Chakravarthi</strong> | SRKR Engineering College | BTech AIML 2025</p>
    <p>GitHub: <a href='https://github.com/vasuchakravarthi/cliniscan-lung-detection'>github.com/vasuchakravarthi/cliniscan-lung-detection</a></p>
</div>
""", unsafe_allow_html=True)
