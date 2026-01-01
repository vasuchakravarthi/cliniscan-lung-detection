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
        with st.spinner("⏳ Downloading detection model (52 MB)... First run only."):
            try:
                url = f"https://drive.google.com/uc?id={DETECTION_MODEL_ID}"
                gdown.download(url, det_path, quiet=False)
                st.success("✅ Detection model downloaded!")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                return False, False
    
    # Download classification model
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

# Download models
det_ready, clf_ready = download_models()

# -----------------------------------------------------------------------------
# 1️⃣ Load Models
# -----------------------------------------------------------------------------

# Define EfficientNet-B3 Classifier (EXACTLY as in your training)
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=False, num_classes=num_classes, drop_rate=dropout)
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_classification_model():
    """Load EfficientNet-B3 classification model"""
    if not clf_ready:
        return None
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model (EXACTLY as in training)
        model = EfficientNetClassifier(num_classes=2, dropout=0.3).to(device)
        
        model_path = "models/classification/best_clf_model.pth"
        
        if not os.path.exists(model_path):
            st.error("⚠️ Model file not found")
            return None
        
        # Load checkpoint (weights_only=False as in your code)
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract model state dict from 'model' key (as in your training)
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            st.success(f"✅ Model loaded! Accuracy: {checkpoint['acc']:.2f}%")
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        import traceback
        st.error(traceback.format_exc())
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
# 2️⃣ Grad-CAM for EfficientNet-B3
# -----------------------------------------------------------------------------

def generate_gradcam(model, img_tensor):
    """Generate Grad-CAM heatmap for EfficientNet-B3"""
    if model is None:
        return None, None
    
    try:
        device = next(model.parameters()).device
        model.eval()
        
        # For EfficientNet-B3, use the last conv layer (conv_head)
        # Access the inner model: model.model (timm model)
        feature_extractor = create_feature_extractor(
            model.model, 
            {"conv_head": "feat"}  # EfficientNet-B3 layer name
        )
        
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0).to(device)
            out = feature_extractor(img_tensor)
            preds = model(img_tensor)
            pred_class = preds.argmax(dim=1).item()
        
        # Generate heatmap
        feat_map = out["feat"].squeeze().detach().mean(dim=0).cpu().numpy()
        heatmap = cv2.resize(feat_map, (512, 512))  # Your training size
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
- 🎯 Detect **14 lung abnormalities** with bounding boxes (YOLOv8-M, mAP: 0.4305)
- 📊 Get **overall classification**: Abnormal vs Normal (EfficientNet-B3, Acc: 95.20%)
- 🧠 View **Grad-CAM heatmap** showing model focus areas

**Note**: Classification trained on 512×512 images, optimized for chest X-ray analysis.
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
    
    **Classification Classes**:
    - Abnormal (Class 0)
    - Normal (Class 1)
    
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
        
        # Preprocessing (EXACTLY as in your training: 512x512)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Your training size
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
        
        # Class names (0: Abnormal, 1: Normal as in your training)
        class_names = ["Abnormal", "Normal"]
        
        st.markdown(f"### Predicted: **{class_names[pred_class]}**")
        st.markdown(f"### Confidence: **{probs[0][pred_class]:.2%}**")
        
        st.markdown("#### Probabilities:")
        for i, name in enumerate(class_names):
            st.write(f"{name}: {probs[0][i].item():.2%}")
            st.progress(float(probs[0][i].item()))
        
        st.markdown("---")
        st.subheader("🧠 Grad-CAM")
        st.markdown("*Red/yellow areas show where the model focused for classification*")
        
        heatmap, _ = generate_gradcam(clf_model, img_tensor)
        
        if heatmap is not None:
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
            original_resized = np.array(image.resize((512, 512)))
            overlay = cv2.addWeighted(original_resized, 0.6, heatmap_colored, 0.4, 0)
            st.image(overlay, caption="Grad-CAM: Model Focus Areas", use_column_width=True)
        else:
            st.warning("Could not generate Grad-CAM")
    
    with col2:
        st.subheader("📦 Detection: 14 Abnormalities")
        
        with st.spinner("Detecting abnormalities..."):
            results = det_model.predict(np.array(image), conf=0.25, verbose=False)
        
        res_img = results[0].plot()
        st.image(res_img, caption="Detected Abnormalities with Bounding Boxes", use_column_width=True)
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            st.markdown("#### 🎯 Detected Abnormalities:")
            
            for i in range(min(5, len(boxes))):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                st.write(f"**{i+1}. {det_model.names[cls_id]}**")
                st.progress(conf)
                st.write(f"Confidence: {conf:.2%}\n")
            
            st.markdown(f"**Total Detections**: {len(boxes)}")
            st.markdown(f"**Average Confidence**: {float(boxes.conf.mean()):.2%}")
        else:
            st.success("✅ No abnormalities detected")
            st.info("This X-ray appears normal based on the detection model.")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<p><strong>⚠️ DISCLAIMER</strong></p>
<p>This system is for <strong>educational and research purposes only</strong>.</p>
<p>It should NOT be used for clinical diagnosis or medical decision-making.</p>
<p>Always consult a qualified radiologist for medical interpretation of chest X-rays.</p>
<hr>
<p><strong>Vasu Chakravarthi</strong> | SRKR Engineering College | BTech AIML 2025</p>
<p><a href='https://github.com/vasuchakravarthi/cliniscan-lung-detection'>GitHub Repository</a></p>
</div>
""", unsafe_allow_html=True)


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


import { useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { FileUpload } from './components/FileUpload';
import { ClassificationResults } from './components/ClassificationResults';
import { DetectionResults } from './components/DetectionResults';
import { Alert, AlertDescription } from './components/ui/alert';
import { AlertCircle, Activity } from 'lucide-react';
import { motion } from 'framer-motion';

export default function App() {
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<any>(null);

  const handleImageUpload = async (file: File) => {
    const imageUrl = URL.createObjectURL(file);
    setUploadedImage(imageUrl);
    setIsAnalyzing(true);
    setResults(null);

    // Simulate model processing time
    setTimeout(() => {
      // Mock results - in production, this would call your API
      const mockResults = {
        classification: {
          predictedClass: Math.random() > 0.5 ? 'Normal' : 'Abnormal',
          confidence: 0.85 + Math.random() * 0.15,
          probabilities: {
            Abnormal: 0.12 + Math.random() * 0.3,
            Normal: 0.58 + Math.random() * 0.3
          }
        },
        detection: {
          detections: generateMockDetections(),
          totalDetections: 0
        },
        gradcam: imageUrl // In production, this would be the actual Grad-CAM overlay
      };

      // Normalize probabilities
      const total = mockResults.classification.probabilities.Abnormal + 
                   mockResults.classification.probabilities.Normal;
      mockResults.classification.probabilities.Abnormal /= total;
      mockResults.classification.probabilities.Normal /= total;

      // Set predicted class based on probabilities
      mockResults.classification.predictedClass = 
        mockResults.classification.probabilities.Normal > mockResults.classification.probabilities.Abnormal 
          ? 'Normal' : 'Abnormal';
      
      mockResults.classification.confidence = 
        Math.max(mockResults.classification.probabilities.Normal, 
                mockResults.classification.probabilities.Abnormal);

      mockResults.detection.totalDetections = mockResults.detection.detections.length;

      setResults(mockResults);
      setIsAnalyzing(false);
    }, 2500);
  };

  const generateMockDetections = () => {
    const abnormalities = [
      'Aortic enlargement',
      'Atelectasis',
      'Calcification',
      'Cardiomegaly',
      'Consolidation',
      'ILD',
      'Infiltration',
      'Lung Opacity',
      'Nodule/Mass',
      'Other lesion',
      'Pleural effusion',
      'Pleural thickening',
      'Pneumothorax',
      'Pulmonary fibrosis'
    ];

    // Randomly decide if there are detections (30% chance of abnormalities)
    if (Math.random() > 0.7) {
      const numDetections = Math.floor(Math.random() * 3) + 1;
      return Array.from({ length: numDetections }, () => ({
        class: abnormalities[Math.floor(Math.random() * abnormalities.length)],
        confidence: 0.6 + Math.random() * 0.35
      }));
    }
    return [];
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-cyan-50 via-blue-50 to-indigo-100">
      <div className="flex">
        <Sidebar />
        
        <main className="flex-1 p-8">
          <div className="max-w-7xl mx-auto">
            {/* Header */}
            <motion.div 
              className="mb-8"
              initial={{ opacity: 0, y: -20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
            >
              <div className="bg-gradient-to-r from-cyan-600 via-blue-600 to-indigo-600 text-white rounded-2xl p-8 shadow-2xl">
                <h1 className="mb-3 flex items-center gap-3">
                  <Activity className="w-10 h-10" />
                  CliniScan: AI-Powered Lung Abnormality Detection
                </h1>
                <p className="text-cyan-50 opacity-90">
                  Upload a <strong>Chest X-ray</strong> image to detect abnormalities and classify diagnosis
                </p>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
            >
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                <div className="bg-gradient-to-br from-cyan-500 to-blue-500 text-white rounded-xl p-6 shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105">
                  <div className="text-3xl mb-2">🎯</div>
                  <h3 className="mb-2">14 Abnormalities</h3>
                  <p className="text-cyan-50 text-sm opacity-90">YOLOv8-M Detection (mAP: 0.4305)</p>
                </div>
                <div className="bg-gradient-to-br from-blue-500 to-indigo-500 text-white rounded-xl p-6 shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105">
                  <div className="text-3xl mb-2">📊</div>
                  <h3 className="mb-2">95.20% Accuracy</h3>
                  <p className="text-blue-50 text-sm opacity-90">EfficientNet-B3 Classification</p>
                </div>
                <div className="bg-gradient-to-br from-indigo-500 to-purple-500 text-white rounded-xl p-6 shadow-lg hover:shadow-2xl transition-all duration-300 hover:scale-105">
                  <div className="text-3xl mb-2">🧠</div>
                  <h3 className="mb-2">Grad-CAM</h3>
                  <p className="text-indigo-50 text-sm opacity-90">Visual AI Focus Areas</p>
                </div>
              </div>
            </motion.div>

            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
            >
              <Alert className="mb-6 bg-gradient-to-r from-blue-50 to-cyan-50 border-2 border-blue-300 shadow-md">
                <AlertCircle className="h-5 w-5 text-blue-600" />
                <AlertDescription className="text-blue-900">
                  <strong>Note:</strong> Classification trained on 512×512 images, optimized for chest X-ray analysis.
                </AlertDescription>
              </Alert>
            </motion.div>

            {/* File Upload */}
            <FileUpload onImageUpload={handleImageUpload} isAnalyzing={isAnalyzing} />

            {/* Uploaded Image */}
            {uploadedImage && (
              <motion.div 
                className="mt-8 mb-8"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5 }}
              >
                <h2 className="mb-4 text-gray-800 flex items-center gap-2">
                  📷 Uploaded X-ray
                  {isAnalyzing && (
                    <span className="text-sm text-blue-600 animate-pulse">Analyzing...</span>
                  )}
                </h2>
                <div className="bg-white rounded-2xl shadow-xl p-6 border-2 border-blue-200">
                  <img 
                    src={uploadedImage} 
                    alt="Uploaded chest X-ray" 
                    className="max-w-full h-auto mx-auto rounded-xl shadow-lg"
                    style={{ maxHeight: '400px' }}
                  />
                </div>
              </motion.div>
            )}

            {/* Results */}
            {results && !isAnalyzing && (
              <motion.div 
                className="mt-8"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6 }}
              >
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                  <ClassificationResults 
                    classification={results.classification}
                    gradcamImage={results.gradcam}
                  />
                  <DetectionResults 
                    detection={results.detection}
                    originalImage={uploadedImage}
                  />
                </div>
              </motion.div>
            )}

            {/* Disclaimer */}
            <motion.div 
              className="mt-12 bg-gradient-to-br from-red-50 to-orange-50 rounded-2xl shadow-xl p-8 text-center border-l-8 border-red-500"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.3 }}
            >
              <div className="text-4xl mb-4">⚠️</div>
              <p className="text-gray-800 mb-2"><strong>MEDICAL DISCLAIMER</strong></p>
              <p className="text-gray-700 mb-2">
                This system is for <strong>educational and research purposes only</strong>.
              </p>
              <p className="text-gray-700 mb-6">
                It should NOT be used for clinical diagnosis or medical decision-making.<br />
                Always consult a qualified radiologist for medical interpretation of chest X-rays.
              </p>
              <hr className="my-6 border-gray-300" />
              <p className="text-gray-800">
                <strong>Vasu Chakravarthi</strong> | SRKR Engineering College | BTech AIML 2025
              </p>
              <p className="text-gray-600 mt-2">
                <a 
                  href="https://github.com/vasuchakravarthi/cliniscan-lung-detection" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-blue-600 hover:text-blue-800 hover:underline transition-colors"
                >
                  GitHub Repository →
                </a>
              </p>
            </motion.div>
          </div>
        </main>
      </div>
    </div>
  );
}
import { Card } from './ui/card';
import { Progress } from './ui/progress';
import { Alert, AlertDescription } from './ui/alert';
import { CheckCircle, AlertTriangle, Target, BarChart3, TrendingUp } from 'lucide-react';
import { motion } from 'motion/react';

interface DetectionResultsProps {
  detection: {
    detections: Array<{
      class: string;
      confidence: number;
    }>;
    totalDetections: number;
  };
  originalImage: string | null;
}

export function DetectionResults({ detection, originalImage }: DetectionResultsProps) {
  const hasDetections = detection.totalDetections > 0;
  const avgConfidence = hasDetections
    ? detection.detections.reduce((sum, d) => sum + d.confidence, 0) / detection.totalDetections
    : 0;

  return (
    <div className="space-y-6">
      <motion.div
        initial={{ opacity: 0, x: 30 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Card className="bg-gradient-to-br from-white to-cyan-50 p-6 shadow-2xl border-2 border-cyan-200 hover:shadow-3xl transition-all duration-300">
          <div className="flex items-center gap-3 mb-4">
            <div className="p-3 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-xl shadow-lg">
              <Target className="w-6 h-6 text-white" />
            </div>
            <h2 className="text-gray-800">Object Detection</h2>
          </div>
          
          {originalImage && (
            <motion.div 
              className="mb-6 relative rounded-2xl overflow-hidden bg-gradient-to-br from-gray-900 to-gray-800 shadow-2xl"
              whileHover={{ scale: 1.02 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <img 
                src={originalImage} 
                alt="Detection visualization" 
                className="w-full h-auto"
              />
              {hasDetections && (
                <>
                  {/* Mock bounding boxes overlay with animations */}
                  <motion.div 
                    className="absolute top-[20%] left-[30%] w-[25%] h-[30%] border-4 border-red-500 rounded-lg shadow-2xl"
                    initial={{ opacity: 0, scale: 0.8 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.3 }}
                  >
                    <div className="absolute -top-8 left-0 bg-gradient-to-r from-red-500 to-rose-600 text-white px-3 py-1.5 text-xs rounded-lg shadow-lg">
                      {detection.detections[0]?.class} {(detection.detections[0]?.confidence * 100).toFixed(0)}%
                    </div>
                  </motion.div>
                  {detection.detections.length > 1 && (
                    <motion.div 
                      className="absolute top-[35%] right-[25%] w-[20%] h-[25%] border-4 border-yellow-500 rounded-lg shadow-2xl"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.5 }}
                    >
                      <div className="absolute -top-8 left-0 bg-gradient-to-r from-yellow-500 to-orange-500 text-white px-3 py-1.5 text-xs rounded-lg shadow-lg whitespace-nowrap">
                        {detection.detections[1]?.class} {(detection.detections[1]?.confidence * 100).toFixed(0)}%
                      </div>
                    </motion.div>
                  )}
                  {detection.detections.length > 2 && (
                    <motion.div 
                      className="absolute bottom-[25%] left-[20%] w-[22%] h-[20%] border-4 border-blue-500 rounded-lg shadow-2xl"
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: 0.7 }}
                    >
                      <div className="absolute -top-8 left-0 bg-gradient-to-r from-blue-500 to-cyan-600 text-white px-3 py-1.5 text-xs rounded-lg shadow-lg whitespace-nowrap">
                        {detection.detections[2]?.class} {(detection.detections[2]?.confidence * 100).toFixed(0)}%
                      </div>
                    </motion.div>
                  )}
                </>
              )}
            </motion.div>
          )}

          <p className="text-gray-600 text-xs mb-4 text-center bg-cyan-50 p-2 rounded-lg border border-cyan-200">
            📦 YOLOv8-M Detection Model (mAP: 0.4305)
          </p>

          {hasDetections ? (
            <div className="space-y-4">
              <div className="flex items-start gap-2 mb-4 bg-gradient-to-r from-yellow-50 to-orange-50 p-4 rounded-xl border-2 border-yellow-300">
                <AlertTriangle className="w-5 h-5 text-yellow-600 mt-0.5 flex-shrink-0" />
                <h3 className="text-gray-800">🎯 Detected Abnormalities</h3>
              </div>
              
              <div className="space-y-3">
                {detection.detections.slice(0, 5).map((det, index) => (
                  <motion.div 
                    key={index} 
                    className="bg-gradient-to-r from-gray-50 to-blue-50 p-4 rounded-xl border-2 border-blue-200 hover:border-blue-400 transition-all duration-300 shadow-md hover:shadow-lg"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 + index * 0.1 }}
                    whileHover={{ scale: 1.02 }}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-gray-800 flex items-center gap-2">
                        <div className="w-8 h-8 bg-gradient-to-br from-cyan-500 to-blue-600 rounded-lg flex items-center justify-center text-white shadow-md">
                          {index + 1}
                        </div>
                        <strong>{det.class}</strong>
                      </span>
                      <span className="text-gray-700 bg-white px-3 py-1 rounded-lg shadow-sm border border-gray-200">
                        {(det.confidence * 100).toFixed(2)}%
                      </span>
                    </div>
                    <Progress value={det.confidence * 100} className="h-3 bg-gray-200" />
                  </motion.div>
                ))}
              </div>

              <motion.div 
                className="mt-6 pt-4 border-t-2 border-gray-200 space-y-3"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.8 }}
              >
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-gradient-to-br from-indigo-50 to-purple-50 p-4 rounded-xl border-2 border-indigo-200 shadow-md">
                    <div className="flex items-center gap-2 mb-1">
                      <BarChart3 className="w-4 h-4 text-indigo-600" />
                      <span className="text-gray-600 text-sm">Total Detections</span>
                    </div>
                    <span className="text-2xl text-indigo-700">{detection.totalDetections}</span>
                  </div>
                  <div className="bg-gradient-to-br from-blue-50 to-cyan-50 p-4 rounded-xl border-2 border-blue-200 shadow-md">
                    <div className="flex items-center gap-2 mb-1">
                      <TrendingUp className="w-4 h-4 text-blue-600" />
                      <span className="text-gray-600 text-sm">Avg Confidence</span>
                    </div>
                    <span className="text-2xl text-blue-700">{(avgConfidence * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </motion.div>
            </div>
          ) : (
            <motion.div
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.3 }}
            >
              <Alert className="bg-gradient-to-br from-green-50 to-emerald-50 border-2 border-green-300 shadow-lg">
                <CheckCircle className="h-6 w-6 text-green-600" />
                <AlertDescription>
                  <p className="text-green-900 mb-2 flex items-center gap-2">
                    <strong className="text-lg">✅ No abnormalities detected</strong>
                  </p>
                  <p className="text-green-800 text-sm">
                    This X-ray appears normal based on the detection model. All systems clear!
                  </p>
                </AlertDescription>
              </Alert>
            </motion.div>
          )}
        </Card>
      </motion.div>
    </div>
  );
}
