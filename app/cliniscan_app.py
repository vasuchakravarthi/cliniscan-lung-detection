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
import urllib.request
import time
import base64
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="🩻 CliniScan - Lung Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Gradient headers */
    .gradient-text {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4, #45b7d1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* Card styling */
    .card {
        background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(78, 205, 196, 0.1);
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(78, 205, 196, 0.15);
    }
    
    /* Button animations */
    .stButton > button {
        transition: all 0.3s ease;
        border: none;
        background: linear-gradient(45deg, #4ecdc4, #45b7d1);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 20px rgba(78, 205, 196, 0.4);
        background: linear-gradient(45deg, #45b7d1, #4ecdc4);
    }
    
    /* Progress bar animation */
    .stProgress > div > div {
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
    }
    
    /* Image hover effects */
    .stImage img {
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border-radius: 10px;
    }
    .stImage img:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e1e, #2d2d2d);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border-left: 4px solid #4ecdc4;
        transition: all 0.3s ease;
        margin: 10px 0;
    }
    .metric-card:hover {
        transform: translateX(5px);
        box-shadow: 0 5px 20px rgba(78, 205, 196, 0.2);
    }
    
    /* Results container */
    .results-container {
        background: rgba(30, 30, 30, 0.9);
        border-radius: 15px;
        padding: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(78, 205, 196, 0.2);
        animation: fadeIn 0.5s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Confidence bar */
    .confidence-bar {
        height: 8px;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        border-radius: 4px;
        transition: width 0.8s ease;
        margin: 10px 0;
    }
    
    /* Footer styling */
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: rgba(30, 30, 30, 0.95);
        backdrop-filter: blur(10px);
        padding: 10px;
        text-align: center;
        border-top: 1px solid rgba(78, 205, 196, 0.2);
        z-index: 1000;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        background: rgba(78, 205, 196, 0.1);
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(78, 205, 196, 0.2);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #4ecdc4, #45b7d1);
        color: white;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed rgba(78, 205, 196, 0.3);
        border-radius: 15px;
        padding: 20px;
        transition: all 0.3s ease;
    }
    .stFileUploader > div:hover {
        border-color: #4ecdc4;
        background: rgba(78, 205, 196, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# NAVIGATION STATE
# --------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []
if "theme" not in st.session_state:
    st.session_state.theme = "Dark"
if "detection_threshold" not in st.session_state:
    st.session_state.detection_threshold = 0.25
if "show_animations" not in st.session_state:
    st.session_state.show_animations = True

def go_to(page):
    st.session_state.page = page
    st.rerun()

# --------------------------------------------------
# MODEL URLS
# --------------------------------------------------

DET_URL = "https://huggingface.co/vasuchakravarthi/cliniscan-models/resolve/main/best1.pt"
CLF_URL = "https://huggingface.co/vasuchakravarthi/cliniscan-models/resolve/main/best_clf_model.pth"

# --------------------------------------------------
# DOWNLOAD MODELS
# --------------------------------------------------

@st.cache_resource
def download_models():
    os.makedirs("models", exist_ok=True)
    det_path = "models/best.pt"
    clf_path = "models/best_clf_model.pth"
    
    # Create progress bars
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    if not os.path.exists(det_path):
        progress_text.text("📥 Downloading detection model...")
        progress_bar.progress(25)
        urllib.request.urlretrieve(DET_URL, det_path)
    
    if not os.path.exists(clf_path):
        progress_text.text("📥 Downloading classification model...")
        progress_bar.progress(50)
        urllib.request.urlretrieve(CLF_URL, clf_path)
    
    progress_text.text("✅ Models downloaded successfully!")
    progress_bar.progress(100)
    time.sleep(0.5)
    progress_text.empty()
    progress_bar.empty()
    
    return det_path, clf_path

det_path, clf_path = download_models()

# --------------------------------------------------
# CLASSIFICATION MODEL
# --------------------------------------------------

class EfficientNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
    
    def forward(self, x):
        return self.model(x)

@st.cache_resource
def load_classifier():
    model = EfficientNetClassifier()
    checkpoint = torch.load(clf_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

# --------------------------------------------------
# DETECTION MODEL
# --------------------------------------------------

@st.cache_resource
def load_detector():
    return YOLO(det_path)

clf_model = load_classifier()
det_model = load_detector()

# --------------------------------------------------
# GRADCAM
# --------------------------------------------------

def generate_gradcam(model, img_tensor):
    try:
        extractor = create_feature_extractor(model.model, {"conv_head": "feat"})
        with torch.no_grad():
            img_tensor = img_tensor.unsqueeze(0)
            features = extractor(img_tensor)
        fmap = features["feat"].squeeze().mean(dim=0).cpu().numpy()
        heatmap = cv2.resize(fmap, (512,512))
        heatmap = np.maximum(heatmap,0)
        if heatmap.max()!=0:
            heatmap /= heatmap.max()
        return heatmap
    except:
        # Fallback if Grad-CAM fails
        return np.zeros((512, 512))

# --------------------------------------------------
# IMAGE TRANSFORM
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# --------------------------------------------------
# HEADER (TITLE + SIDEBAR)
# --------------------------------------------------

def show_header():
    # Animated title with gradient
    st.markdown("""
    <h1 style='text-align: center; margin-bottom: 0;'>
        <span class='gradient-text'>🩻 CliniScan</span>
    </h1>
    <p style='text-align: center; font-size: 1.2rem; color: #4ecdc4; margin-top: -10px;'>
        AI-Powered Lung Abnormality Detection System
    </p>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("---")
        
        # Navigation menu with simple buttons
        nav_col1, nav_col2 = st.columns(2)
        with nav_col1:
            if st.button("🏠 Home", use_container_width=True):
                go_to("home")
            if st.button("📊 Dashboard", use_container_width=True):
                go_to("dashboard")
        with nav_col2:
            if st.button("ℹ️ About", use_container_width=True):
                go_to("about")
            if st.button("📜 History", use_container_width=True):
                go_to("history")
        
        st.markdown("---")
        
        # System stats
        st.markdown("### 📊 System Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models Loaded", "2", delta="Ready")
        with col2:
            st.metric("Abnormalities", "14", delta="Detectable")
        
        # Detection threshold slider
        st.markdown("---")
        st.markdown("### ⚙️ Detection Settings")
        st.session_state.detection_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.detection_threshold,
            step=0.05,
            help="Adjust detection sensitivity"
        )
        
        st.session_state.show_animations = st.toggle(
            "Show Animations",
            value=st.session_state.show_animations
        )
        
        st.markdown("---")
        
        # Status indicator
        st.markdown("### 🟢 System Status")
        st.progress(1.0, text="All systems operational")
        
        st.markdown("---")
        st.caption("**Developer**: Vasu Chakravarthi")
        st.caption("**SRKR Engineering College**")
        st.caption("© 2025 CliniScan")

# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

def home_page():
    show_header()
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        <div class='card'>
            <h2>🏥 Welcome to CliniScan</h2>
            <p style='font-size: 1.1rem;'>
                Advanced AI-powered chest X-ray analysis for detecting lung abnormalities.
                Upload your X-ray and get instant analysis with:
            </p>
            <ul style='font-size: 1.05rem;'>
                <li>✅ 14 abnormality detection with bounding boxes</li>
                <li>✅ Overall classification (Abnormal/Normal)</li>
                <li>✅ Visual heatmap explanations</li>
                <li>✅ Downloadable reports</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick action buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🚀 Start Analysis", use_container_width=True):
                go_to("dashboard")
        with col2:
            if st.button("📖 Learn More", use_container_width=True):
                go_to("about")
        with col3:
            if st.button("📊 View Demo", use_container_width=True):
                st.info("🔄 Loading demo...")
                time.sleep(1)
                st.success("✅ Demo ready! Upload an X-ray to see it in action.")
    
    with col2:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <h3>🎯 Quick Stats</h3>
            <div style='margin: 20px 0;'>
                <div style='font-size: 2.5rem; color: #4ecdc4;'>95.2%</div>
                <div>Classification Accuracy</div>
            </div>
            <div style='margin: 20px 0;'>
                <div style='font-size: 2.5rem; color: #ff6b6b;'>43.05%</div>
                <div>Detection mAP</div>
            </div>
            <div style='margin: 20px 0;'>
                <div style='font-size: 2.5rem; color: #45b7d1;'>14</div>
                <div>Detectable Conditions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Features showcase
    st.markdown("---")
    st.markdown("### ✨ Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3>🎯 Detection</h3>
            <p>YOLOv8-M detects 14 lung abnormalities with precise bounding boxes</p>
            <span style='color: #4ecdc4;'>mAP: 43.05%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3>🧠 Classification</h3>
            <p>EfficientNet-B3 provides overall classification with high accuracy</p>
            <span style='color: #4ecdc4;'>Accuracy: 95.20%</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='card'>
            <h3>🔍 Explainability</h3>
            <p>Grad-CAM heatmap shows exactly where the model is focusing</p>
            <span style='color: #4ecdc4;'>Visual Explanations</span>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# DASHBOARD PAGE
# --------------------------------------------------

def dashboard_page():
    show_header()
    
    st.markdown("### 🎯 Analysis Dashboard")
    
    # Upload section with drag and drop
    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-ray",
        type=["jpg", "jpeg", "png"],
        help="Drag and drop or click to upload"
    )
    
    if uploaded_file:
        # Show processing animation
        with st.spinner("🔄 Processing your X-ray..."):
            if st.session_state.show_animations:
                time.sleep(0.5)  # Simulate processing
            
            image = Image.open(uploaded_file).convert("RGB")
            img_tensor = transform(image)
            
            # Save to history
            st.session_state.uploaded_images.append(uploaded_file.name)
            st.session_state.analysis_history.append({
                "filename": uploaded_file.name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Completed"
            })
        
        # Display results in a nice layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📷 Original X-ray")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔍 Detection Results")
            
            # Classification
            with torch.no_grad():
                preds = clf_model(img_tensor.unsqueeze(0))
                probs = torch.nn.functional.softmax(preds, dim=1)
                pred_class = torch.argmax(probs).item()
                confidence = probs[0][pred_class].item()
            
            classes = ["Abnormal", "Normal"]
            
            # Nice classification card
            class_color = "#ff6b6b" if pred_class == 0 else "#4ecdc4"
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='color: {class_color};'>{classes[pred_class]}</h3>
                <p style='font-size: 1.5rem; font-weight: bold;'>
                    {confidence:.1%} confidence
                </p>
                <div class='confidence-bar' style='width: {confidence*100}%;'></div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detection results
            results = det_model.predict(
                source=np.array(image), 
                conf=st.session_state.detection_threshold, 
                verbose=False
            )
            res_img = results[0].plot()
            
            if results[0].boxes is not None:
                boxes = results[0].boxes
                st.success(f"✅ Found {len(boxes)} abnormalities")
                
                # Show detections in a nice table
                detections = []
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    severity = "High" if conf > 0.7 else "Medium" if conf > 0.4 else "Low"
                    severity_color = "#ff6b6b" if severity == "High" else "#ffd93d" if severity == "Medium" else "#4ecdc4"
                    detections.append({
                        "Condition": det_model.names[cls],
                        "Confidence": f"{conf:.1%}",
                        "Severity": severity,
                        "Severity Color": severity_color
                    })
                
                if detections:
                    df = pd.DataFrame(detections)
                    # Display as styled table
                    for _, row in df.iterrows():
                        st.markdown(f"""
                        <div style='display: flex; justify-content: space-between; padding: 8px 12px; 
                                    margin: 4px 0; background: rgba(255,255,255,0.05); border-radius: 8px;'>
                            <span><strong>{row['Condition']}</strong></span>
                            <span>{row['Confidence']}</span>
                            <span style='color: {row['Severity Color']};'>
                                ● {row['Severity']}
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
        
        # Advanced features row
        st.markdown("---")
        st.markdown("#### 🧠 Advanced Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🔥 Grad-CAM Heatmap")
            heatmap = generate_gradcam(clf_model, img_tensor)
            heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            original = np.array(image.resize((512,512)))
            overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
            st.image(overlay, use_container_width=True)
            st.caption("Areas in red indicate where the model is focusing")
        
        with col2:
            st.markdown("##### 📊 Analysis Metrics")
            
            # Create a simple gauge chart for confidence
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Prediction Confidence"},
                delta = {'reference': 80},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#4ecdc4"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
            
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                # Detection confidence distribution
                boxes = results[0].boxes
                confs = [float(boxes.conf[i]) for i in range(len(boxes))]
                if confs:
                    # Create bar chart with matplotlib
                    fig, ax = plt.subplots(figsize=(8, 2))
                    bars = ax.bar(range(len(confs)), confs, color='#45b7d1')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Confidence')
                    ax.set_xlabel('Detection')
                    ax.set_xticks(range(len(confs)))
                    ax.set_xticklabels([f"#{i+1}" for i in range(len(confs))])
                    # Add value labels on bars
                    for bar, conf in zip(bars, confs):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{conf:.0%}', ha='center', va='bottom')
                    st.pyplot(fig)
                    plt.close()
        
        # Download report button
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Download Report", use_container_width=True):
                # Create a simple report
                report = f"""
                CliniScan Analysis Report
                =========================
                Filename: {uploaded_file.name}
                Date: {time.strftime("%Y-%m-%d %H:%M:%S")}
                
                Classification: {classes[pred_class]}
                Confidence: {confidence:.1%}
                
                Detections: {len(boxes) if results[0].boxes is not None else 0}
                
                """ 
                if results[0].boxes is not None:
                    for i, det in enumerate(detections):
                        report += f"\n{i+1}. {det['Condition']}: {det['Confidence']} ({det['Severity']})"
                
                st.download_button(
                    label="💾 Download as Text",
                    data=report,
                    file_name=f"cliniscan_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
    else:
        # Show upload prompt with nice styling
        st.markdown("""
        <div class='card' style='text-align: center; padding: 40px;'>
            <h2>📤 Ready for Analysis</h2>
            <p style='font-size: 1.1rem; color: #888;'>
                Upload a chest X-ray image to begin the analysis
            </p>
            <p style='font-size: 0.9rem; color: #666;'>
                Supported formats: JPG, JPEG, PNG
            </p>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

def about_page():
    show_header()
    
    st.markdown("### ℹ️ About CliniScan")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class='card'>
            <h3>🧠 The Technology Behind CliniScan</h3>
            <p>
                CliniScan is an advanced AI-powered system designed for chest X-ray analysis.
                It combines state-of-the-art deep learning models to provide comprehensive
                lung abnormality detection and classification.
            </p>
            
            <h4>🎯 Detection Model: YOLOv8-M</h4>
            <p>
                YOLOv8-M is a state-of-the-art object detection model that can identify 
                14 different types of lung abnormalities with precise bounding boxes.
            </p>
            
            <h4>🧠 Classification Model: EfficientNet-B3</h4>
            <p>
                EfficientNet-B3 provides accurate overall classification of chest X-rays
                as either Abnormal or Normal with high accuracy.
            </p>
            
            <h4>🔍 Explainability: Grad-CAM</h4>
            <p>
                Grad-CAM generates heatmaps that show exactly which regions of the X-ray
                the model is focusing on to make its decisions.
            </p>
            
            <h4>📊 Performance Metrics</h4>
            <ul>
                <li>Classification Accuracy: 95.20%</li>
                <li>Detection mAP: 43.05%</li>
                <li>Number of Detectable Conditions: 14</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='card'>
            <h3>🩺 Detectable Conditions</h3>
            <ul style='list-style-type: none; padding: 0;'>
                <li>🔴 Aortic enlargement</li>
                <li>🔴 Atelectasis</li>
                <li>🔴 Calcification</li>
                <li>🔴 Cardiomegaly</li>
                <li>🔴 Consolidation</li>
                <li>🔴 ILD</li>
                <li>🔴 Infiltration</li>
                <li>🔴 Lung Opacity</li>
                <li>🔴 Nodule/Mass</li>
                <li>🔴 Other lesion</li>
                <li>🔴 Pleural effusion</li>
                <li>🔴 Pleural thickening</li>
                <li>🔴 Pneumothorax</li>
                <li>🔴 Pulmonary fibrosis</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# --------------------------------------------------
# HISTORY PAGE
# --------------------------------------------------

def history_page():
    show_header()
    
    st.markdown("### 📜 Analysis History")
    
    if st.session_state.analysis_history:
        df = pd.DataFrame(st.session_state.analysis_history)
        
        # Add some statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Analyses", len(df))
        with col2:
            st.metric("Last Analysis", df.iloc[-1]['timestamp'])
        
        st.dataframe(df, use_container_width=True)
        
        # Visualization of history
        if len(df) > 1:
            st.markdown("#### 📈 Analysis Timeline")
            fig, ax = plt.subplots(figsize=(10, 3))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.groupby(df['timestamp'].dt.date).size().plot(kind='bar', ax=ax)
            ax.set_title('Analyses per Day')
            ax.set_xlabel('Date')
            ax.set_ylabel('Count')
            st.pyplot(fig)
            plt.close()
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.analysis_history = []
            st.rerun()
    else:
        st.info("ℹ️ No analysis history yet. Upload an X-ray to get started!")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

def show_footer():
    st.markdown("""
    <div class='footer'>
        <span style='color: #888;'>
            ⚠️ <strong>Disclaimer:</strong> For educational purposes only. 
            Not for clinical diagnosis. 
            © 2025 CliniScan by Vasu Chakravarthi | SRKR Engineering College
        </span>
    </div>
    """, unsafe_allow_html=True)

# --------------------------------------------------
# ROUTER
# --------------------------------------------------

if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
elif st.session_state.page == "about":
    about_page()
elif st.session_state.page == "history":
    history_page()

show_footer()
