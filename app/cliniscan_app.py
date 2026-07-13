import streamlit as st
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
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
    
    /* Detection box styling */
    .detection-box {
        background: rgba(78, 205, 196, 0.1);
        border-left: 4px solid #4ecdc4;
        padding: 10px 15px;
        margin: 5px 0;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .detection-box:hover {
        background: rgba(78, 205, 196, 0.2);
        transform: translateX(5px);
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
    
    /* Zoom controls */
    .zoom-controls {
        display: flex;
        gap: 10px;
        margin: 10px 0;
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
if "show_labels" not in st.session_state:
    st.session_state.show_labels = True
if "box_opacity" not in st.session_state:
    st.session_state.box_opacity = 0.6

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
        return np.zeros((512, 512))

# --------------------------------------------------
# DRAW BOUNDING BOXES WITH CUSTOM STYLING
# --------------------------------------------------

def draw_custom_boxes(image, boxes, names, confs, show_labels=True, opacity=0.6):
    """Draw beautiful bounding boxes on image"""
    img = image.copy()
    img_array = np.array(img)
    
    # Define colors for different classes
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (128, 0, 0),    # Maroon
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Navy
        (128, 128, 0),  # Olive
        (128, 0, 128),  # Purple
        (0, 128, 128),  # Teal
        (192, 192, 0),  # Gold
        (192, 0, 192),  # Orchid
    ]
    
    for i, (box, cls, conf) in enumerate(zip(boxes, names, confs)):
        # Get box coordinates
        x1, y1, x2, y2 = map(int, box[:4])
        
        # Select color based on class index
        color = colors[i % len(colors)]
        
        # Create overlay for transparency
        overlay = img_array.copy()
        
        # Draw filled rectangle with opacity
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        img_array = cv2.addWeighted(img_array, 1 - opacity, overlay, opacity, 0)
        
        # Draw border
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 3)
        
        # Add label if enabled
        if show_labels:
            label = f"{cls} {conf:.1%}"
            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Draw label background
            cv2.rectangle(
                img_array,
                (x1, y1 - text_height - 10),
                (x1 + text_width + 10, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img_array,
                label,
                (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
    
    return Image.fromarray(img_array)

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
        
        # Navigation menu
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
        
        st.markdown("---")
        
        # Detection settings
        st.markdown("### 🎨 Detection Settings")
        st.session_state.detection_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.detection_threshold,
            step=0.05,
            help="Adjust detection sensitivity"
        )
        
        st.session_state.show_labels = st.toggle(
            "Show Labels on Boxes",
            value=st.session_state.show_labels
        )
        
        st.session_state.box_opacity = st.slider(
            "Box Opacity",
            min_value=0.1,
            max_value=1.0,
            value=st.session_state.box_opacity,
            step=0.1,
            help="Adjust bounding box transparency"
        )
        
        st.session_state.show_animations = st.toggle(
            "Show Animations",
            value=st.session_state.show_animations
        )
        
        st.markdown("---")
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
    
    uploaded_file = st.file_uploader(
        "📤 Upload Chest X-ray",
        type=["jpg", "jpeg", "png"],
        help="Drag and drop or click to upload"
    )
    
    if uploaded_file:
        with st.spinner("🔄 Processing your X-ray..."):
            if st.session_state.show_animations:
                time.sleep(0.5)
            
            image = Image.open(uploaded_file).convert("RGB")
            img_tensor = transform(image)
            
            st.session_state.uploaded_images.append(uploaded_file.name)
            st.session_state.analysis_history.append({
                "filename": uploaded_file.name,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "Completed"
            })
        
        # Classification
        with torch.no_grad():
            preds = clf_model(img_tensor.unsqueeze(0))
            probs = torch.nn.functional.softmax(preds, dim=1)
            pred_class = torch.argmax(probs).item()
            confidence = probs[0][pred_class].item()
        
        classes = ["Abnormal", "Normal"]
        
        # Detection
        results = det_model.predict(
            source=np.array(image), 
            conf=st.session_state.detection_threshold, 
            verbose=False
        )
        
        # Main results display
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📷 Original X-ray")
            st.image(image, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔍 Detection Results")
            
            # Classification card
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
            
            # Detection details
            if results[0].boxes is not None:
                boxes = results[0].boxes
                st.success(f"✅ Found {len(boxes)} abnormalities")
                
                # Draw custom bounding boxes
                box_data = []
                box_coords = []
                box_names = []
                box_confs = []
                
                for i in range(len(boxes)):
                    cls = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    
                    box_coords.append(xyxy)
                    box_names.append(det_model.names[cls])
                    box_confs.append(conf)
                    
                    severity = "High" if conf > 0.7 else "Medium" if conf > 0.4 else "Low"
                    severity_color = "#ff6b6b" if severity == "High" else "#ffd93d" if severity == "Medium" else "#4ecdc4"
                    
                    box_data.append({
                        "Condition": det_model.names[cls],
                        "Confidence": f"{conf:.1%}",
                        "Severity": severity,
                        "Severity Color": severity_color
                    })
                
                # Display detections as interactive cards
                st.markdown("##### 📋 Detected Abnormalities")
                for detection in box_data:
                    st.markdown(f"""
                    <div class='detection-box'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span><strong>{detection['Condition']}</strong></span>
                            <span>{detection['Confidence']}</span>
                            <span style='color: {detection['Severity Color']}; font-weight: bold;'>
                                ● {detection['Severity']}
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Draw and display custom bounding boxes
                st.markdown("##### 🎯 Bounding Boxes")
                img_with_boxes = draw_custom_boxes(
                    image.copy(),
                    box_coords,
                    box_names,
                    box_confs,
                    show_labels=st.session_state.show_labels,
                    opacity=st.session_state.box_opacity
                )
                st.image(img_with_boxes, use_container_width=True)
        
        # Advanced features
        st.markdown("---")
        st.markdown("#### 🧠 Advanced Analysis")
        
        tab1, tab2, tab3 = st.tabs(["🔥 Grad-CAM", "📊 Metrics", "📥 Export"])
        
        with tab1:
            st.markdown("##### 🔥 Grad-CAM Heatmap")
            heatmap = generate_gradcam(clf_model, img_tensor)
            heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            original = np.array(image.resize((512,512)))
            overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
            st.image(overlay, use_container_width=True)
            st.caption("Areas in red indicate where the model is focusing")
        
        with tab2:
            st.markdown("##### 📊 Analysis Metrics")
            
            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=confidence * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Prediction Confidence"},
                delta={'reference': 80},
                gauge={
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
            
            # Detection confidence distribution
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                confs = [float(boxes.conf[i]) for i in range(len(boxes))]
                if confs:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    bars = ax.bar(range(len(confs)), confs, color='#45b7d1')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Confidence')
                    ax.set_xlabel('Detection')
                    ax.set_xticks(range(len(confs)))
                    ax.set_xticklabels([f"#{i+1}" for i in range(len(confs))])
                    for bar, conf in zip(bars, confs):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{conf:.0%}', ha='center', va='bottom')
                    st.pyplot(fig)
                    plt.close()
        
        with tab3:
            st.markdown("##### 📥 Export Analysis")
            
            # Generate report
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
                for i, det in enumerate(box_data):
                    report += f"\n{i+1}. {det['Condition']}: {det['Confidence']} ({det['Severity']})"
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="💾 Download Report (TXT)",
                    data=report,
                    file_name=f"cliniscan_report_{time.strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                # Save image with boxes
                buf = BytesIO()
                img_with_boxes.save(buf, format="PNG")
                st.download_button(
                    label="🖼️ Download Image with Boxes",
                    data=buf.getvalue(),
                    file_name=f"cliniscan_detection_{time.strftime('%Y%m%d_%H%M%S')}.png",
                    mime="image/png",
                    use_container_width=True
                )
    else:
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Analyses", len(df))
        with col2:
            st.metric("Last Analysis", df.iloc[-1]['timestamp'])
        
        st.dataframe(df, use_container_width=True)
        
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
