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
    .stApp {
        background: linear-gradient(135deg, #f8fbff 0%, #e8f4f8 50%, #f0f7ff 100%);
    }
    
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
