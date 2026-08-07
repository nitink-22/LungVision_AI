import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. PAGE CONFIG & STYLING ---
st.set_page_config(page_title="LungVision AI Audit", page_icon="🫁", layout="wide")

# Safe CSS strictly for the dark theme and placeholder boxes
st.markdown("""
    <style>
    .placeholder-box {
        border: 2px dashed #3e404b;
        border-radius: 10px;
        height: 350px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #5d5f6e;
        font-style: italic;
        background-color: #161821;
    }
    .medical-card {
        background-color: #1a1c24;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2d2e38;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOGO & TITLE (Native & Safe) ---
st.markdown("# 🫁 LungVision")
st.markdown("<p style='font-size: 1.3rem; color: #00d4ff;'>Clinical AI Trustworthiness & Interpretability Audit</p>", unsafe_allow_html=True)
st.divider()

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_model():
    device = torch.device('cpu')
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 3) 
    try:
        model.load_state_dict(torch.load('lung_cancer_xai_model.pth', map_location=device))
        model.eval()
        return model, device
    except FileNotFoundError:
        return None, None

model, device = load_model()
if model is None:
    st.error("⚠️ System Error: 'lung_cancer_xai_model.pth' not found. Please ensure weights are in the directory.")
    st.stop()

# --- 4. PROCESSING FUNCTIONS ---
class ClinicalNoiseSimulator:
    @staticmethod
    def get_baseline_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @staticmethod
    def get_noise_transforms(severity):
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.1*severity, contrast=0.2*severity, saturation=0.5*severity, hue=0.15*severity),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def generate_heatmap(image, transform, model, device):
    tensor = transform(image).unsqueeze(0).to(device)
    cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
    grayscale_cam = cam(input_tensor=tensor)[0, :] 
    img_display = tensor.cpu().squeeze().permute(1, 2, 0).numpy()
    img_display = np.clip(img_display * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406], 0, 1)
    return show_cam_on_image(img_display, grayscale_cam, use_rgb=True)

# --- 5. MAIN LAYOUT ---
col_ctrl, col_res = st.columns([1, 2.5], gap="large")

with col_ctrl:
    st.markdown("### ⚙️ Audit Controls")
    st.write("1. **Upload Biopsy Slide**")
    uploaded_file = st.file_uploader("Select Image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.write("2. **Simulate Clinical Variance**")
    st.caption("Adjust this to mimic different hospital H&E staining protocols.")
    noise_severity = st.slider("H&E Stain Shift Severity", min_value=0.0, max_value=4.0, value=2.0, step=0.5)

with col_res:
    st.markdown("### 📊 XAI Reasoning Output")
    
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert('RGB')
        
        with st.spinner('Calculating pixel-level importance...'):
            baseline_cam = generate_heatmap(raw_image, ClinicalNoiseSimulator.get_baseline_transforms(), model, device)
            noisy_cam = generate_heatmap(raw_image, ClinicalNoiseSimulator.get_noise_transforms(noise_severity), model, device)
            
            h_col1, h_col2 = st.columns(2)
            with h_col1:
                st.image(baseline_cam, caption="Baseline Reasoning (Trained Data)", use_container_width=True)
            with h_col2:
                st.image(noisy_cam, caption=f"Stressed Reasoning (Noise: {noise_severity})", use_container_width=True)
        
        st.markdown("""
            <div class="medical-card">
                <h4 style="color: #00d4ff; margin-top: 0;">💡 Clinical Interpretation</h4>
                <p style="color: #d1d5db;">During this audit, we check for <b>Reasoning Consistency</b>. If the heatmap 'hotspots' shift 
                significantly away from the cellular morphology in the second image, it suggests the model 
                is <b>shortcut learning</b> (overfitting to color) rather than learning biological indicators.</p>
            </div>
        """, unsafe_allow_html=True)
        
    else:
        # Beautiful placeholders using native layout
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown('<div class="placeholder-box">Awaiting Baseline Audit...</div>', unsafe_allow_html=True)
        with p_col2:
            st.markdown('<div class="placeholder-box">Awaiting Stress Test Audit...</div>', unsafe_allow_html=True)