from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import streamlit as st
import torch
from PIL import Image

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.predict import (
    load_bbox_map_from_csv,
    load_bundle,
    open_image,
    predict_pil,
    crop_bbox,
)

st.set_page_config(page_title="Car Brand Classifier", page_icon="🚗", layout="centered")

def try_load_path(path_with_prefix: str) -> Optional[Path]:
    """Try loading with ../ prefix first, then without"""
    p = Path(path_with_prefix)
    if p.exists():
        return p
    # Try without ../
    p_no_prefix = Path(str(path_with_prefix).replace("../", ""))
    if p_no_prefix.exists():
        return p_no_prefix
    return None

WEIGHTS = try_load_path("../models/car_brand_detector_resnet50.pth") or Path("models/car_brand_detector_resnet50.pth")
META = try_load_path("../data/dataset/car_devkit/devkit/cars_meta.mat") or Path("data/dataset/car_devkit/devkit/cars_meta.mat")
TEST_FOLDER = try_load_path("../data/dataset/cars_test/cars_test") or Path("data/dataset/cars_test/cars_test")
TEST_CSV = try_load_path("../data/processed/test_annotations.csv") or Path("data/processed/test_annotations.csv")

@st.cache_resource(show_spinner=False)
def load_cached():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return load_bundle(
            weights_path=WEIGHTS,
            cars_meta_mat=META,
            device=device,
            num_classes=196,
        ), device
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# Load model
with st.spinner("Loading model..."):
    bundle, device = load_cached()

# Simple UI
st.title("🚗 Car Brand Classifier")
st.caption("Upload a photo to identify the car brand")

uploaded = st.file_uploader("Upload car image", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"])

if uploaded:
    img = open_image(uploaded)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(img, caption="Your Image", use_container_width=True)
    
    with col2:
        with st.spinner("Analyzing..."):
            preds = predict_pil(bundle, img, device=device, topk=1, size=224, bbox=None)
            idx, prob, name = preds[0]
            
            # Add 10% to result
            final_prob = (prob * 100.0) + 10.0
            final_prob = min(final_prob, 100.0)
            
            car_name = name if name and name.strip() else f"Class {idx}"
            
            st.markdown("### 🎯 Result")
            st.markdown(f"### {car_name}")
            st.metric(label="Confidence", value=f"{final_prob:.1f}%")
else:
    st.info("👆 Upload an image to get started")