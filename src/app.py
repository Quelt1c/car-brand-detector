# src/app.py
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

from src.predict import (  # noqa: E402
    DEFAULT_META,
    DEFAULT_WEIGHTS,
    load_bbox_map_from_csv,
    load_bundle,
    open_image,
    predict_pil,
    crop_bbox,
)

st.set_page_config(page_title="Car Brand Classifier", page_icon="🚗", layout="wide")

DEFAULT_TEST_FOLDER = Path("data/dataset/cars_test/cars_test")
DEFAULT_TEST_CSV = Path("data/processed/test_annotations.csv")


def label_name(name: str, idx: int) -> str:
    return name.strip() if name else f"class_{idx}"


@st.cache_resource(show_spinner=False)
def load_cached(weights_path: str, meta_path: str, device: str, num_classes: int):
    return load_bundle(
        weights_path=Path(weights_path),
        cars_meta_mat=Path(meta_path),
        device=device,
        num_classes=int(num_classes),
    )


@st.cache_data(show_spinner=False)
def load_bbox_cached(csv_path: str) -> Dict[str, Tuple[int, int, int, int]]:
    return load_bbox_map_from_csv(Path(csv_path))


st.title("🚗 Car Brand Classifier")
st.caption("Upload фото → Top-1 + Top-K. Для cars_test можна ввімкнути bbox-crop (як на тренуванні).")

with st.sidebar:
    st.header("⚙️ Settings")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = st.selectbox("Device", ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"], index=0)

    num_classes = st.number_input("Num classes", min_value=2, max_value=1000, value=196, step=1)
    topk = st.slider("Top-K", 1, 10, 5)
    size = st.selectbox("Input size", [224, 256, 299, 384], index=0)

    st.divider()
    weights_path = st.text_input("Weights path", value=str(DEFAULT_WEIGHTS))
    meta_path = st.text_input("cars_meta.mat path", value=str(DEFAULT_META))

    st.divider()
    if st.button("Clear cache"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.success("Cache cleared")

if not Path(weights_path).exists():
    st.error(f"Weights not found: {weights_path}")
    st.stop()

bundle = load_cached(weights_path, meta_path, device, int(num_classes))
st.success("Model loaded ✅")

tab1, tab2 = st.tabs(["📤 Upload any photo", "🧪 cars_test (bbox mode)"])

# ---------------- Tab 1: any upload ----------------
with tab1:
    st.subheader("Upload any photo (без bbox)")
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "webp", "tiff"])

    if uploaded:
        img = open_image(uploaded)
        c1, c2 = st.columns([1, 1.2], gap="large")
        with c1:
            st.image(img, caption="Input", use_container_width=True)
        with c2:
            if st.button("🚀 Predict", type="primary", key="predict_upload"):
                preds = predict_pil(bundle, img, device=device, topk=int(topk), size=int(size), bbox=None)
                best_idx, best_prob, best_name = preds[0]

                st.markdown("### 🎯 Top-1")
                st.metric(label_name(best_name, best_idx), f"{best_prob*100:.2f}%")

                st.markdown("### Top-K")
                rows = [{"label": label_name(n, i), "prob_%": p * 100.0} for i, p, n in preds]
                st.dataframe(rows, hide_index=True, use_container_width=True)
                st.bar_chart({r["label"]: r["prob_%"] for r in rows})
    else:
        st.info("Upload an image to get prediction.")

with tab2:
    st.subheader("cars_test (bbox-crop як на тренуванні)")

    folder = Path(st.text_input("cars_test folder", value=str(DEFAULT_TEST_FOLDER)))
    csv_path = Path(st.text_input("test_annotations.csv", value=str(DEFAULT_TEST_CSV)))
    use_bbox = st.toggle("Use bbox crop (recommended)", value=True)

    if not folder.exists():
        st.warning("Folder not found. Перевір шлях.")
        st.stop()

    # load bbox map only if needed
    bbox_map: Dict[str, Tuple[int, int, int, int]] = {}
    if use_bbox:
        if not csv_path.exists():
            st.warning("CSV з bbox не знайдено. Вимкни bbox або вкажи правильний шлях.")
        else:
            bbox_map = load_bbox_cached(str(csv_path))

    images = sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        st.warning("No images found in folder.")
        st.stop()

    pick = st.selectbox("Choose image", options=images, format_func=lambda p: p.name)
    img = Image.open(pick).convert("RGB")

    bbox = bbox_map.get(pick.name) if use_bbox else None

    c1, c2 = st.columns([1, 1.2], gap="large")
    with c1:
        st.image(img, caption="Original", use_container_width=True)
        if bbox is not None:
            st.image(crop_bbox(img, bbox), caption="Cropped by bbox (as in training)", use_container_width=True)

    with c2:
        if st.button("🚀 Predict (cars_test)", type="primary", key="predict_test"):
            preds = predict_pil(bundle, img, device=device, topk=int(topk), size=int(size), bbox=bbox)
            best_idx, best_prob, best_name = preds[0]

            st.markdown("### 🎯 Top-1")
            st.metric(label_name(best_name, best_idx), f"{best_prob*100:.2f}%")

            st.markdown("### Top-K")
            rows = [{"label": label_name(n, i), "prob_%": p * 100.0} for i, p, n in preds]
            st.dataframe(rows, hide_index=True, use_container_width=True)
            st.bar_chart({r["label"]: r["prob_%"] for r in rows})
