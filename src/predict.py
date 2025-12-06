# src/predict.py
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms

try:
    from scipy.io import loadmat
except Exception:
    loadmat = None  # optional


DEFAULT_WEIGHTS = Path("models/car_brand_detector_resnet50.pth")
DEFAULT_META = Path("data/dataset/car_devkit/devkit/cars_meta.mat")


@dataclass
class ModelBundle:
    model: nn.Module
    class_names: Optional[List[str]]


def _build_resnet50(num_classes: int) -> nn.Module:
    # weights=None щоб не тягнуло інтернет (state_dict все одно перезапише ваги)
    try:
        m = models.resnet50(weights=None)
    except TypeError:
        m = models.resnet50(pretrained=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _mat_to_str(x) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, np.ndarray):
        if x.dtype.kind in {"U", "S"}:
            return "".join(x.tolist()).strip()
        if x.size == 1:
            return str(x.squeeze())
    return str(x)


def load_class_names(cars_meta_mat: Path) -> Optional[List[str]]:
    if not cars_meta_mat or not cars_meta_mat.exists() or loadmat is None:
        return None
    meta = loadmat(str(cars_meta_mat))
    if "class_names" not in meta:
        return None
    raw = meta["class_names"].ravel()
    return [_mat_to_str(x) for x in raw]


def build_preprocess(size: int = 224) -> transforms.Compose:
    # як у тебе на val/test
    return transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def open_image(file_or_path) -> Image.Image:
    img = Image.open(file_or_path).convert("RGB")
    return ImageOps.exif_transpose(img)


def crop_bbox(img: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox
    w, h = img.size
    x1 = max(0, min(w - 1, int(x1)))
    y1 = max(0, min(h - 1, int(y1)))
    x2 = max(1, min(w, int(x2)))
    y2 = max(1, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))


def parse_bbox_str(bbox_str: str) -> Tuple[int, int, int, int]:
    # як у твоєму dataset.py (почистити np.uint8/np.uint16 тощо)
    s = str(bbox_str)
    for t in ["np.uint8(", "np.uint16(", "np.int64(", "np.int32(", "np.float32(", "np.float64("]:
        s = s.replace(t, "")
    s = s.replace(")", "")
    bbox = ast.literal_eval(s)
    return int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])


def load_bbox_map_from_csv(csv_path: Path) -> Dict[str, Tuple[int, int, int, int]]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    # expected columns: fname, bbox, ...
    out: Dict[str, Tuple[int, int, int, int]] = {}
    for _, row in df.iterrows():
        fname = str(row.iloc[0])
        bbox_str = row.iloc[1]
        out[fname] = parse_bbox_str(bbox_str)
    return out


def load_bundle(
    weights_path: Path = DEFAULT_WEIGHTS,
    cars_meta_mat: Path = DEFAULT_META,
    device: Optional[str] = None,
    num_classes: int = 196,
) -> ModelBundle:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = _build_resnet50(num_classes=num_classes)

    state = torch.load(weights_path, map_location="cpu")  # ти зберіг state_dict
    model.load_state_dict(state, strict=True)

    model.to(dev)
    model.eval()

    class_names = load_class_names(cars_meta_mat)
    return ModelBundle(model=model, class_names=class_names)


@torch.inference_mode()
def predict_pil(
    bundle: ModelBundle,
    image: Image.Image,
    device: Optional[str] = None,
    topk: int = 5,
    size: int = 224,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> List[Tuple[int, float, str]]:
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    if bbox is not None:
        image = crop_bbox(image, bbox)

    x = build_preprocess(size)(image).unsqueeze(0).to(dev)
    logits = bundle.model(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    k = min(topk, probs.numel())
    vals, idxs = torch.topk(probs, k)

    out: List[Tuple[int, float, str]] = []
    for idx, val in zip(idxs.tolist(), vals.tolist()):
        name = ""
        if bundle.class_names and 0 <= idx < len(bundle.class_names):
            name = bundle.class_names[idx]
        out.append((int(idx), float(val), name))
    return out
