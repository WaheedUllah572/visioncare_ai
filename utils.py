# utils.py
# -----------------------------------------
# VisionCare AI utilities (Streamlit-friendly, no matplotlib)
# -----------------------------------------

import base64
from io import BytesIO
from typing import List, Dict, Union, Optional

import numpy as np
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
import streamlit as st

# -------------------------------
# OpenAI client (safe init)
# -------------------------------
try:
    from openai import OpenAI  # OpenAI SDK v1
except Exception:
    OpenAI = None  # allow app to load even if package missing

def _init_openai_client() -> Optional["OpenAI"]:
    """Initialize OpenAI client from Streamlit secrets, if available."""
    if OpenAI is None:
        return None
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None

_openai_client = _init_openai_client()

# -------------------------------
# Image helpers
# -------------------------------
def convert_image_to_base64(image_file) -> str:
    """Convert uploaded image (file-like) to base64 JPEG string."""
    image = Image.open(image_file).convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=92)
    img_bytes = buffered.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    return img_b64

def extract_text_from_image(image_file) -> str:
    """
    Use GPT-4o to extract text from an uploaded image.
    Returns plain text or an error message string (so UI won't crash).
    """
    if _openai_client is None:
        return "Error during OCR: OpenAI client not configured. Add OPENAI_API_KEY to st.secrets."

    img_b64 = convert_image_to_base64(image_file)
    prompt = (
        "Extract all readable text from this image. "
        "Focus on documents, signs, or labels. Return only the plain text."
    )

    try:
        res = _openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            max_tokens=1024
        )
        return res.choices[0].message.content or ""
    except Exception as e:
        return f"Error during OCR: {e}"

# -------------------------------
# Detection model
# -------------------------------
def load_detection_model():
    """
    Load a pre-trained Faster R-CNN (torchvision) and set eval mode.
    Stores labels map on the model for later use.
    """
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn,
        FasterRCNN_ResNet50_FPN_Weights,
    )
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    # stash labels map for later use
    model._labels_map = weights.meta.get("categories", [])
    return model

def detect_objects(
    image: Image.Image,
    model,
    threshold: float = 0.6
) -> List[Dict[str, Union[str, float, list, int]]]:
    """
    Run object detection and return a list of dicts:
    { "label": int, "box": [x1,y1,x2,y2], "score": float }
    Coordinates are pixel-space floats from the model; we'll round them when drawing.
    """
    to_tensor = T.ToTensor()
    img_tensor = to_tensor(image).unsqueeze(0)  # [1,3,H,W], float32[0..1]

    with torch.no_grad():
        out = model(img_tensor)[0]

    result = []
    boxes = out.get("boxes", [])
    scores = out.get("scores", [])
    labels = out.get("labels", [])

    for i in range(len(boxes)):
        score = float(scores[i].item())
        if score >= threshold:
            label_id = int(labels[i].item())
            box = [float(v) for v in boxes[i].tolist()]  # x1,y1,x2,y2
            result.append({"label": label_id, "box": box, "score": score})

    return result

# -------------------------------
# Drawing utilities (no matplotlib)
# -------------------------------
def _get_labels_map(model=None):
    """Try to get a readable labels map (COCO names)."""
    # Prefer the map stashed on the model
    if model is not None and hasattr(model, "_labels_map"):
        lm = getattr(model, "_labels_map")
        if isinstance(lm, list) and lm:
            return lm
    # Fallback to torchvision weights meta
    try:
        return torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta["categories"]
    except Exception:
        # last resort: numeric strings
        return [str(i) for i in range(1000)]

def draw_boxes(
    image: Image.Image,
    predictions: List[Dict[str, Union[str, float, list, int]]],
    model=None,
    color: Union[str, List[str]] = "red",
    width: int = 3,
    font_size: int = 16,
) -> Image.Image:
    """
    Draw bounding boxes + labels using torchvision.utils.draw_bounding_boxes.
    Returns a PIL.Image.
    - predictions: [{"label": int, "box":[x1,y1,x2,y2], "score": float}, ...]
    """
    if not predictions:
        return image

    labels_map = _get_labels_map(model)

    # Convert float boxes -> int pixel coords (required by draw_bounding_boxes)
    boxes_f = torch.tensor([p["box"] for p in predictions], dtype=torch.float32)
    boxes = boxes_f.round().to(torch.int64)  # [N, 4]

    # Build readable labels like "person 0.92"
    lbls: List[str] = []
    for p in predictions:
        lid = int(p["label"])
        name = labels_map[lid] if 0 <= lid < len(labels_map) else str(lid)
        score = float(p.get("score", 0.0))
        lbls.append(f"{name} {score:.2f}")

    # Convert PIL -> tensor uint8 [C,H,W]
    img_t = T.ToTensor()(image)
    if img_t.dtype != torch.uint8:
        img_t = (img_t * 255).clamp(0, 255).to(torch.uint8)

    img_boxed = draw_bounding_boxes(
        img_t,
        boxes,
        labels=lbls,
        colors=color,    # can be a single color or list per box
        width=width,
        font_size=font_size,
    )
    return F.to_pil_image(img_boxed)

# -------------------------------
# Convenience function to run everything
# -------------------------------
def run_detection_pipeline(
    image_file,
    model,
    threshold: float = 0.6
) -> Image.Image:
    """
    Load PIL image from file-like, run detection, draw boxes, and return boxed PIL image.
    """
    image = Image.open(image_file).convert("RGB")
    preds = detect_objects(image, model, threshold=threshold)
    boxed = draw_boxes(image.copy(), preds, model=model)
    return boxed
