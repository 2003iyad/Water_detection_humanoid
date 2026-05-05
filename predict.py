"""
predict.py — Water-only inference.
Author: Iyad Laphir

Runs WaterSegNet on a single image: segments the water region and
reports the fill level (0-4) from the model's classification head.

    result = predict("photo.jpg")

    result.image      — original image with water mask overlay + level badge
    result.mask       — binary mask (white = water, black = background)
    result.water_box  — (x1, y1, x2, y2) bounding box around water, or None
    result.level      — int 0-4  (from classifier head)
    result.level_name — e.g. "level-2"
    result.confidence — classifier confidence 0-1

Train the model first:
    python water_detection_2.py
"""

import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import WaterSegNet, CLASS_NAMES

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT  = os.path.join(BASE_DIR, "segmentation_checkpoints", "best.pth")

# ── Settings ───────────────────────────────────────────────────────────────────
DEVICE         = "cuda"
IMG_SIZE       = 416
MEAN           = [0.485, 0.456, 0.406]
STD            = [0.229, 0.224, 0.225]
MASK_THRESHOLD = 0.5
BOX_COVERAGE   = 0.92

# ── Colours (BGR for OpenCV) ───────────────────────────────────────────────────
WATER_COLOUR = (219, 152,  52)
BADGE_BG     = ( 40,  40,  40)

_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT OBJECT
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WaterResult:
    """
    Output of predict().

    Attributes:
        image      : BGR numpy array — original image with water overlay drawn.
        mask       : Binary numpy array (255 = water, 0 = background).
        water_box  : (x1, y1, x2, y2) bounding box in pixels, or None.
        level      : Predicted level int 0-4 (from classifier head).
        level_name : e.g. "level-2".
        confidence : Classifier confidence 0-1.
    """
    image:      np.ndarray
    mask:       np.ndarray
    water_box:  tuple | None
    level:      int
    level_name: str
    confidence: float


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING  (cached so repeated calls don't reload from disk)
# ══════════════════════════════════════════════════════════════════════════════

_model_cache = None


def _get_model(device: torch.device) -> WaterSegNet:
    global _model_cache
    if _model_cache is None:
        model = WaterSegNet().to(device)
        ckpt  = torch.load(CHECKPOINT, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        _model_cache = model
    return _model_cache


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _box_from_mask(mask_np: np.ndarray, coverage: float = BOX_COVERAGE):
    """
    Bounding box covering `coverage` fraction of white pixels.
    Trims (1-coverage)/2 from each edge using percentiles.
    Returns (x1, y1, x2, y2) or None.
    """
    ys, xs = np.where(mask_np > 0)
    if len(ys) == 0:
        return None
    trim = (1.0 - coverage) / 2.0 * 100
    return (
        int(np.percentile(xs, trim)),
        int(np.percentile(ys, trim)),
        int(np.percentile(xs, 100 - trim)),
        int(np.percentile(ys, 100 - trim)),
    )


def _draw_badge(img, x, y, text, colour, font, below=False):
    scale = 0.6
    pad   = 4
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    by = (y + pad) if below else max(0, y - th - pad * 2)
    cv2.rectangle(img, (x, by), (x + tw + pad*2, by + th + pad*2), colour, -1)
    cv2.putText(img, text, (x + pad, by + th + pad),
                font, scale, (255, 255, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PREDICT FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def predict(image_input, device_name: str = None) -> WaterResult:
    """
    Run WaterSegNet on a single image.

    Args:
        image_input : file path (str) OR BGR numpy array (from OpenCV).
        device_name : "cuda" or "cpu". Defaults to DEVICE setting above.

    Returns:
        WaterResult with .image, .mask, .water_box, .level, .level_name, .confidence
    """
    device = torch.device(
        (device_name or DEVICE) if torch.cuda.is_available() else "cpu"
    )
    model = _get_model(device)

    if isinstance(image_input, str):
        frame_bgr = cv2.imread(image_input)
        if frame_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_input}")
    else:
        frame_bgr = image_input

    orig_h, orig_w = frame_bgr.shape[:2]

    pil_image    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    input_tensor = _transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_logits, cls_logits = model(input_tensor)

    # Binary mask resized back to original dimensions
    prob   = torch.sigmoid(mask_logits[0, 0]).cpu().numpy()
    binary = (prob > MASK_THRESHOLD).astype(np.uint8) * 255
    water_mask = cv2.resize(binary, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Level from classification head
    cls_probs  = torch.softmax(cls_logits[0], dim=0).cpu()
    level      = int(cls_probs.argmax())
    confidence = float(cls_probs[level])
    level_name = CLASS_NAMES[level]

    water_box = _box_from_mask(water_mask)

    # ── Draw ──────────────────────────────────────────────────────────────────
    annotated = frame_bgr.copy()
    thickness = max(2, int(min(orig_w, orig_h) * 0.004))
    font      = cv2.FONT_HERSHEY_SIMPLEX

    # Blue tint overlay on water region
    overlay      = annotated.copy()
    water_region = water_mask > 0
    overlay[water_region] = (
        overlay[water_region] * 0.4 +
        np.array([200, 100, 30], dtype=np.float32) * 0.6
    ).astype(np.uint8)
    cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0, annotated)

    # Water bounding box
    if water_box is not None:
        cv2.rectangle(annotated,
                      (water_box[0], water_box[1]),
                      (water_box[2], water_box[3]),
                      WATER_COLOUR, thickness)
        _draw_badge(annotated, water_box[0], water_box[3],
                    "water", WATER_COLOUR, font, below=True)

    # Level + confidence badge — top left
    summary = f"{level_name}  {confidence:.0%}"
    (tw, th), _ = cv2.getTextSize(summary, font, 0.9, 2)
    pad = 8
    cv2.rectangle(annotated, (10, 10),
                  (10 + tw + pad*2, 10 + th + pad*2), BADGE_BG, -1)
    cv2.putText(annotated, summary, (10 + pad, 10 + th + pad),
                font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return WaterResult(
        image=annotated,
        mask=water_mask,
        water_box=water_box,
        level=level,
        level_name=level_name,
        confidence=confidence,
    )


# ── CLI convenience ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [output_path]")
        sys.exit(1)

    out_path = sys.argv[2] if len(sys.argv) > 2 else "water_output.jpg"
    result   = predict(sys.argv[1])

    print(f"Level      : {result.level_name}")
    print(f"Confidence : {result.confidence:.2%}")
    print(f"Water box  : {result.water_box}")

    cv2.imwrite(out_path, result.image)
    print(f"Saved to   : {out_path}")
