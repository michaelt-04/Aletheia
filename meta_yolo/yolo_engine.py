"""
yolo_engine.py - YOLO26 ExecuTorch Detection Engine for Project Aletheia
=========================================================================

Reusable module for YOLO26 object detection via ExecuTorch.

This version adds SAFE label filtering to reduce false positives:
- Keeps ONLY "vampire electricity" / electronics / small appliances
- Drops transport (car/airplane/etc.), animals, plants, food, sports gear, etc.

IMPORTANT:
- DO NOT filter or reorder COCO_CLASSES (class-id mapping must remain intact).
- Filtering is done AFTER inference in postprocess().
"""

import os
import numpy as np
import torch
import cv2

from executorch.runtime import Runtime


# --- COCO 80 Class Names (DO NOT FILTER THIS LIST) ---
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# --- Carbon impact categories for Aletheia ---
# NOTE: COCO doesn't include "charger" or "PC monitor" explicitly.
# Best approximation is "tv" for monitor/display, plus laptop/keyboard/mouse/phone.
CARBON_IMPACT = {
    # High impact - electronics and appliances
    "tv": "high",            # treat as monitor/display/TV
    "laptop": "high",
    "cell phone": "high",
    "microwave": "high",
    "oven": "high",
    "toaster": "high",
    "refrigerator": "high",
    "hair drier": "high",

    # Medium impact - consumer electronics / accessories
    "keyboard": "medium",
    "mouse": "medium",
    "remote": "medium",

    # Optional: keep person if you want presence to affect UI (set to low)
    "person": "low",
}

# --- STRICT allowlist: only keep labels relevant to vampire electricity/appliances ---
# This is the main false-positive reducer.
VAMPIRE_ELECTRICITY_LABELS = {
    # displays / electronics
    "tv",
    "laptop",
    "cell phone",
    "keyboard",
    "mouse",
    "remote",

    # small / major appliances
    "microwave",
    "oven",
    "toaster",
    "refrigerator",
    "hair drier",

    # Optional: keep "person" if you still want it detected
    "person",
}

# If True, we only keep labels in VAMPIRE_ELECTRICITY_LABELS
STRICT_ALLOWLIST = True


class YOLODetector:
    """
    YOLO26 object detector using ExecuTorch runtime.

    Handles preprocessing, inference, and postprocessing in one clean API.
    Applies safe filtering in postprocess() (does not affect class-id mapping).
    """

    def __init__(self, model_path, input_size=640, confidence_threshold=0.25):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)

        runtime = Runtime.get()
        self.program = runtime.load_program(model_path)
        self.method = self.program.load_method("forward")

        # Warm up
        dummy = torch.zeros(1, 3, self.input_size, self.input_size, dtype=torch.float32).contiguous()
        self.method.execute([dummy])

        print(f"[YOLODetector] Model loaded: {os.path.basename(model_path)}")

    def preprocess(self, image):
        h, w = image.shape[:2]

        # Letterbox resize
        scale = min(self.input_size / h, self.input_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_w = (self.input_size - new_w) / 2
        pad_h = (self.input_size - new_h) / 2

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        top, left = int(pad_h), int(pad_w)
        canvas[top:top + new_h, left:left + new_w] = resized

        # Normalize to [0,1], HWC -> CHW, add batch, contiguous
        # Camera feeds RGB; skip cvtColor unless caller passes BGR
        img_float = canvas.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).contiguous()

        return tensor, (scale, (pad_w, pad_h))

    def postprocess(self, output, scale_info, orig_shape):
        if isinstance(output, (list, tuple)):
            raw = output[0]
        else:
            raw = output

        if hasattr(raw, "numpy"):
            raw = raw.numpy()
        raw = np.array(raw, dtype=np.float32)

        if raw.ndim == 3:
            raw = raw[0]  # [300, 6]

        confidences = raw[:, 4]
        class_ids = np.round(raw[:, 5]).astype(int)

        # Auto-detect output format: end2end models output [x1,y1,x2,y2],
        # standard models output [cx,cy,w,h]. In xyxy, col2 > col0 always.
        mask = confidences >= self.confidence_threshold
        if mask.any():
            v = raw[mask]
            is_xyxy = bool(np.all(v[:, 2] > v[:, 0]) and np.all(v[:, 3] > v[:, 1]))
        else:
            is_xyxy = False

        if not hasattr(self, '_format_logged'):
            self._format_logged = True
            fmt = "xyxy" if is_xyxy else "cxcywh"
            print(f"[YOLODetector] Output format: {fmt}")
            if mask.any():
                s = raw[mask][0]
                print(f"[YOLODetector]   Sample cols 0-3: {s[:4]}")

        if is_xyxy:
            x1, y1, x2, y2 = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        else:
            cx, cy, bw, bh = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2
        x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        scale, (pad_w, pad_h) = scale_info
        orig_h, orig_w = orig_shape[:2]

        detections = []
        for i in range(len(confidences)):
            cls_id = int(np.clip(class_ids[i], 0, len(COCO_CLASSES) - 1))
            label = COCO_CLASSES[cls_id]

            # STRICT filtering to reduce hallucinations
            if STRICT_ALLOWLIST and label not in VAMPIRE_ELECTRICITY_LABELS:
                continue

            impact = CARBON_IMPACT.get(label, "unknown")

            bx1 = max(0, (x1[i] - pad_w) / scale)
            by1 = max(0, (y1[i] - pad_h) / scale)
            bx2 = min(orig_w, (x2[i] - pad_w) / scale)
            by2 = min(orig_h, (y2[i] - pad_h) / scale)

            detections.append({
                "label": label,
                "confidence": float(confidences[i]),
                "box": (int(bx1), int(by1), int(bx2), int(by2)),
                "class_id": cls_id,
                "carbon_impact": impact,
            })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def detect(self, image, confidence_threshold=None):
        old_thresh = self.confidence_threshold
        if confidence_threshold is not None:
            self.confidence_threshold = float(confidence_threshold)

        tensor, scale_info = self.preprocess(image)
        output = self.method.execute([tensor])
        detections = self.postprocess(output, scale_info, image.shape)

        self.confidence_threshold = old_thresh
        return detections

    @staticmethod
    def compute_carbon_velocity(detections):
        if not detections:
            return 0.0

        counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
        for d in detections:
            impact = d.get("carbon_impact", "unknown")
            counts[impact] = counts.get(impact, 0) + 1

        total = len(detections)
        velocity = min(
            1.0,
            (counts["high"] * 0.3 + counts["medium"] * 0.15 + counts["low"] * 0.02) / max(total, 1)
        )
        return velocity
