"""
yolo_engine.py - YOLO26 ExecuTorch Detection Engine for Project Aletheia
=========================================================================

Reusable module for YOLO26 object detection via ExecuTorch.
Used by both the standalone test script (yolo.py) and the live camera
pipeline (yolo_live.py / aletheia_os.py).

Usage:
    from yolo_engine import YOLODetector

    detector = YOLODetector("yolo26n_xnnpack.pte")
    detections = detector.detect(cv2_bgr_image)
    for det in detections:
        print(det["label"], det["confidence"], det["box"])
"""

import os
import numpy as np
import torch
import cv2

from executorch.runtime import Runtime


# --- COCO 80 Class Names ---
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
CARBON_IMPACT = {
    # High impact - electronics and appliances
    "tv": "high", "laptop": "high", "cell phone": "high", "microwave": "high",
    "oven": "high", "toaster": "high", "refrigerator": "high", "hair drier": "high",
    # Medium impact - transportation and consumer goods
    "car": "medium", "motorcycle": "medium", "bus": "medium", "truck": "medium",
    "airplane": "medium", "train": "medium", "bottle": "medium", "cup": "medium",
    "backpack": "medium", "suitcase": "medium", "handbag": "medium",
    # Low impact - natural / organic
    "person": "low", "dog": "low", "cat": "low", "bird": "low",
    "banana": "low", "apple": "low", "orange": "low", "broccoli": "low",
    "carrot": "low", "potted plant": "low",
}


class YOLODetector:
    """
    YOLO26 object detector using ExecuTorch runtime.

    Handles preprocessing, inference, and postprocessing in one clean API.
    """

    def __init__(self, model_path, input_size=640, confidence_threshold=0.25):
        """
        Load the YOLO26 ExecuTorch model.

        Args:
            model_path: Path to the .pte model file
            input_size: Input resolution (default 640)
            confidence_threshold: Minimum confidence to return a detection
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.input_size = input_size
        self.confidence_threshold = confidence_threshold

        # Load ExecuTorch model
        runtime = Runtime.get()
        self.program = runtime.load_program(model_path)
        self.method = self.program.load_method("forward")

        # Warm up the model (first run is slower)
        dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32).contiguous()
        self.method.execute([dummy])

        print(f"[YOLODetector] Model loaded: {os.path.basename(model_path)}")

    def preprocess(self, image):
        """
        Preprocess a BGR image for YOLO26 inference.

        Key requirements (from model README):
            - float32 in [0, 1] range
            - NCHW format [1, 3, H, W]
            - Tensor MUST be .contiguous()

        Returns:
            (tensor, scale_info) where scale_info is used by postprocess
        """
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

        # BGR -> RGB, normalize to [0,1], HWC -> CHW, add batch, make CONTIGUOUS
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        img_float = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).contiguous()

        return tensor, (scale, (pad_w, pad_h))

    def postprocess(self, output, scale_info, orig_shape):
        """
        Parse YOLO26 NMS-free output [1, 300, 6] -> list of detections.

        Output format: [cx, cy, w, h, confidence, class_id]
        """
        if isinstance(output, (list, tuple)):
            raw = output[0]
        else:
            raw = output

        if hasattr(raw, 'numpy'):
            raw = raw.numpy()
        raw = np.array(raw, dtype=np.float32)

        if raw.ndim == 3:
            raw = raw[0]  # Remove batch dim -> [300, 6]

        # Parse: [cx, cy, w, h, confidence, class_id]
        cx, cy, bw, bh = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
        confidences = raw[:, 4]
        class_ids = np.round(raw[:, 5]).astype(int)

        # Center format -> corner format
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # Filter by confidence
        mask = confidences >= self.confidence_threshold
        x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        # Map coordinates back to original image
        scale, (pad_w, pad_h) = scale_info
        orig_h, orig_w = orig_shape[:2]

        detections = []
        for i in range(len(confidences)):
            bx1 = max(0, (x1[i] - pad_w) / scale)
            by1 = max(0, (y1[i] - pad_h) / scale)
            bx2 = min(orig_w, (x2[i] - pad_w) / scale)
            by2 = min(orig_h, (y2[i] - pad_h) / scale)

            cls_id = int(np.clip(class_ids[i], 0, len(COCO_CLASSES) - 1))
            label = COCO_CLASSES[cls_id]
            impact = CARBON_IMPACT.get(label, "unknown")

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
        """
        Run full detection pipeline on a BGR image.

        Args:
            image: OpenCV BGR image (any size)
            confidence_threshold: Override default threshold (optional)

        Returns:
            List of detection dicts with keys:
                label, confidence, box (x1,y1,x2,y2), class_id, carbon_impact
        """
        old_thresh = self.confidence_threshold
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold

        tensor, scale_info = self.preprocess(image)
        output = self.method.execute([tensor])
        detections = self.postprocess(output, scale_info, image.shape)

        self.confidence_threshold = old_thresh
        return detections

    @staticmethod
    def compute_carbon_velocity(detections):
        """
        Compute a carbon velocity score (0.0 - 1.0) from detections.
        Used to drive the EcoSprite and GreyFog in the Aletheia HUD.
        """
        if not detections:
            return 0.0

        counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
        for d in detections:
            impact = d.get("carbon_impact", "unknown")
            counts[impact] = counts.get(impact, 0) + 1

        total = len(detections)
        velocity = min(1.0,
            (counts["high"] * 0.3 + counts["medium"] * 0.15 + counts["low"] * 0.02) / max(total, 1)
        )
        return velocity
