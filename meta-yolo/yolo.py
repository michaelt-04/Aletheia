#!/usr/bin/env python3
"""
yolo.py - YOLO26 ExecuTorch Test Script for Project Aletheia
=============================================================

Tests the YOLO26 XNNPACK model on a static image (no camera needed).
Outputs detection results to the terminal so you can verify the model works
before integrating into the full Aletheia OS pipeline.

Usage:
    python yolo.py                            # Uses default test image path
    python yolo.py --image path/to/image.jpg  # Specify your own image
    python yolo.py --image path/to/image.jpg --save  # Save annotated image

Requirements:
    pip install executorch opencv-python numpy torch
"""

import argparse
import time
import sys
import os
import numpy as np
import torch

try:
    import cv2
except ImportError:
    print("ERROR: OpenCV not found. Install with: pip install opencv-python")
    sys.exit(1)

try:
    from executorch.runtime import Runtime
except ImportError:
    print("ERROR: ExecuTorch not found. Install with: pip install executorch")
    sys.exit(1)


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


# --- Model Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo26n_xnnpack.pte")
INPUT_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25


def preprocess(image: np.ndarray):
    """
    Preprocess an image for YOLO26 inference.

    Key requirements (from model README):
        - Input: float32 in [0, 1] range (normalized)
        - Format: NCHW [1, 3, H, W]
        - Tensor MUST be .contiguous() or outputs will be wrong
        - Image should be RGB

    Returns:
        input_tensor: torch.Tensor, contiguous, shape [1, 3, 640, 640], float32, range [0, 1]
        scale_info:   tuple of (scale, (pad_w, pad_h)) for mapping coords back
    """
    h, w = image.shape[:2]

    # Calculate letterbox scaling
    scale = min(INPUT_SIZE / h, INPUT_SIZE / w)
    new_w, new_h = int(w * scale), int(h * scale)
    pad_w = (INPUT_SIZE - new_w) / 2
    pad_h = (INPUT_SIZE - new_h) / 2

    # Resize with aspect ratio preserved
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded canvas (114 grey is YOLO standard padding)
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    top, left = int(pad_h), int(pad_w)
    canvas[top:top + new_h, left:left + new_w] = resized

    # BGR -> RGB, normalize to [0, 1], HWC -> CHW, add batch dim
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    img_float = rgb.astype(np.float32) / 255.0  # Normalize to [0, 1]

    # Convert to tensor: permute creates non-contiguous memory, so .contiguous() is CRITICAL
    tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).contiguous()

    return tensor, (scale, (pad_w, pad_h))


def postprocess(output, scale_info, orig_shape, conf_threshold=CONFIDENCE_THRESHOLD):
    """
    Parse YOLO26 output into detections.

    YOLO26 NMS-free output shape: [1, 300, 6]
        - 300 candidate detections (max)
        - 6 values per detection: [cx, cy, w, h, confidence, class_id]
        - Box format is CENTER x, CENTER y, WIDTH, HEIGHT (not x1,y1,x2,y2)

    Returns:
        List of dicts with keys: label, confidence, box, class_id, carbon_impact
    """
    # Handle output format
    if isinstance(output, (list, tuple)):
        raw = output[0]
    else:
        raw = output

    # Convert to numpy
    if hasattr(raw, 'numpy'):
        raw = raw.numpy()
    raw = np.array(raw, dtype=np.float32)

    # Debug info
    print(f"  Raw output shape: {raw.shape}")
    if raw.size > 0:
        print(f"  Value range: [{raw.min():.4f}, {raw.max():.4f}]")

    # Remove batch dimension: [1, 300, 6] -> [300, 6]
    if raw.ndim == 3:
        raw = raw[0]

    # Parse columns: [cx, cy, w, h, confidence, class_id]
    cx = raw[:, 0]
    cy = raw[:, 1]
    bw = raw[:, 2]
    bh = raw[:, 3]
    confidences = raw[:, 4]
    class_ids = np.round(raw[:, 5]).astype(int)

    # Convert center format to corner format: cx,cy,w,h -> x1,y1,x2,y2
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    print(f"  Confidence range: [{confidences.min():.6f}, {confidences.max():.6f}]")
    print(f"  Detections above {conf_threshold}: {(confidences >= conf_threshold).sum()}")

    # Filter by confidence
    mask = confidences >= conf_threshold
    x1, y1, x2, y2 = x1[mask], y1[mask], x2[mask], y2[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]

    # Map coordinates back to original image space
    scale, (pad_w, pad_h) = scale_info
    orig_h, orig_w = orig_shape[:2]

    detections = []
    for i in range(len(confidences)):
        # Remove padding offset and rescale to original image
        bx1 = max(0, (x1[i] - pad_w) / scale)
        by1 = max(0, (y1[i] - pad_h) / scale)
        bx2 = min(orig_w, (x2[i] - pad_w) / scale)
        by2 = min(orig_h, (y2[i] - pad_h) / scale)

        # Get class name (clamp to valid range)
        cls_id = int(np.clip(class_ids[i], 0, len(COCO_CLASSES) - 1))
        label = COCO_CLASSES[cls_id]

        # Get carbon impact category
        impact = CARBON_IMPACT.get(label, "unknown")

        detections.append({
            "label": label,
            "confidence": float(confidences[i]),
            "box": (int(bx1), int(by1), int(bx2), int(by2)),
            "class_id": cls_id,
            "carbon_impact": impact,
        })

    # Sort by confidence (highest first)
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def draw_detections(image, detections):
    """Draw bounding boxes and labels on the image (for --save mode)."""
    impact_colors = {
        "high": (0, 0, 255),     # Red
        "medium": (0, 165, 255), # Orange
        "low": (0, 200, 0),      # Green
        "unknown": (200, 200, 0) # Cyan
    }

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = det["label"]
        conf = det["confidence"]
        impact = det["carbon_impact"]
        color = impact_colors.get(impact, (255, 255, 255))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = f"{label} {conf:.2f} [{impact}]"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return image


def main():
    parser = argparse.ArgumentParser(description="Test YOLO26 ExecuTorch model")
    parser.add_argument("--image", type=str, default="test.jpg",
                        help="Path to test image (default: test.jpg)")
    parser.add_argument("--model", type=str, default=MODEL_PATH,
                        help="Path to .pte model file")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE_THRESHOLD,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--save", action="store_true",
                        help="Save annotated image as 'output.jpg'")
    args = parser.parse_args()

    print("=" * 60)
    print("  YOLO26 ExecuTorch Test - Project Aletheia")
    print("=" * 60)

    # --- Step 1: Load Model ---
    print(f"\n[1/4] Loading model: {args.model}")
    if not os.path.exists(args.model):
        print(f"  ERROR: Model file not found at '{args.model}'")
        sys.exit(1)

    t0 = time.time()
    runtime = Runtime.get()
    program = runtime.load_program(args.model)
    method = program.load_method("forward")
    load_time = time.time() - t0
    print(f"  Model loaded successfully in {load_time:.2f}s")

    # --- Step 2: Load Image ---
    print(f"\n[2/4] Loading image: {args.image}")
    if not os.path.exists(args.image):
        print(f"  ERROR: Image not found at '{args.image}'")
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"  ERROR: Could not read image '{args.image}'")
        sys.exit(1)
    print(f"  Image size: {image.shape[1]}x{image.shape[0]} (WxH)")

    # --- Step 3: Run Inference ---
    print(f"\n[3/4] Running inference...")
    input_tensor, scale_info = preprocess(image)
    print(f"  Input tensor shape: {input_tensor.shape}")
    print(f"  Input tensor dtype: {input_tensor.dtype}")
    print(f"  Input value range: [{input_tensor.min().item():.4f}, {input_tensor.max().item():.4f}]")
    print(f"  Tensor is contiguous: {input_tensor.is_contiguous()}")

    # Warm-up run
    print(f"  Warm-up run...")
    _ = method.execute([input_tensor])

    # Timed run
    print(f"  Timed inference run...")
    t0 = time.time()
    output = method.execute([input_tensor])
    inference_time = time.time() - t0
    print(f"  Inference completed in {inference_time * 1000:.1f}ms")
    print(f"  Estimated FPS: {1.0 / inference_time:.1f}")

    # Average FPS over 5 runs
    print(f"  Running 5 more iterations for average FPS...")
    times = []
    for _ in range(5):
        t0 = time.time()
        method.execute([input_tensor])
        times.append(time.time() - t0)
    avg_time = sum(times) / len(times)
    print(f"  Average inference: {avg_time * 1000:.1f}ms ({1.0 / avg_time:.1f} FPS)")

    # --- Step 4: Process Results ---
    print(f"\n[4/4] Processing detections (threshold: {args.confidence})...")
    detections = postprocess(output, scale_info, image.shape, args.confidence)

    # --- Print Results ---
    print("\n" + "=" * 60)
    print(f"  RESULTS: {len(detections)} object(s) detected")
    print("=" * 60)

    if len(detections) == 0:
        print("  No objects detected above confidence threshold.")
        print("  Try: --confidence 0.1")
    else:
        print(f"  {'#':<4} {'Label':<18} {'Conf':<10} {'Box (x1,y1,x2,y2)':<28} {'Carbon'}")
        print(f"  {'-'*4} {'-'*18} {'-'*10} {'-'*28} {'-'*8}")

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det["box"]
            print(f"  {i+1:<4} {det['label']:<18} {det['confidence']:<10.4f} "
                  f"({x1:>4}, {y1:>4}, {x2:>4}, {y2:>4})      {det['carbon_impact']}")

    # Carbon summary
    if detections:
        print(f"\n  --- Carbon Impact Summary ---")
        counts = {"high": 0, "medium": 0, "low": 0, "unknown": 0}
        for d in detections:
            counts[d["carbon_impact"]] = counts.get(d["carbon_impact"], 0) + 1
        print(f"  High impact:    {counts['high']}")
        print(f"  Medium impact:  {counts['medium']}")
        print(f"  Low impact:     {counts['low']}")
        print(f"  Uncategorized:  {counts['unknown']}")

        total = len(detections)
        carbon_velocity = min(1.0,
            (counts["high"] * 0.3 + counts["medium"] * 0.15 + counts["low"] * 0.02) / max(total, 1)
        )
        print(f"\n  Estimated Carbon Velocity: {carbon_velocity:.2f}")

    # Save annotated image
    if args.save:
        annotated = draw_detections(image.copy(), detections) if detections else image.copy()
        output_path = os.path.join(os.path.dirname(os.path.abspath(args.image)), "output.jpg")
        cv2.imwrite(output_path, annotated)
        print(f"\n  Annotated image saved to: {output_path}")

    print("\n" + "=" * 60)
    print("  Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
