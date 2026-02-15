# blazepalm_engine.py
# BlazeHand Tracker Engine for Project Aletheia
#
# Two-stage hand tracking pipeline using ExecuTorch:
#   Stage 1: BlazePalm — detect hand bounding boxes (256x256 input)
#   Stage 2: BlazeHand — predict 21 hand landmarks (256x256 cropped hand)
#
# Extracted from blazepalm_executorch/test_blazehand_mac.py and adapted
# for RGB camera input (Picamera2 / webcam fallback).

import os
import math
import numpy as np
import cv2
import torch
from executorch.runtime import Runtime

# --- Landmark indices ---
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20


class BlazeHandTracker:
    """
    Hand tracker using BlazePalm (detection) + BlazeHand (landmarks).
    ExecuTorch .pte models only.

    Input:  RGB image (H, W, 3) uint8
    Output: dict with landmarks_px, confidence, palm_score — or None
    """

    def __init__(self, palm_model_path, hand_model_path, anchors_path,
                 palm_score_thresh=0.5, palm_nms_thresh=0.3):
        if not os.path.exists(palm_model_path):
            raise FileNotFoundError(f"Palm model not found: {palm_model_path}")
        if not os.path.exists(hand_model_path):
            raise FileNotFoundError(f"Hand model not found: {hand_model_path}")
        if not os.path.exists(anchors_path):
            raise FileNotFoundError(f"Anchors file not found: {anchors_path}")

        self.anchors = np.load(anchors_path).astype(np.float32)
        self.palm_score_thresh = palm_score_thresh
        self.palm_nms_thresh = palm_nms_thresh
        print(f"[BlazeHand] Anchors: {self.anchors.shape}")

        # Load palm detector
        print(f"[BlazeHand] Loading palm detector: {os.path.basename(palm_model_path)}")
        runtime = Runtime.get()
        self._palm_prog = runtime.load_program(palm_model_path)
        self._palm_method = self._palm_prog.load_method("forward")

        # Load hand landmark model
        print(f"[BlazeHand] Loading hand landmark: {os.path.basename(hand_model_path)}")
        self._hand_prog = runtime.load_program(hand_model_path)
        self._hand_method = self._hand_prog.load_method("forward")

        # Warm up both models
        dummy = torch.zeros(1, 3, 256, 256, dtype=torch.float32).contiguous()
        self._palm_method.execute([dummy])
        self._hand_method.execute([dummy])
        print("[BlazeHand] Models loaded and warmed up.")

    # ----- Palm Detection -----

    def _preprocess_palm(self, image):
        """Pad to square, resize to 256x256, normalize to [-1, 1]."""
        h, w = image.shape[:2]
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        padded = np.full((size, size, 3), 128, dtype=np.uint8)
        padded[pad_h:pad_h + h, pad_w:pad_w + w] = image
        resized = cv2.resize(padded, (256, 256), interpolation=cv2.INTER_LINEAR)

        # Input is already RGB from camera — no cvtColor needed
        tensor = resized.astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).contiguous()
        return tensor, {"orig_h": h, "orig_w": w, "pad_h": pad_h, "pad_w": pad_w, "size": size}

    def _decode_palms(self, raw_scores, raw_boxes):
        """Decode palm detections using anchors + NMS."""
        scores = raw_scores[0, :, 0]
        boxes = raw_boxes[0]
        with np.errstate(over='ignore'):
            scores = 1.0 / (1.0 + np.exp(-np.clip(scores, -100, 100)))

        mask = scores >= self.palm_score_thresh
        if not mask.any():
            return []

        scores = scores[mask]
        boxes = boxes[mask]
        anchors = self.anchors[mask]

        cx = boxes[:, 0] / 256.0 + anchors[:, 0]
        cy = boxes[:, 1] / 256.0 + anchors[:, 1]
        bw = boxes[:, 2] / 256.0
        bh = boxes[:, 3] / 256.0

        kps = []
        for i in range(7):
            kp_x = boxes[:, 4 + i * 2] / 256.0 + anchors[:, 0]
            kp_y = boxes[:, 4 + i * 2 + 1] / 256.0 + anchors[:, 1]
            kps.append(np.stack([kp_x, kp_y], axis=-1))

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        indices = self._nms(np.stack([x1, y1, x2, y2], axis=-1), scores, self.palm_nms_thresh)

        dets = []
        for idx in indices:
            dets.append({
                "box": np.array([x1[idx], y1[idx], x2[idx], y2[idx]]),
                "score": float(scores[idx]),
                "keypoints": np.array([kp[idx] for kp in kps]),
            })
        return dets

    def _nms(self, boxes, scores, thresh):
        """Non-maximum suppression."""
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            remaining = np.where(iou <= thresh)[0]
            order = order[remaining + 1]
        return keep

    # ----- Hand Crop -----

    def _crop_hand(self, image, palm):
        """Crop and rotate hand region using palm keypoints."""
        h, w = image.shape[:2]
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2

        kps = palm["keypoints"]
        wrist = kps[0]
        middle_mcp = kps[2]

        center_x = (wrist[0] + middle_mcp[0]) / 2
        center_y = (wrist[1] + middle_mcp[1]) / 2
        hand_size = math.hypot(middle_mcp[0] - wrist[0], middle_mcp[1] - wrist[1]) * 2.5

        angle = math.atan2(middle_mcp[1] - wrist[1], middle_mcp[0] - wrist[0])
        angle_deg = math.degrees(angle) - 90

        cx_px = center_x * size
        cy_px = center_y * size
        hand_size_px = max(hand_size * size, 1)

        M = cv2.getRotationMatrix2D((cx_px, cy_px), angle_deg, 1.0)
        scale = 256.0 / hand_size_px
        M[0, 0] *= scale; M[0, 1] *= scale
        M[1, 0] *= scale; M[1, 1] *= scale
        M[0, 2] = M[0, 2] * scale + 128 - cx_px * scale
        M[1, 2] = M[1, 2] * scale + 128 - cy_px * scale

        padded = np.full((size, size, 3), 128, dtype=np.uint8)
        padded[pad_h:pad_h + h, pad_w:pad_w + w] = image
        crop = cv2.warpAffine(padded, M, (256, 256), borderValue=(128, 128, 128))

        return crop, {"M": M, "pad_h": pad_h, "pad_w": pad_w,
                      "orig_h": h, "orig_w": w, "size": size}

    # ----- Landmark Prediction -----

    def _predict_landmarks(self, hand_crop):
        """Run hand landmark model on cropped hand image."""
        # Input is already RGB — no cvtColor needed
        tensor = hand_crop.astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).contiguous()

        output = self._hand_method.execute([tensor])

        # Output: (hand_flag [1], handed [1], landmarks [1, 21, 3])
        if isinstance(output, (list, tuple)) and len(output) >= 3:
            flag_raw = output[0]
            lm_raw = output[2]
        elif isinstance(output, (list, tuple)) and len(output) == 2:
            flag_raw = output[0]
            lm_raw = output[1]
        else:
            lm_raw = output[0] if isinstance(output, (list, tuple)) else output
            flag_raw = None

        if hasattr(lm_raw, 'numpy'):
            lm_raw = lm_raw.numpy()
        lm = np.array(lm_raw, dtype=np.float32)

        if lm.ndim == 3 and lm.shape[1] == 21:
            landmarks = lm[0]
        elif lm.size >= 63:
            landmarks = lm.flatten()[:63].reshape(21, 3)
        else:
            landmarks = np.zeros((21, 3))

        flag = 0.5
        if flag_raw is not None:
            if hasattr(flag_raw, 'numpy'):
                flag_raw = flag_raw.numpy()
            flag = float(np.array(flag_raw).flatten()[0])

        return landmarks, flag

    def _landmarks_to_original(self, landmarks, transform):
        """Map landmarks from crop space back to original image coordinates."""
        M_inv = cv2.invertAffineTransform(transform["M"])
        pts = landmarks[:, :2].copy() * 256.0
        ones = np.ones((pts.shape[0], 1))
        pts_h = np.hstack([pts, ones])
        orig_pts = pts_h @ M_inv.T
        orig_pts[:, 0] -= transform["pad_w"]
        orig_pts[:, 1] -= transform["pad_h"]
        return orig_pts

    # ----- Main Detection -----

    def detect(self, image, cached_palm=None):
        """
        Full pipeline: image -> hand dict with landmarks, or None.

        Args:
            image: RGB image (H, W, 3) uint8
            cached_palm: if provided, skip palm detection and reuse this palm

        Returns:
            dict with landmarks_px [21,2], confidence, palm_score, _palm — or None
        """
        if cached_palm is not None:
            palm = cached_palm
        else:
            tensor, scale_info = self._preprocess_palm(image)
            palm_out = self._palm_method.execute([tensor])

            # Parse: (regressions [1,N,18], classifications [1,N,1])
            if isinstance(palm_out, (list, tuple)) and len(palm_out) >= 2:
                rb = palm_out[0]
                rs = palm_out[1]
            else:
                return None

            if hasattr(rs, 'numpy'): rs = rs.numpy()
            if hasattr(rb, 'numpy'): rb = rb.numpy()
            rs = np.array(rs, dtype=np.float32)
            rb = np.array(rb, dtype=np.float32)

            palms = self._decode_palms(rs, rb)
            if not palms:
                return None

            palm = palms[0]

        crop, transform = self._crop_hand(image, palm)
        lm, flag = self._predict_landmarks(crop)

        if flag < 0.3:
            return None

        lm_px = self._landmarks_to_original(lm, transform)

        return {
            "landmarks_px": lm_px,
            "confidence": float(flag),
            "palm_score": palm["score"],
            "_palm": palm,
        }
