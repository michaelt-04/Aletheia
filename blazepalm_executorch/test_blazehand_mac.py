#!/usr/bin/env python3
"""
test_blazehand_mac.py - Test BlazePalm + BlazeHand on Mac
==========================================================

Pinch thumb + index finger near the green dot to grab and drag it.

Run from the MediaPipePyTorch directory after exporting models:
    python test_blazehand_mac.py

Controls: 'q'=quit  'r'=reset dot  'd'=toggle debug
"""

import os
import sys
import time
import math
import numpy as np
import cv2
import torch

USE_EXECUTORCH = True
try:
    from executorch.runtime import Runtime
except ImportError:
    USE_EXECUTORCH = False
    print("WARNING: ExecuTorch not available, will try TorchScript fallback")

# --- Landmark indices ---
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


class BlazeHandTrackerMac:
    """
    Hand tracker using BlazePalm + BlazeHandLandmark.
    Supports both ExecuTorch .pte and TorchScript .pt files.
    
    This is the same tracker that was working for hand detection,
    with support for both model formats.
    """

    def __init__(self, palm_model_path, hand_model_path, anchors_path):
        self.anchors = np.load(anchors_path).astype(np.float32)
        self.palm_score_thresh = 0.5
        self.palm_nms_thresh = 0.3
        print(f"[BlazeHand] Anchors: {self.anchors.shape}")

        # Load palm detector
        print(f"[BlazeHand] Loading palm detector: {palm_model_path}")
        if palm_model_path.endswith(".pte") and USE_EXECUTORCH:
            runtime = Runtime.get()
            self._palm_prog = runtime.load_program(palm_model_path)
            self._palm_method = self._palm_prog.load_method("forward")
            self._palm_type = "pte"
        else:
            self._palm_jit = torch.jit.load(palm_model_path, map_location="cpu")
            self._palm_jit.eval()
            self._palm_type = "jit"

        # Load hand landmark model
        print(f"[BlazeHand] Loading hand landmark: {hand_model_path}")
        if hand_model_path.endswith(".pte") and USE_EXECUTORCH:
            if self._palm_type != "pte":
                runtime = Runtime.get()
            self._hand_prog = runtime.load_program(hand_model_path)
            self._hand_method = self._hand_prog.load_method("forward")
            self._hand_type = "pte"
        else:
            self._hand_jit = torch.jit.load(hand_model_path, map_location="cpu")
            self._hand_jit.eval()
            self._hand_type = "jit"

        # Warm up
        dummy = torch.zeros(1, 3, 256, 256, dtype=torch.float32).contiguous()
        self._run_palm(dummy)
        self._run_hand(dummy)
        print("[BlazeHand] Models loaded and warmed up!")

    def _run_palm(self, tensor):
        if self._palm_type == "pte":
            return self._palm_method.execute([tensor])
        with torch.no_grad():
            return self._palm_jit(tensor)

    def _run_hand(self, tensor):
        if self._hand_type == "pte":
            return self._hand_method.execute([tensor])
        with torch.no_grad():
            return self._hand_jit(tensor)

    # ----- Palm Detection -----

    def _preprocess_palm(self, image):
        h, w = image.shape[:2]
        size = max(h, w)
        pad_h = (size - h) // 2
        pad_w = (size - w) // 2
        padded = np.full((size, size, 3), 128, dtype=np.uint8)
        padded[pad_h:pad_h + h, pad_w:pad_w + w] = image
        resized = cv2.resize(padded, (256, 256), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).contiguous()
        return tensor, {"orig_h": h, "orig_w": w, "pad_h": pad_h, "pad_w": pad_w, "size": size}

    def _decode_palms(self, raw_scores, raw_boxes):
        """Decode palm detections using anchors + NMS."""
        scores = raw_scores[0, :, 0]
        boxes = raw_boxes[0]
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
        rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).contiguous()

        output = self._run_hand(tensor)

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

        # Already [1, 21, 3] and already /256 normalized
        if lm.ndim == 3 and lm.shape[1] == 21:
            landmarks = lm[0]
        elif lm.size >= 63:
            landmarks = lm.flatten()[:63].reshape(21, 3)
        else:
            landmarks = np.zeros((21, 3))

        # Parse flag (already sigmoid from model)
        flag = 0.5
        if flag_raw is not None:
            if hasattr(flag_raw, 'numpy'):
                flag_raw = flag_raw.numpy()
            flag = float(np.array(flag_raw).flatten()[0])

        return landmarks, flag

    def _landmarks_to_original(self, landmarks, transform):
        M_inv = cv2.invertAffineTransform(transform["M"])
        pts = landmarks[:, :2].copy() * 256.0
        ones = np.ones((pts.shape[0], 1))
        pts_h = np.hstack([pts, ones])
        orig_pts = pts_h @ M_inv.T
        orig_pts[:, 0] -= transform["pad_w"]
        orig_pts[:, 1] -= transform["pad_h"]
        return orig_pts

    # ----- Main Detection -----

    def detect(self, image):
        """Full pipeline: image -> hand dict with landmarks + pinch info, or None."""
        tensor, scale_info = self._preprocess_palm(image)
        palm_out = self._run_palm(tensor)

        # Parse: (regressions [1,2944,18], classifications [1,2944,1])
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

        # Take best palm only
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
        }


def find_model(pte, pt):
    if os.path.exists(pte): return pte
    if os.path.exists(pt): return pt
    return None


def main():
    print("=" * 60)
    print("  BlazeHand Mac Test - Pinch & Drag the Dot")
    print("=" * 60)

    palm = find_model("blazepalm_xnnpack.pte", "blazepalm_traced.pt")
    hand = find_model("blazehand_xnnpack.pte", "blazehand_traced.pt")
    if not palm or not hand or not os.path.exists("anchors_palm.npy"):
        print("ERROR: Model files not found. Run export_blazehand.py first.")
        sys.exit(1)

    print(f"  Palm: {palm}")
    print(f"  Hand: {hand}")

    tracker = BlazeHandTrackerMac(palm, hand, "anchors_palm.npy")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: No webcam"); sys.exit(1)

    # --- State ---
    dot_x, dot_y = 320.0, 240.0
    dot_radius = 20
    show_debug = True

    # Pinch state with hysteresis
    is_pinching = False
    is_grabbed = False
    grab_offset_x, grab_offset_y = 0.0, 0.0
    grace_frames = 0          # Keep state during brief tracking loss
    MAX_GRACE = 4

    # Thresholds in pixels
    PINCH_GRAB_DIST = 50      # Fingers must be this close to START pinching
    PINCH_RELEASE_DIST = 70   # Fingers must be this far to STOP pinching
    GRAB_RADIUS = 50          # Must be close to dot to grab it

    # Smoothing
    smooth_cx, smooth_cy = 0.0, 0.0  # Smoothed cursor
    CURSOR_SMOOTH = 0.35              # 0=laggy, 1=instant
    DOT_SMOOTH = 0.25                 # Slow dot movement to prevent jumps

    fps_times = []

    print("\n  Pinch near the dot to grab. 'q'=quit 'r'=reset 'd'=debug\n")

    while True:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        t0 = time.time()
        result = tracker.detect(frame)
        detect_ms = (time.time() - t0) * 1000

        # FPS
        fps_times.append(time.time())
        fps_times = [t for t in fps_times if t > time.time() - 1.0]
        fps = len(fps_times)

        # --- Process detection ---
        cursor_x, cursor_y = 0, 0
        pinch_dist_px = 999

        if result is not None:
            grace_frames = 0
            lm = result["landmarks_px"]

            thumb = lm[THUMB_TIP]
            index = lm[INDEX_TIP]

            # Pixel distance between thumb tip and index tip
            pinch_dist_px = math.hypot(thumb[0] - index[0], thumb[1] - index[1])

            # Cursor = midpoint of thumb + index
            raw_cx = (thumb[0] + index[0]) / 2.0
            raw_cy = (thumb[1] + index[1]) / 2.0

            # Smooth cursor
            if smooth_cx == 0 and smooth_cy == 0:
                smooth_cx, smooth_cy = raw_cx, raw_cy
            else:
                smooth_cx = smooth_cx * (1 - CURSOR_SMOOTH) + raw_cx * CURSOR_SMOOTH
                smooth_cy = smooth_cy * (1 - CURSOR_SMOOTH) + raw_cy * CURSOR_SMOOTH

            cursor_x, cursor_y = int(smooth_cx), int(smooth_cy)

            # Hysteresis pinch detection
            if not is_pinching and pinch_dist_px < PINCH_GRAB_DIST:
                is_pinching = True
            elif is_pinching and pinch_dist_px > PINCH_RELEASE_DIST:
                is_pinching = False

            # Debug drawing
            if show_debug:
                for i, pt in enumerate(lm):
                    px, py = int(pt[0]), int(pt[1])
                    if i == THUMB_TIP or i == INDEX_TIP:
                        color = (0, 255, 0) if is_pinching else (0, 100, 255)
                        cv2.circle(frame, (px, py), 6, color, -1)
                        cv2.circle(frame, (px, py), 8, (255, 255, 255), 1)
                    else:
                        cv2.circle(frame, (px, py), 2, (60, 140, 60), -1)

                for c1, c2 in HAND_CONNECTIONS:
                    p1 = tuple(lm[c1].astype(int))
                    p2 = tuple(lm[c2].astype(int))
                    cv2.line(frame, p1, p2, (40, 100, 40), 1)

                # Thumb-index line
                t_pt = tuple(lm[THUMB_TIP].astype(int))
                i_pt = tuple(lm[INDEX_TIP].astype(int))
                cv2.line(frame, t_pt, i_pt, (0, 255, 0) if is_pinching else (0, 0, 200), 2)

                # Cursor
                cv2.drawMarker(frame, (cursor_x, cursor_y),
                               (0, 255, 0) if is_pinching else (200, 200, 200),
                               cv2.MARKER_CROSS, 15, 2)
        else:
            # No detection
            grace_frames += 1
            if grace_frames > MAX_GRACE:
                is_pinching = False
                is_grabbed = False

        # --- Grab logic ---
        if is_pinching and not is_grabbed:
            dist_to_dot = math.hypot(cursor_x - dot_x, cursor_y - dot_y)
            if dist_to_dot < GRAB_RADIUS:
                is_grabbed = True
                grab_offset_x = dot_x - smooth_cx
                grab_offset_y = dot_y - smooth_cy

        if not is_pinching:
            is_grabbed = False

        # Move dot (with speed limit to prevent jumps)
        if is_grabbed:
            target_x = smooth_cx + grab_offset_x
            target_y = smooth_cy + grab_offset_y

            # Limit max movement per frame to 15px to prevent dramatic jumps
            dx_move = (target_x - dot_x) * DOT_SMOOTH
            dy_move = (target_y - dot_y) * DOT_SMOOTH
            max_move = 15.0
            move_dist = math.hypot(dx_move, dy_move)
            if move_dist > max_move:
                scale = max_move / move_dist
                dx_move *= scale
                dy_move *= scale

            dot_x += dx_move
            dot_y += dy_move
            dot_x = max(dot_radius, min(w - dot_radius, dot_x))
            dot_y = max(dot_radius, min(h - dot_radius, dot_y))

        # --- Draw dot ---
        dx, dy = int(dot_x), int(dot_y)
        if is_grabbed:
            cv2.circle(frame, (dx, dy), dot_radius + 6, (0, 180, 255), 2)
            cv2.circle(frame, (dx, dy), dot_radius, (0, 220, 255), -1)
            cv2.circle(frame, (dx, dy), dot_radius, (255, 255, 255), 2)
            cv2.putText(frame, "GRABBED", (dx - 35, dy - dot_radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 2)
        else:
            cv2.circle(frame, (dx, dy), dot_radius, (0, 255, 100), -1)
            cv2.circle(frame, (dx, dy), dot_radius, (255, 255, 255), 2)

        # --- HUD ---
        cv2.rectangle(frame, (0, 0), (w, 30), (0, 0, 0), -1)
        hud = f"FPS:{fps} | {detect_ms:.0f}ms"
        if result is not None:
            hud += f" | Dist:{pinch_dist_px:.0f}px"
            hud += f" | {'PINCH' if is_pinching else 'open'}"
            hud += f" | {'GRAB' if is_grabbed else '-'}"
        else:
            hud += " | No hand"
        cv2.putText(frame, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.putText(frame, "'d'=debug  'r'=reset  'q'=quit",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1)

        cv2.imshow("BlazeHand ExecuTorch Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            dot_x, dot_y = w / 2.0, h / 2.0
            is_grabbed = False
        elif key == ord('d'):
            show_debug = not show_debug

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
