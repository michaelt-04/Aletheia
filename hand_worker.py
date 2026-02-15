# hand_worker.py - One Euro Filter Smoothing
# Out-of-process hand tracking worker for Project Aletheia

import os
import time
import math
import numpy as np
import cv2
from multiprocessing.shared_memory import SharedMemory

# Landmark indices
WRIST = 0
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20


class OneEuroFilter:
    """1-Euro Filter: low-latency smoothing that adapts to signal speed.

    - When still: heavy smoothing (kills jitter)
    - When moving fast: light smoothing (kills lag)
    """
    def __init__(self, min_cutoff=1.7, beta=0.3, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x, t):
        if self.t_prev is None:
            self.x_prev = x
            self.t_prev = t
            return x

        dt = max(t - self.t_prev, 1e-6)
        self.t_prev = t

        # Smooth the derivative to avoid noise spikes
        a_d = self._alpha(self.d_cutoff, dt)
        dx = (x - self.x_prev) / dt
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        # Adaptive cutoff: faster movement → higher cutoff → less smoothing
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # Apply low-pass filter
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat

        return x_hat


def hand_worker_fn(
    palm_model_path,
    hand_model_path,
    anchors_path,
    shm_name,
    frame_shape,
    frame_dtype_str,
    frame_seq,
    result_queue,
    stop_event,
    screen_width,
    screen_height,
    target_hz=30.0,
    detect_width=320,
    detect_height=180,
):
    # Mirror flip: enable for selfie webcam testing, disable for Pi forward-facing camera
    mirror = os.getenv("ALETHEIA_MIRROR", "0") == "1"
    print(f"[HandWorker] Starting (mirror={'on' if mirror else 'off'}, "
          f"detect={detect_width}x{detect_height}, target_hz={target_hz})...")

    import torch
    torch.set_num_threads(2)
    from blazepalm_engine import BlazeHandTracker

    shm = SharedMemory(name=shm_name, create=False)
    frame_dtype = np.dtype(frame_dtype_str)
    frame_nbytes = int(np.prod(frame_shape)) * frame_dtype.itemsize

    try:
        tracker = BlazeHandTracker(palm_model_path, hand_model_path, anchors_path)
    except Exception as e:
        print(f"[HandWorker] FATAL: {e}")
        result_queue.put({"error": str(e)})
        shm.close()
        return

    period = 1.0 / max(target_hz, 0.1)
    last_seq = -1
    next_t = time.time()
    got_first = False

    # One Euro Filters for cursor (smooth when still, responsive when moving)
    filter_x = OneEuroFilter(min_cutoff=1.7, beta=0.3, d_cutoff=1.0)
    filter_y = OneEuroFilter(min_cutoff=1.7, beta=0.3, d_cutoff=1.0)
    cursor_x, cursor_y = 0.0, 0.0

    # Fist detection: ratio-based (resolution & hand-size independent)
    # Ratio = avg_fingertip_to_wrist / wrist_to_middle_mcp
    #   Open hand: ratio ~2.0-3.0 (fingertips extend well past palm base)
    #   Fist:      ratio ~0.5-1.0 (fingertips curl back toward wrist)
    is_fist = False
    FIST_CLOSE_RATIO = 1.3   # below this → fist closed
    FIST_OPEN_RATIO = 1.8    # above this → hand open

    # Debounce: require N consecutive frames to confirm state change
    fist_close_count = 0
    fist_open_count = 0
    FIST_DEBOUNCE = 3

    grace_frames = 0
    MAX_GRACE = 4

    # Palm caching: skip expensive palm detection most of the time
    cached_palm = None
    last_palm_time = 0.0
    PALM_REDETECT_SECS = 1.5  # Shorter cache to avoid stale crops during movement

    DETECT_W, DETECT_H = detect_width, detect_height

    # Timing diagnostics
    detect_count = 0

    while not stop_event.is_set():
        now = time.time()
        if now < next_t:
            time.sleep(min(0.005, next_t - now))
            continue
        next_t += period
        if now - next_t > 0.5: next_t = now + period

        current_seq = frame_seq.value
        if current_seq == last_seq:
            time.sleep(0.002)
            continue
        last_seq = current_seq

        frame = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf[:frame_nbytes]).copy()
        if frame_seq.value != last_seq: continue

        if not got_first:
            print("[HandWorker] Tracking started.")
            got_first = True

        # Downsample for faster preprocessing (model resizes to 256x256 internally)
        detect_frame = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_LINEAR)
        if mirror:
            detect_frame = cv2.flip(detect_frame, 1)

        # Decide whether to run full pipeline or landmarks-only
        use_cache = cached_palm is not None and (now - last_palm_time) < PALM_REDETECT_SECS

        t_start = time.perf_counter()
        try:
            if use_cache:
                result = tracker.detect(detect_frame, cached_palm=cached_palm)
            else:
                result = tracker.detect(detect_frame)
        except:
            result = None
        t_elapsed = time.perf_counter() - t_start

        detect_count += 1
        if detect_count <= 3 or detect_count % 30 == 0:
            mode = "landmarks" if use_cache else "full"
            print(f"[HandWorker] detect ({mode}): {t_elapsed*1000:.0f}ms")

        if result is not None:
            # Cache palm for reuse
            new_palm = result.get("_palm")
            if new_palm is not None and not use_cache:
                cached_palm = new_palm
                last_palm_time = now

            grace_frames = 0
            lm = result["landmarks_px"]

            # 1. Cursor = palm center (stable regardless of finger position)
            wrist = lm[WRIST]
            palm_x = (wrist[0] + lm[MIDDLE_MCP][0]) / 2.0
            palm_y = (wrist[1] + lm[MIDDLE_MCP][1]) / 2.0

            # Hand base distance (wrist → middle MCP) — stable reference
            hand_base = math.hypot(
                lm[MIDDLE_MCP][0] - wrist[0],
                lm[MIDDLE_MCP][1] - wrist[1],
            )

            # Sanity check: if hand_base is tiny, landmarks are garbage
            if hand_base < 5.0:
                cached_palm = None
                continue

            # 2. Fist detection: ratio of fingertip distance to hand base
            avg_tip_dist = sum(
                math.hypot(lm[tip][0] - wrist[0], lm[tip][1] - wrist[1])
                for tip in (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
            ) / 4.0

            fist_ratio = avg_tip_dist / hand_base

            # Log ratio periodically for diagnostics
            if detect_count % 30 == 0:
                print(f"[HandWorker] fist_ratio={fist_ratio:.2f} "
                      f"(tip={avg_tip_dist:.0f}px, base={hand_base:.0f}px, fist={is_fist})")

            # Debounce: require consecutive frames to confirm state change
            if not is_fist:
                if fist_ratio < FIST_CLOSE_RATIO:
                    fist_close_count += 1
                    if fist_close_count >= FIST_DEBOUNCE:
                        is_fist = True
                        fist_close_count = 0
                        print(f"[HandWorker] FIST closed (ratio={fist_ratio:.2f})")
                else:
                    fist_close_count = 0
            else:
                if fist_ratio > FIST_OPEN_RATIO:
                    fist_open_count += 1
                    if fist_open_count >= FIST_DEBOUNCE:
                        is_fist = False
                        fist_open_count = 0
                        print(f"[HandWorker] FIST opened (ratio={fist_ratio:.2f})")
                else:
                    fist_open_count = 0

            # 3. Map palm center to screen coordinates
            raw_x = float(palm_x) * screen_width / max(DETECT_W, 1)
            raw_y = float(palm_y) * screen_height / max(DETECT_H, 1)

            # 4. One Euro Filter — smooth when still, responsive when moving
            cursor_x = filter_x(raw_x, now)
            cursor_y = filter_y(raw_y, now)

            cursor = (int(cursor_x), int(cursor_y))
        else:
            grace_frames += 1
            if grace_frames > MAX_GRACE:
                is_fist = False
            # Invalidate palm cache if hand lost too long
            if grace_frames > MAX_GRACE * 2:
                cached_palm = None
            cursor = (int(cursor_x), int(cursor_y))

        # Send (keep key names for GUI compatibility)
        try:
            if result_queue.full(): result_queue.get_nowait()
            result_queue.put_nowait({
                "index_finger_tip": cursor,
                "is_pinching": is_fist,
            })
        except: pass

    shm.close()
    print("[HandWorker] Stopped.")