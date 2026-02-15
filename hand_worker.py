# hand_worker.py - Adaptive Smoothing
# Out-of-process hand tracking worker for Project Aletheia

import os
import time
import math
import numpy as np
import cv2
from multiprocessing.shared_memory import SharedMemory

# Landmark indices
THUMB_TIP = 4
INDEX_TIP = 8

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
    smoothing=0.0, # Ignored, we use adaptive now
):
    # Mirror flip: enable for selfie webcam testing, disable for Pi forward-facing camera
    mirror = os.getenv("ALETHEIA_MIRROR", "0") == "1"
    print(f"[HandWorker] Starting (mirror={'on' if mirror else 'off'})...")

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

    # Smoothing State
    smooth_x, smooth_y = 0.0, 0.0

    # Pinch Configuration
    is_pinching = False
    PINCH_GRAB_DIST = 65     # px
    PINCH_RELEASE_DIST = 80  # px

    grace_frames = 0
    MAX_GRACE = 4

    # Palm caching: skip expensive palm detection most of the time
    cached_palm = None
    last_palm_time = 0.0
    PALM_REDETECT_SECS = 3.0  # Full palm re-detection interval

    # Downsampled detection size (model uses 256x256 internally anyway)
    DETECT_W, DETECT_H = 320, 180

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
        detect_frame = cv2.resize(frame, (DETECT_W, DETECT_H), interpolation=cv2.INTER_NEAREST)
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
        if detect_count <= 3 or detect_count % 20 == 0:
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
            thumb = lm[THUMB_TIP]
            index = lm[INDEX_TIP]

            # 1. Detect Pinch
            dist_pinch = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
            if not is_pinching and dist_pinch < PINCH_GRAB_DIST:
                is_pinching = True
            elif is_pinching and dist_pinch > PINCH_RELEASE_DIST:
                is_pinching = False

            # 2. Map to screen coordinates (detect_frame dimensions)
            raw_x = float(index[0]) * screen_width / max(DETECT_W, 1)
            raw_y = float(index[1]) * screen_height / max(DETECT_H, 1)

            # 3. Adaptive Smoothing
            move_dist = math.hypot(raw_x - smooth_x, raw_y - smooth_y)
            alpha = 0.15 + (0.7 * min(move_dist / 150.0, 1.0))

            if smooth_x == 0.0:
                smooth_x, smooth_y = raw_x, raw_y
            else:
                smooth_x = smooth_x * (1.0 - alpha) + raw_x * alpha
                smooth_y = smooth_y * (1.0 - alpha) + raw_y * alpha

            cursor = (int(smooth_x), int(smooth_y))
        else:
            grace_frames += 1
            if grace_frames > MAX_GRACE:
                is_pinching = False
            # Invalidate palm cache if hand lost too long
            if grace_frames > MAX_GRACE * 2:
                cached_palm = None
            cursor = (int(smooth_x), int(smooth_y))

        # Send
        try:
            if result_queue.full(): result_queue.get_nowait()
            result_queue.put_nowait({
                "index_finger_tip": cursor,
                "is_pinching": is_pinching,
            })
        except: pass

    shm.close()
    print("[HandWorker] Stopped.")