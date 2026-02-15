# hand_worker.py - Adaptive Smoothing & Pinch Lock
# Out-of-process hand tracking worker for Project Aletheia

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
    print("[HandWorker] Starting with Adaptive Smoothing...")

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

        frame = cv2.flip(frame, 1)

        try:
            result = tracker.detect(frame)
        except:
            result = None

        if result is not None:
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

            # 2. Map Raw Coordinates
            h, w = frame.shape[:2]
            raw_x = float(index[0]) * screen_width / max(w, 1)
            raw_y = float(index[1]) * screen_height / max(h, 1)

            # 3. Adaptive Smoothing (The Fix)
            # Calculate how far the cursor wants to move this frame
            move_dist = math.hypot(raw_x - smooth_x, raw_y - smooth_y)
            
            # Dynamic Alpha:
            # - Fast movement (>150px) -> Alpha 0.8 (Fast/Responsive)
            # - Slow movement (<10px)  -> Alpha 0.05 (Very Smooth/Stable)
            alpha = 0.05 + (0.75 * min(move_dist / 150.0, 1.0))
            
            # PINCH LOCK: When pinching, force high stability to prevent "jump on click"
            if is_pinching:
                alpha *= 0.3  # Reduces sensitivity by 70% while holding
            
            if smooth_x == 0.0:
                smooth_x, smooth_y = raw_x, raw_y
            else:
                smooth_x = smooth_x * (1.0 - alpha) + raw_x * alpha
                smooth_y = smooth_y * (1.0 - alpha) + raw_y * alpha

            cursor = (int(smooth_x), int(smooth_y))
        else:
            grace_frames += 1
            if grace_frames > MAX_GRACE: is_pinching = False
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