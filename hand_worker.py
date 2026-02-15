# hand_worker.py
# Out-of-process hand tracking worker for Project Aletheia
#
# Runs BlazeHandTracker (BlazePalm + BlazeHand) in a separate process
# to avoid GIL contention with the Pygame GUI thread.
#
# Communication:
#   Frames IN:   multiprocessing SharedMemory (same buffer as YOLO worker)
#   Results OUT:  multiprocessing Queue
#   Sync:         multiprocessing Value (frame sequence counter)
#   Shutdown:     multiprocessing Event

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
    target_hz=15.0,
    smoothing=0.1, # CHANGED: Lower smoothing (0.15 -> 0.1) for faster cursor
):
    """
    Hand tracking worker — runs in a separate process with its own GIL.

    Shares the same SharedMemory frame buffer as the YOLO worker.
    Produces cursor position + pinch state for the GUI.
    """
    print("[HandWorker] Starting in separate process...")

    # Heavy imports inside worker process only
    import torch
    torch.set_num_threads(2)  # Limit XNNPACK threadpool to avoid contention on Pi 4
    from blazepalm_engine import BlazeHandTracker

    # Attach to shared memory (created by main process, shared with YOLO worker)
    shm = SharedMemory(name=shm_name, create=False)
    frame_dtype = np.dtype(frame_dtype_str)
    frame_nbytes = int(np.prod(frame_shape)) * frame_dtype.itemsize

    try:
        tracker = BlazeHandTracker(palm_model_path, hand_model_path, anchors_path)
    except Exception as e:
        print(f"[HandWorker] FATAL: Could not load models: {e}")
        result_queue.put({"error": str(e)})
        shm.close()
        return

    print(f"[HandWorker] Models loaded. Entering tracking loop "
          f"(target_hz={target_hz}, screen={screen_width}x{screen_height})")

    period = 1.0 / max(target_hz, 0.1)
    last_seq = -1
    next_t = time.time()
    got_first = False

    # Cursor smoothing state
    smooth_x, smooth_y = 0.0, 0.0
    alpha = 1.0 - smoothing  # higher alpha = more responsive

    # Pinch state with hysteresis
    is_pinching = False
    
    # CHANGED: Increased Grab distance (50 -> 60) to make pinching easier
    PINCH_GRAB_DIST = 60     
    PINCH_RELEASE_DIST = 75  # pixels — far enough to release pinch
    
    grace_frames = 0
    MAX_GRACE = 4

    while not stop_event.is_set():
        now = time.time()
        if now < next_t:
            time.sleep(min(0.005, next_t - now))
            continue

        next_t += period
        if now - next_t > 0.5:
            next_t = now + period

        # Check for new frame (seqlock pattern — same as YOLO worker)
        current_seq = frame_seq.value
        if current_seq == last_seq:
            time.sleep(0.002)
            continue
        last_seq = current_seq

        # Copy frame from shared memory
        frame = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf[:frame_nbytes]).copy()

        # Torn read detection
        if frame_seq.value != last_seq:
            continue

        if not got_first:
            print("[HandWorker] First frame received. Starting tracking loop.")
            got_first = True

        # Mirror flip for natural interaction
        frame = cv2.flip(frame, 1)

        # Run full hand tracking pipeline
        try:
            result = tracker.detect(frame)
        except Exception as e:
            print(f"[HandWorker] Detection error: {e}")
            result = None

        if result is not None:
            if grace_frames >= MAX_GRACE:
                # Optional: Log occasionally to verify tracking health
                if now % 5.0 < 0.1:
                    print(f"[HandWorker] Hand detected (conf={result['confidence']:.2f}, palm={result['palm_score']:.2f})")
            grace_frames = 0
            lm = result["landmarks_px"]

            # Extract fingertip positions
            thumb = lm[THUMB_TIP]
            index = lm[INDEX_TIP]

            # Pinch detection: thumb-to-index distance with hysteresis
            pinch_dist = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
            if not is_pinching and pinch_dist < PINCH_GRAB_DIST:
                is_pinching = True
                print(f"[HandWorker] PINCH detected (dist={pinch_dist:.0f}px)")
            elif is_pinching and pinch_dist > PINCH_RELEASE_DIST:
                is_pinching = False
                print(f"[HandWorker] PINCH released (dist={pinch_dist:.0f}px)")

            # Cursor = index finger tip, mapped to screen coordinates
            h, w = frame.shape[:2]
            raw_x = float(index[0]) * screen_width / max(w, 1)
            raw_y = float(index[1]) * screen_height / max(h, 1)

            # Exponential smoothing
            if smooth_x == 0.0 and smooth_y == 0.0:
                smooth_x, smooth_y = raw_x, raw_y
            else:
                smooth_x = smooth_x * (1.0 - alpha) + raw_x * alpha
                smooth_y = smooth_y * (1.0 - alpha) + raw_y * alpha

            cursor = (int(smooth_x), int(smooth_y))
        else:
            # No hand detected
            grace_frames += 1
            if grace_frames > MAX_GRACE:
                is_pinching = False
            cursor = (int(smooth_x), int(smooth_y))

        # Send result to main process
        hand_result = {
            "index_finger_tip": cursor,
            "is_pinching": is_pinching,
        }

        if result_queue.full():
            try:
                result_queue.get_nowait()
            except Exception:
                pass
        try:
            result_queue.put_nowait(hand_result)
        except Exception:
            pass

    print("[HandWorker] Stop event received. Cleaning up.")
    shm.close()
    print("[HandWorker] Exited cleanly.")