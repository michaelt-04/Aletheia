# yolo_worker.py
# Out-of-process YOLO inference worker for Project Aletheia
#
# Runs YOLODetector in a separate process to avoid GIL contention
# with the Pygame GUI thread.
#
# Communication:
#   Frames IN:   multiprocessing SharedMemory (written by main process)
#   Results OUT:  multiprocessing Queue
#   Sync:         multiprocessing Value (frame sequence counter)
#   Shutdown:     multiprocessing Event

import time
import numpy as np
from multiprocessing.shared_memory import SharedMemory


def yolo_worker_fn(
    model_path,
    shm_name,
    frame_shape,
    frame_dtype_str,
    frame_seq,
    result_queue,
    stop_event,
    input_size=640,
    confidence_threshold=0.25,
    target_hz=10.0,
):
    """
    YOLO inference worker — runs in a separate process with its own GIL.

    1. Attaches to SharedMemory created by the main process
    2. Loads YOLODetector (imports torch/executorch only in this process)
    3. Reads frames from shared memory, runs inference, sends results via Queue
    4. Exits on stop_event
    """
    print("[YoloWorker] Starting in separate process...")

    # Heavy imports happen HERE — the main process never loads these
    import torch
    torch.set_num_threads(2)  # Limit XNNPACK threadpool to avoid contention on Pi 4
    from meta_yolo.yolo_engine import YOLODetector

    # Attach to shared memory (main process owns create/unlink lifecycle)
    shm = SharedMemory(name=shm_name, create=False)
    frame_dtype = np.dtype(frame_dtype_str)
    frame_nbytes = int(np.prod(frame_shape)) * frame_dtype.itemsize

    try:
        detector = YOLODetector(
            model_path,
            input_size=input_size,
            confidence_threshold=confidence_threshold,
        )
    except Exception as e:
        print(f"[YoloWorker] FATAL: Could not load model: {e}")
        result_queue.put({"error": str(e)})
        shm.close()
        return

    print(f"[YoloWorker] Model loaded (input_size={input_size}). Entering detection loop.")

    period = 1.0 / max(target_hz, 0.1)
    last_seq = -1
    next_t = time.time()
    got_first = False

    while not stop_event.is_set():
        now = time.time()
        if now < next_t:
            time.sleep(min(0.005, next_t - now))
            continue

        next_t += period
        if now - next_t > 0.5:
            next_t = now + period

        # Check for new frame
        current_seq = frame_seq.value
        if current_seq == last_seq:
            time.sleep(0.002)
            continue
        last_seq = current_seq

        # Copy frame from shared memory
        frame = np.ndarray(frame_shape, dtype=frame_dtype, buffer=shm.buf[:frame_nbytes]).copy()

        # Seqlock: discard if frame was overwritten during our copy
        if frame_seq.value != last_seq:
            continue

        if not got_first:
            print("[YoloWorker] First frame received. Starting detection loop.")
            got_first = True

        try:
            detections = detector.detect(frame)
        except Exception as e:
            print(f"[YoloWorker] Detection error: {e}")
            detections = []

        # Non-blocking put — drop oldest if queue is full
        if result_queue.full():
            try:
                result_queue.get_nowait()
            except Exception:
                pass
        try:
            result_queue.put_nowait(detections)
        except Exception:
            pass

    print("[YoloWorker] Stop event received. Cleaning up.")
    shm.close()
    print("[YoloWorker] Exited cleanly.")
