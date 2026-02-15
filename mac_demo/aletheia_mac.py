# aletheia_mac.py
# Mac Demo for Project Aletheia - SFHacks 2026
#
# Windowed version that uses the Mac webcam, same models, same GUI.
# Usage: python mac_demo/aletheia_mac.py

import sys
import os

# Add repo root to path so shared modules (blazepalm_engine, hand_worker, etc.) resolve
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import pygame
import threading
import time
import numpy as np
import cv2
import multiprocessing
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Process, Queue as MPQueue, Event as MPEvent, Value

from camera_rpi import WebcamCamera
from aletheia_gui import (
    SpiritCompanion, DetectionOverlay, HealthBar,
    MissionTracker, CarbonSavingsWidget, QuestManager,
)
from yolo_worker import yolo_worker_fn
from hand_worker import hand_worker_fn


# --- Model path resolution (relative to repo root) ---

def resolve_model_path(filename):
    candidates = [
        os.path.join(REPO_ROOT, filename),
        os.path.join(REPO_ROOT, "models", filename),
        os.path.join(REPO_ROOT, "meta_yolo", filename),
        os.path.join(REPO_ROOT, "blazepalm_executorch", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            print(f"[ModelPath] Using {p}")
            return p
    print(f"[ModelPath] WARNING: {filename} not found. Tried:")
    for p in candidates:
        print(f"  - {p}")
    return candidates[0]


# --- Constants ---

SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
FPS = 30

FRAME_SHAPE = (720, 1280, 3)
FRAME_DTYPE = np.uint8
FRAME_NBYTES = int(np.prod(FRAME_SHAPE)) * np.dtype(FRAME_DTYPE).itemsize

YOLO_MODEL_PATH = resolve_model_path("yolo26n_xnnpack.pte")
YOLO_INPUT_SIZE = 640
YOLO_TARGET_HZ = 10.0

BLAZEPALM_MODEL_PATH = resolve_model_path("blazepalm_xnnpack.pte")
BLAZEHAND_MODEL_PATH = resolve_model_path("blazehand_xnnpack.pte")
BLAZEPALM_ANCHORS_PATH = resolve_model_path("anchors_palm.npy")
HAND_TARGET_HZ = 30.0
# Higher detect resolution on Mac — reduces landmark jitter
# (320x180 on Pi amplifies 1px error to 4px on screen; 640x360 halves that)
HAND_DETECT_W, HAND_DETECT_H = 640, 360

# --- Shared State ---

shared_state = {
    "is_pinching": False,
    "index_finger_tip": (0, 0),
    "detected_objects": [],
    "energy_waste_count": 0,
    "last_savings_event": "",
    "last_savings_event_time": 0.0,
    "health": 100,
    "missions_completed": 0,
    "missions_total": 5,
    "carbon_saved_g": 0.0,
    "app_quit": False,
}

state_lock = threading.Lock()


def main():
    # Ensure spawned worker processes can import modules from repo root
    os.environ["PYTHONPATH"] = REPO_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
    multiprocessing.set_start_method("spawn", force=True)

    pygame.init()
    print(f"[Mac Demo] SDL video driver: {pygame.display.get_driver()}")

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Aletheia Mac Demo")
    clock = pygame.time.Clock()

    # GUI components
    spirit = SpiritCompanion(shared_state, state_lock)
    spirit_group = pygame.sprite.Group(spirit)
    overlay = DetectionOverlay(shared_state, state_lock)
    health_bar = HealthBar(shared_state, state_lock)
    mission_tracker = MissionTracker(shared_state, state_lock)
    carbon_widget = CarbonSavingsWidget(shared_state, state_lock)
    quest_manager = QuestManager(shared_state, state_lock)

    # Camera
    print("[Mac Demo] Starting webcam...")
    camera = WebcamCamera(width=1280, height=720)
    camera.start()
    time.sleep(0.5)  # let camera warm up
    print("[Mac Demo] Camera started.")

    # --- Shared memory for worker processes ---
    try:
        _stale = SharedMemory(name="aletheia_frame", create=False)
        _stale.close()
        _stale.unlink()
        print("[Mac Demo] Cleaned up stale shared memory.")
    except FileNotFoundError:
        pass

    shm = SharedMemory(name="aletheia_frame", create=True, size=FRAME_NBYTES)
    frame_seq = Value('q', 0)

    import atexit
    def _cleanup_shm():
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass
    atexit.register(_cleanup_shm)

    _shm_array = np.ndarray(FRAME_SHAPE, dtype=FRAME_DTYPE, buffer=shm.buf[:FRAME_NBYTES])

    # --- YOLO worker ---
    yolo_result_queue = MPQueue(maxsize=2)
    yolo_stop_event = MPEvent()
    yolo_proc = Process(
        target=yolo_worker_fn,
        args=(
            YOLO_MODEL_PATH, shm.name, FRAME_SHAPE,
            np.dtype(FRAME_DTYPE).name, frame_seq,
            yolo_result_queue, yolo_stop_event,
        ),
        kwargs={"input_size": YOLO_INPUT_SIZE, "target_hz": YOLO_TARGET_HZ},
        daemon=True,
    )
    yolo_proc.start()
    print(f"[Mac Demo] YOLO worker started (PID={yolo_proc.pid})")

    # --- Hand tracking worker ---
    hand_result_queue = MPQueue(maxsize=2)
    hand_stop_event = MPEvent()
    hand_proc = Process(
        target=hand_worker_fn,
        args=(
            BLAZEPALM_MODEL_PATH, BLAZEHAND_MODEL_PATH, BLAZEPALM_ANCHORS_PATH,
            shm.name, FRAME_SHAPE, np.dtype(FRAME_DTYPE).name,
            frame_seq, hand_result_queue, hand_stop_event,
            SCREEN_WIDTH, SCREEN_HEIGHT,
        ),
        kwargs={"target_hz": HAND_TARGET_HZ,
                "detect_width": HAND_DETECT_W, "detect_height": HAND_DETECT_H},
        daemon=True,
    )
    hand_proc.start()
    print(f"[Mac Demo] Hand worker started (PID={hand_proc.pid})")

    # --- Main loop ---
    print("[Mac Demo] Entering main loop. Press ESC or close window to quit.")

    running = True
    cam_surface = None
    cam_update_every = 2
    cam_counter = 0

    _feed_every = max(1, int(FPS / max(YOLO_TARGET_HZ, HAND_TARGET_HZ)))
    _feed_counter = 0

    _next_fps_print = time.time() + 3.0
    hand_has_data = False

    while running:
        dt_ms = clock.tick(FPS)
        dt = min(dt_ms / 1000.0, 0.1)

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Mouse fallback: always available, hand tracking overrides when active
        if not hand_has_data:
            mx, my = pygame.mouse.get_pos()
            with state_lock:
                shared_state["index_finger_tip"] = (mx, my)
                shared_state["is_pinching"] = pygame.mouse.get_pressed()[0]

        # Feed mirrored camera frame to shared memory
        _feed_counter += 1
        if _feed_counter >= _feed_every:
            _feed_counter = 0
            feed_frame = camera.get_frame()
            if feed_frame is not None and feed_frame.shape == tuple(FRAME_SHAPE):
                # Mirror for selfie webcam — both workers see same mirrored view
                feed_frame = cv2.flip(feed_frame, 1)
                np.copyto(_shm_array, feed_frame)
                frame_seq.value += 1

        # Drain YOLO results
        try:
            while True:
                result = yolo_result_queue.get_nowait()
                if isinstance(result, dict) and "error" in result:
                    print(f"[Mac Demo] YOLO error: {result['error']}")
                    running = False
                    break
                with state_lock:
                    shared_state["detected_objects"] = result
        except Exception:
            pass

        # Drain hand tracking results
        try:
            while True:
                hand_result = hand_result_queue.get_nowait()
                if isinstance(hand_result, dict) and "error" in hand_result:
                    print(f"[Mac Demo] Hand error: {hand_result['error']}")
                    break
                hand_has_data = True
                with state_lock:
                    shared_state["index_finger_tip"] = hand_result["index_finger_tip"]
                    shared_state["is_pinching"] = hand_result["is_pinching"]
        except Exception:
            pass

        # State snapshot
        with state_lock:
            state_snapshot = dict(shared_state)

        # Update spirit
        spirit_group.update(state_snapshot, dt)

        # Draw
        screen.fill((0, 0, 0))

        # Camera background (always on for demo)
        # np.rot90 inherently flips horizontally (transpose + flip), giving us
        # the selfie-mirror effect that matches the mirrored frame workers see.
        # Do NOT add cv2.flip here — it would cancel the rot90 mirror.
        cam_counter += 1
        if cam_counter % cam_update_every == 0:
            frame = camera.get_frame()
            if frame is not None:
                resized = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT),
                                     interpolation=cv2.INTER_NEAREST)
                cam_surface = pygame.surfarray.make_surface(np.rot90(resized))

        if cam_surface is not None:
            screen.blit(cam_surface, (0, 0))

        # Widgets
        overlay.draw(screen, state_snapshot)
        spirit_group.draw(screen)
        health_bar.draw(screen, state_snapshot)
        mission_tracker.draw(screen, state_snapshot)
        carbon_widget.draw(screen, state_snapshot)
        quest_manager.draw(screen, state_snapshot, dt)

        pygame.display.flip()

        # FPS logging
        now = time.time()
        if now >= _next_fps_print:
            _next_fps_print = now + 3.0
            print(f"[Mac Demo] FPS: {clock.get_fps():.1f}")

    # --- Shutdown ---
    print("[Mac Demo] Shutting down...")
    with state_lock:
        shared_state["app_quit"] = True

    yolo_stop_event.set()
    hand_stop_event.set()

    yolo_proc.join(timeout=3)
    if yolo_proc.is_alive():
        yolo_proc.terminate()

    hand_proc.join(timeout=3)
    if hand_proc.is_alive():
        hand_proc.terminate()

    try:
        shm.close()
        shm.unlink()
    except Exception:
        pass

    camera.stop()
    pygame.quit()
    print("[Mac Demo] Goodbye.")


if __name__ == "__main__":
    main()
