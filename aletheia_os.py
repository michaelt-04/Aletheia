# aletheia_os.py
# Core foundation for Project Aletheia - A Standalone AR OS for Raspberry Pi
# SFHacks 2026 - Merged and Refactored Version

import pygame
import threading
import time
import math
import os
import sys
import numpy as np

# Feature flags
# Set this to 0 to run YOLO + GUI only (no hand model required):
#   ALETHEIA_ENABLE_HANDS=0 python aletheia_os.py
ENABLE_HANDS = os.getenv("ALETHEIA_ENABLE_HANDS", "1") == "1"

# RPi-specific camera controller (now also handles webcam fallback)
from camera_rpi import get_camera_manager

# GUI Components
from aletheia_gui import SpiritCompanion, GreyFog, DetectionOverlay, HealthBar, MissionTracker, CarbonSavingsWidget

# Vision libraries
import cv2

# Hand detection (optional): ExecuTorch BlazePalm (.pte)
# Disabled by default via ALETHEIA_ENABLE_HANDS=0

# YOLO Detection Components
from yolo_engine import YOLODetector

# --- Global Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
FPS = 60

# BlazePalm / hand detector model (.pte)
BLAZEPALM_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "blazepalm_xnnpack.pte"  # <-- replace with your actual .pte filename
)

# YOLO Object Detection model path
YOLO_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "yolo26n_xnnpack.pte"
)

# --- Shared State Dictionary ---
shared_state = {
    "is_pinching": False,
    "index_finger_tip": (0, 0),
    "detections": [],
    "health": 100,
    "mission": "Explore",
    "carbon_saved": 0.0,
    "app_quit": False,
}

state_lock = threading.Lock()


# --- YOLO Detection Thread ---
class YoloDetectionThread(threading.Thread):
    def __init__(self, model_path, camera, shared_state, state_lock):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.camera = camera
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.detector = None
        self._got_first_frame = False

    def run(self):
        print("[YoloDetectionThread] Starting...")

        try:
            self.detector = YOLODetector(self.model_path)
        except Exception as e:
            print(f"[YoloDetectionThread] ERROR loading YOLO model: {e}")
            with self.state_lock:
                self.shared_state["app_quit"] = True
            return

        print("[YoloDetectionThread] Waiting for camera frames...")
        while True:
            with self.state_lock:
                if self.shared_state["app_quit"]:
                    break

            frame = self.camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            if not self._got_first_frame:
                print("[YoloDetectionThread] First frame received. Starting detection loop.")
                self._got_first_frame = True

            # frame should be RGB for our pipeline
            # YOLODetector in this repo expects RGB numpy array
            try:
                detections = self.detector.detect(frame)
            except Exception as e:
                print(f"[YoloDetectionThread] ERROR during detection: {e}")
                detections = []

            with self.state_lock:
                self.shared_state["detections"] = detections

            time.sleep(0.03)  # ~30 FPS target-ish, tune as needed

        print("[YoloDetectionThread] Stopped.")


# --- HandTrackingThread (Replaced: ExecuTorch BlazePalm detector, no MediaPipe) ---
class HandTrackingThread(threading.Thread):
    """
    Replaces MediaPipe with an ExecuTorch hand detector (.pte).

    What it provides to the rest of Aletheia:
      - shared_state["index_finger_tip"]: a cursor point (x,y) in screen coords
      - shared_state["is_pinching"]: pinch boolean (placeholder: always False unless you add a landmarks model)

    IMPORTANT:
    BlazePalm is a *palm/hand box detector*. It does not output fingertip landmarks by default.
    Until you add a landmark model, this thread sets cursor to the CENTER of the best detected hand box
    and sets pinch=False.

    If/when you export a landmarks model to .pte, extend this thread to:
      1) detect hand box (BlazePalm)
      2) crop/warp ROI
      3) run landmarks model
      4) compute pinch + fingertip
    """

    def __init__(self, model_path, camera, shared_state, state_lock,
                 input_size=256, confidence=0.6):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.camera = camera
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.input_size = input_size
        self.confidence = confidence
        self.detector = None

    def run(self):
        print("[HandTrackingThread] Starting (ExecuTorch BlazePalm)...")

        if not os.path.exists(self.model_path):
            print(f"[HandTrackingThread] ERROR: BlazePalm .pte not found at {self.model_path}")
            with self.state_lock:
                self.shared_state["app_quit"] = True
            return

        try:
            from blazepalm_engine import BlazePalmDetector  # lazy import
            self.detector = BlazePalmDetector(
                self.model_path,
                input_size=self.input_size,
                confidence_threshold=self.confidence,
            )
        except Exception as e:
            print(f"[HandTrackingThread] ERROR loading BlazePalm model: {e}")
            with self.state_lock:
                self.shared_state["app_quit"] = True
            return

        print("[HandTrackingThread] Waiting for camera frames...")
        while True:
            with self.state_lock:
                if self.shared_state["app_quit"]:
                    break

            frame_rgb = self.camera.get_frame()
            if frame_rgb is None:
                time.sleep(0.01)
                continue

            # Mirror for natural interaction (keep consistent with previous behavior)
            frame_rgb = cv2.flip(frame_rgb, 1)

            dets = self.detector.detect(frame_rgb)

            # Cursor behavior (temporary): use center of best hand box
            cursor = (0, 0)
            pinch = False

            if dets:
                best = dets[0]
                # Convert from camera frame coords -> screen coords by scaling
                h, w = frame_rgb.shape[:2]
                cx, cy = best["center"]

                # Use a slightly higher point (closer to "index area") to feel more like a pointer
                x1, y1, x2, y2 = best["box"]
                cy = int(y1 + (y2 - y1) * 0.25)

                cursor = (int(cx * SCREEN_WIDTH / w), int(cy * SCREEN_HEIGHT / h))

            with self.state_lock:
                self.shared_state["index_finger_tip"] = cursor
                self.shared_state["is_pinching"] = pinch

            time.sleep(0.01)

        print("[HandTrackingThread] Stopped.")


# --- Main Application Logic ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Aletheia OS")
    clock = pygame.time.Clock()

    # Init GUI Components
    spirit = SpiritCompanion()
    grey_fog = GreyFog()
    overlay = DetectionOverlay()
    health_bar = HealthBar()
    mission_tracker = MissionTracker()
    carbon_widget = CarbonSavingsWidget()

    print("[Main] Initializing Camera...")
    camera = get_camera_manager()
    camera.start()
    print("[Main] Camera started.")

    # Start YOLO detection thread
    yolo_thread = YoloDetectionThread(
        model_path=YOLO_MODEL_PATH,
        camera=camera,
        shared_state=shared_state,
        state_lock=state_lock
    )
    yolo_thread.start()

    # Start Hand thread (optional)
    hand_thread = None
    if ENABLE_HANDS:
        hand_thread = HandTrackingThread(
            model_path=BLAZEPALM_MODEL_PATH,
            camera=camera,
            shared_state=shared_state,
            state_lock=state_lock
        )
        hand_thread.start()
    else:
        print("[Main] Hand tracking disabled (ALETHEIA_ENABLE_HANDS=0). Running YOLO+GUI only.")

    print("[Main] Entering main loop...")

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Read shared state
        with state_lock:
            detections = shared_state.get("detections", [])
            cursor = shared_state.get("index_finger_tip", (0, 0))
            is_pinching = shared_state.get("is_pinching", False)

        # Update UI logic
        spirit.update(dt, cursor, is_pinching)
        grey_fog.update(dt)
        overlay.update(detections)
        health_bar.update(shared_state.get("health", 100))
        mission_tracker.update(shared_state.get("mission", "Explore"))
        carbon_widget.update(shared_state.get("carbon_saved", 0.0))

        # Draw everything
        screen.fill((0, 0, 0))

        # Draw camera background if your GUI expects it (optional)
        frame = camera.get_frame()
        if frame is not None:
            # frame is RGB; pygame expects (w,h) surface with 3 channels
            # Convert to surface (note: pygame uses (width,height))
            surf = pygame.surfarray.make_surface(np.rot90(frame))
            surf = pygame.transform.scale(surf, (SCREEN_WIDTH, SCREEN_HEIGHT))
            screen.blit(surf, (0, 0))

        overlay.draw(screen)
        grey_fog.draw(screen)
        spirit.draw(screen)
        health_bar.draw(screen)
        mission_tracker.draw(screen)
        carbon_widget.draw(screen)

        pygame.display.flip()

    # Shutdown
    print("[Main] Shutting down...")
    with state_lock:
        shared_state["app_quit"] = True

    camera.stop()
    pygame.quit()
    print("[Main] Goodbye.")


if __name__ == "__main__":
    main()
