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
from meta_yolo.yolo_engine import YOLODetector


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
    def __init__(self, model_path, camera, shared_state, state_lock,
                 target_hz=10):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.camera = camera
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.detector = None
        self._got_first_frame = False
        self.target_hz = float(target_hz)

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

        period = 1.0 / max(self.target_hz, 0.1)
        next_t = time.time()

        while True:
            with self.state_lock:
                if self.shared_state["app_quit"]:
                    break

            # Rate-limit to target_hz (time-based, smooth cadence)
            now = time.time()
            if now < next_t:
                time.sleep(min(0.002, next_t - now))
                continue

            # Schedule the next tick; if we fell behind, don't drift forever
            next_t += period
            if now - next_t > 0.5:
                next_t = now + period

            frame = self.camera.get_frame()
            if frame is None:
                continue

            if not self._got_first_frame:
                print("[YoloDetectionThread] First frame received. Starting detection loop.")
                self._got_first_frame = True

            try:
                detections = self.detector.detect(frame)
            except Exception as e:
                print(f"[YoloDetectionThread] ERROR during detection: {e}")
                detections = []

            with self.state_lock:
                self.shared_state["detections"] = detections

        print("[YoloDetectionThread] Stopped.")


# --- HandTrackingThread (ExecuTorch BlazePalm detector, no MediaPipe) ---
class HandTrackingThread(threading.Thread):
    """
    Uses ExecuTorch hand detector (.pte). Provides:
      - shared_state["index_finger_tip"]: cursor point (x,y) in screen coords
      - shared_state["is_pinching"]: pinch boolean (placeholder False)

    Smoothness:
      - Runs at target_hz (default 18Hz)
      - Applies light exponential smoothing to cursor to reduce jitter
        without noticeable added latency.
    """
    def __init__(self, model_path, camera, shared_state, state_lock,
                 input_size=256, confidence=0.6,
                 target_hz=18,
                 smoothing=0.35):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.camera = camera
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.input_size = input_size
        self.confidence = confidence
        self.target_hz = float(target_hz)

        # smoothing in [0..1]:
        #   0.0 = no smoothing (most responsive, most jitter)
        #   0.25-0.45 = good on Pi
        #   1.0 = extremely smooth but laggy (avoid)
        self.smoothing = float(smoothing)

        self.detector = None
        self._cursor_f = None  # float cursor for smoothing

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

        period = 1.0 / max(self.target_hz, 0.1)
        next_t = time.time()

        while True:
            with self.state_lock:
                if self.shared_state["app_quit"]:
                    break

            # Rate-limit to target_hz
            now = time.time()
            if now < next_t:
                time.sleep(min(0.002, next_t - now))
                continue

            next_t += period
            if now - next_t > 0.5:
                next_t = now + period

            frame_rgb = self.camera.get_frame()
            if frame_rgb is None:
                continue

            # Mirror for natural interaction
            frame_rgb = cv2.flip(frame_rgb, 1)

            try:
                dets = self.detector.detect(frame_rgb)
            except Exception as e:
                print(f"[HandTrackingThread] ERROR during detect: {e}")
                dets = []

            # Cursor behavior: best hand box -> use "upper" point in box
            cursor = None
            pinch = False

            if dets:
                best = dets[0]
                h, w = frame_rgb.shape[:2]
                cx, cy = best["center"]

                x1, y1, x2, y2 = best["box"]
                cy = int(y1 + (y2 - y1) * 0.25)

                sx = float(cx) * SCREEN_WIDTH / max(w, 1)
                sy = float(cy) * SCREEN_HEIGHT / max(h, 1)
                cursor = (sx, sy)

            # If no detection this frame, keep last cursor (prevents snapping to 0,0)
            if cursor is None:
                if self._cursor_f is not None:
                    cursor = self._cursor_f
                else:
                    cursor = (0.0, 0.0)

            # Exponential smoothing (low latency)
            if self._cursor_f is None:
                self._cursor_f = cursor
            else:
                a = self.smoothing
                self._cursor_f = (
                    self._cursor_f[0] * a + cursor[0] * (1.0 - a),
                    self._cursor_f[1] * a + cursor[1] * (1.0 - a),
                )

            cursor_int = (int(self._cursor_f[0]), int(self._cursor_f[1]))

            with self.state_lock:
                self.shared_state["index_finger_tip"] = cursor_int
                self.shared_state["is_pinching"] = pinch

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
