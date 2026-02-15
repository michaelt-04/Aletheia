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
os.environ['GLOG_minloglevel'] = '2'
# RPi-specific camera controller (now also handles webcam fallback)
from camera_rpi import get_camera_manager

# GUI Components
from aletheia_gui import SpiritCompanion, GreyFog, DetectionOverlay, HealthBar, MissionTracker, CarbonSavingsWidget

# Vision libraries
import cv2
import mediapipe as mp

# Add meta-yolo to path for YOLO imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta-yolo"))
from yolo_engine import YOLODetector
from executorch.runtime import Runtime # Needed by YOLODetector


# --- Global Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
BLACK = (0, 0, 0)
VERSION = "Aletheia OS v0.4.0 RPi"

# Path to YOLO model
YOLO_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "meta-yolo", "yolo26n_xnnpack.pte"
)

# MediaPipe Hand Landmarker model
HAND_LANDMARKER_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task"
)

# --- Shared State & Thread Safety ---
shared_state = {
    "carbon_velocity": 0.0,
    "index_finger_tip": (0, 0),
    "is_pinching": False,
    "detected_objects": [], # List of dicts from YOLO
    "active_quests": {},    # Dict of {object_label: time_activated} for detected objects that are 'active quests'
    "carbon_saved_g": 0.0,  # Total carbon saved in grams
    "last_savings_event_time": 0.0, # Timestamp of last carbon saving event
    "app_quit": False,
    "cpu_temp": 0.0,
    "yolo_inference_ms": 0.0,
    "detection_count": 0,
    "health": 100,
    "missions_completed": 0,
    "missions_total": 5
    }
state_lock = threading.Lock()


# --- Thermal Management (Copied from yolo_live.py) ---
def get_cpu_temp():
    """
    Read Raspberry Pi CPU temperature.
    Returns temperature in Celsius, or -1 if not available.
    """
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp = int(f.read().strip()) / 1000.0
            return temp
    except (FileNotFoundError, ValueError):
        return -1.0


def get_throttle_delay(temp, base_interval):
    """
    Calculate extra delay based on CPU temperature.

    Thermal zones:
        < 70°C  : No throttle (run at base_interval)
        70-75°C : Slight slowdown (+1s)
        75-80°C : Moderate slowdown (+3s)
        80-85°C : Heavy throttle (+6s)
        > 85°C  : Emergency throttle (+10s)

    Returns:
        Total interval (base + throttle) in seconds
    """
    if temp < 0:
        return base_interval  # Temp not available, no throttle

    if temp < 70:
        return base_interval
    elif temp < 75:
        return base_interval + 1.0
    elif temp < 80:
        return base_interval + 3.0
    elif temp < 85:
        return base_interval + 6.0
    else:
        return base_interval + 10.0


# --- YOLODetectionThread (Refactored from yolo_live.py) ---

class YoloDetectionThread(threading.Thread):
    """
    Background thread that receives camera frames, runs YOLO detection,
    and updates a shared state dictionary.
    """

    def __init__(self, model_path, camera, shared_state, state_lock,
                 base_interval=2.5, confidence=0.25):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.camera = camera
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.base_interval = base_interval
        self.confidence = confidence
        self._detector = None

    def run(self):
        print(f"[YoloDetectionThread] Starting...")
        # Load model
        try:
            self._detector = YOLODetector(
                self.model_path,
                confidence_threshold=self.confidence
            )
        except Exception as e:
            print(f"[YoloDetectionThread] ERROR loading model: {e}")
            with self.state_lock:
                self.shared_state["app_quit"] = True # Signal main app to quit on YOLO failure
            return

        print(f"[YoloDetectionThread] Waiting for camera frames...")
        # Wait until the camera provides the first frame
        while True:
            if self.camera.get_frame() is not None:
                print(f"[YoloDetectionThread] First frame received. Starting detection loop.")
                break
            with self.state_lock:
                if self.shared_state["app_quit"]:
                    return
            time.sleep(0.5)

        frame_count = 0
        while True:
            with self.state_lock:
                if self.shared_state["app_quit"]:
                    break

            frame_rgb = self.camera.get_frame()
            if frame_rgb is None:
                time.sleep(0.01) # Wait for frames
                continue
            
            # The YOLODetector expects BGR image, convert from RGB
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            t0 = time.time()
            detections = self._detector.detect(frame_bgr)
            inference_time = time.time() - t0

            carbon_v = YOLODetector.compute_carbon_velocity(detections)
            cpu_temp = get_cpu_temp()

            with self.state_lock:
                self.shared_state["detected_objects"] = detections
                self.shared_state["carbon_velocity"] = carbon_v
                self.shared_state["cpu_temp"] = cpu_temp
                self.shared_state["yolo_inference_ms"] = inference_time * 1000
                self.shared_state["detection_count"] = len(detections)

            frame_count += 1
            # print(f"[YoloDetectionThread] Detected {len(detections)} objects, Carbon V: {carbon_v:.2f}")

            interval = get_throttle_delay(cpu_temp, self.base_interval)
            sleep_time = max(0, interval - inference_time)
            time.sleep(sleep_time)

        print("[YoloDetectionThread] Stopped.")


# --- HandTrackingThread (New implementation using MediaPipe Tasks API) ---

class HandTrackingThread(threading.Thread):
    """
    Handles hand tracking using MediaPipe's HandLandmarker (async LIVE_STREAM mode)
    on frames from the shared camera.
    """
    def __init__(self, model_path, camera, shared_state, state_lock):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.camera = camera
        self.shared_state = shared_state
        self.state_lock = state_lock
        
        self.landmarker = None
        self.mp_hand_landmarks = None # Store results from callback
        self.mp_image_timestamp = 0
        
        # MediaPipe Tasks API aliases
        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        self.VisionRunningMode = mp.tasks.vision.RunningMode

        # Hand landmark indices
        self.THUMB_TIP = 4
        self.INDEX_FINGER_TIP = 8
        self.PINCH_THRESHOLD = 0.05 # Distance threshold for pinch detection

    def _on_hand_landmarker_result(self, result, output_image, timestamp_ms):
        """Callback function for MediaPipe's LIVE_STREAM mode."""
        self.mp_hand_landmarks = result.hand_landmarks
        self.mp_image_timestamp = timestamp_ms

    def run(self):
        print("[HandTrackingThread] Starting...")
        # Load hand landmarker model
        if not os.path.exists(self.model_path):
            print(f"[HandTrackingThread] ERROR: Hand landmarker model not found at {self.model_path}")
            with self.state_lock:
                self.shared_state["app_quit"] = True
            return

        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(model_asset_path=self.model_path),
            running_mode=self.VisionRunningMode.LIVE_STREAM,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            result_callback=self._on_hand_landmarker_result,
        )
        try:
            self.landmarker = self.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[HandTrackingThread] ERROR: Could not create HandLandmarker: {e}")
            with self.state_lock:
                self.shared_state["app_quit"] = True
            return

        print("[HandTrackingThread] Waiting for camera frames...")
        frame_timestamp_counter = 0
        while True:
            with self.state_lock:
                if self.shared_state["app_quit"]:
                    break
            
            frame_rgb = self.camera.get_frame()
            if frame_rgb is None:
                time.sleep(0.01) # Wait for frames
                continue
            
            # Mirror frame for natural interaction
            frame_rgb = cv2.flip(frame_rgb, 1)

            # Send frame to MediaPipe async (non-blocking)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            frame_timestamp_counter += 1
            self.landmarker.detect_async(mp_image, frame_timestamp_counter)
            
            # Process results from the last callback
            is_pinching_now = False
            finger_tip_pos = (0, 0)
            
            if self.mp_hand_landmarks:
                hand = self.mp_hand_landmarks[0]
                thumb, index = hand[self.THUMB_TIP], hand[self.INDEX_FINGER_TIP]
                
                pinch_dist = math.hypot(index.x - thumb.x, index.y - thumb.y)
                is_pinching_now = pinch_dist < self.PINCH_THRESHOLD

                # Convert normalized coordinates to screen coordinates
                finger_tip_pos = (int(index.x * SCREEN_WIDTH), int(index.y * SCREEN_HEIGHT))

            with self.state_lock:
                self.shared_state["index_finger_tip"] = finger_tip_pos
                self.shared_state["is_pinching"] = is_pinching_now
            
            time.sleep(0.01) # Small sleep to prevent busy-waiting

        self.landmarker.close()
        print("[HandTrackingThread] Stopped.")


# --- Main Application Logic ---

def main():
    pygame.init()
    # Patch time module for SpiritCompanion to use pygame's time
    # (SpiritCompanion expects a 'time' module with .time() and .get_ticks())
    SpiritCompanion.time = pygame.time
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.NOFRAME)
    pygame.display.set_caption("Aletheia OS")
    pygame.mouse.set_visible(False)

    # --- GUI Components ---
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()
    
    spirit_companion = SpiritCompanion(shared_state, state_lock)
    all_sprites = pygame.sprite.Group(spirit_companion)
    grey_fog = GreyFog(shared_state, state_lock)
    detection_overlay = DetectionOverlay(shared_state, state_lock)
    health_bar = HealthBar(shared_state, state_lock)
    carbon_savings_widget = CarbonSavingsWidget(shared_state, state_lock)
    mission_tracker = MissionTracker(shared_state, state_lock)


    # --- Start Camera Controller ---
    print("[Main] Initializing Camera...")
    # The get_camera_manager function handles RPi vs. Webcam
    camera = get_camera_manager() 
    camera.start()
    print("[Main] Camera started.")

    # --- Start Worker Threads ---
    detection_thread = YoloDetectionThread(
        model_path=YOLO_MODEL_PATH,
        camera=camera,
        shared_state=shared_state,
        state_lock=state_lock
    )
    detection_thread.start()

    hand_thread = HandTrackingThread(
        model_path=HAND_LANDMARKER_MODEL_PATH,
        camera=camera,
        shared_state=shared_state,
        state_lock=state_lock
    )
    hand_thread.start()

    # --- Main Loop ---
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or \
               (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # --- Update shared state with current time for SpiritCompanion ---
        with state_lock:
            # Check if any of the threads have signaled to quit
            if shared_state["app_quit"]:
                running = False
            # Pass current time to shared state for accurate event tracking
            shared_state["current_time"] = pygame.time.get_ticks() * 0.001


        # --- Quest Logic: Pinch-to-activate detected objects ---
        with state_lock:
            is_pinching = shared_state["is_pinching"]
            index_finger_tip = shared_state["index_finger_tip"]
            detected_objects = shared_state["detected_objects"]
            active_quests = shared_state["active_quests"]
            carbon_saved_g = shared_state["carbon_saved_g"]
            missions_completed = shared_state["missions_completed"]

        if is_pinching and index_finger_tip != (0,0):
            finger_rect = pygame.Rect(index_finger_tip[0]-10, index_finger_tip[1]-10, 20, 20)
            for obj in detected_objects:
                obj_label = obj.get("label")
                obj_box_coords = obj.get("box") # (x1, y1, x2, y2)
                obj_impact = obj.get("carbon_impact", "unknown")

                if obj_label and obj_box_coords:
                    obj_rect = pygame.Rect(obj_box_coords[0], obj_box_coords[1],
                                           obj_box_coords[2]-obj_box_coords[0],
                                           obj_box_coords[3]-obj_box_coords[1])
                    
                    if finger_rect.colliderect(obj_rect):
                        if obj_label not in active_quests:
                            # Activate quest: mark object as 'quested'
                            active_quests[obj_label] = pygame.time.get_ticks() * 0.001 # Store activation time
                            
                            # Add carbon savings based on impact (example values)
                            if obj_impact == "high":
                                carbon_saved_g += 1000 # 1kg
                            elif obj_impact == "medium":
                                carbon_saved_g += 250 # 0.25kg
                            elif obj_impact == "low":
                                carbon_saved_g += 50 # 0.05kg
                            
                            missions_completed += 1
                            print(f"[Main] Quest activated for '{obj_label}'! Carbon Saved: {carbon_saved_g/1000:.2f}kg")

                            with state_lock:
                                shared_state["carbon_saved_g"] = carbon_saved_g
                                shared_state["active_quests"] = active_quests # Update the entire dict
                                shared_state["missions_completed"] = missions_completed
                                shared_state["last_savings_event_time"] = pygame.time.get_ticks() * 0.001


        # --- Drawing ---
        screen.fill(BLACK)

        # Draw all GUI elements
        grey_fog.draw(screen)
        all_sprites.update() # SpiritCompanion update depends on carbon_velocity
        all_sprites.draw(screen)
        detection_overlay.draw(screen)
        health_bar.draw(screen)
        carbon_savings_widget.draw(screen)
        mission_tracker.draw(screen)

        # Draw hand cursor
        # Cursor needs to be drawn *after* GUI elements, but before status bar text
        with state_lock:
            cursor_pos = shared_state["index_finger_tip"]
            is_pinching_draw = shared_state["is_pinching"]
        
        cursor_img = pygame.Surface((20, 20), pygame.SRCALPHA)
        if is_pinching_draw:
            pygame.draw.circle(cursor_img, (50, 255, 50, 220), (10, 10), 10, width=4)
        else:
            pygame.draw.circle(cursor_img, (255, 255, 255, 200), (10, 10), 10)
        # Blit cursor at its position, clamped to screen bounds
        cursor_rect = cursor_img.get_rect(center=cursor_pos)
        cursor_rect.clamp_ip(screen.get_rect()) # Ensure cursor stays on screen
        screen.blit(cursor_img, cursor_rect)


        # --- Debug / Status Bar ---
        with state_lock:
            carbon_v = shared_state["carbon_velocity"]
            det_count = shared_state["detection_count"]
            cpu_temp = shared_state["cpu_temp"]
            yolo_inference_ms = shared_state["yolo_inference_ms"]

        temp_str = f"{cpu_temp:.0f}°C" if cpu_temp > 0 else "N/A"
        temp_color = (255, 180, 0) if cpu_temp >= 70 else ((255, 50, 50) if cpu_temp >= 80 else (255, 255, 255))
        
        debug_text = f"{VERSION} | HUD FPS: {clock.get_fps():.0f} | Carbon: {carbon_v:.2f} | Objects: {det_count}"
        text_surface = font.render(debug_text, True, (255, 255, 255))
        screen.blit(text_surface, (20, 20))

        bottom_text = f"CPU: {temp_str} | YOLO: {yolo_inference_ms:.0f}ms"
        bottom_surface = small_font.render(bottom_text, True, temp_color)
        screen.blit(bottom_surface, (20, SCREEN_HEIGHT - 40))

        if cpu_temp >= 80:
            warn_surface = font.render("⚠ THERMAL THROTTLE", True, (255, 50, 50))
            screen.blit(warn_surface, (SCREEN_WIDTH // 2 - warn_surface.get_width() // 2, SCREEN_HEIGHT - 40))

        pygame.display.flip()
        clock.tick(30) # Cap main loop at 30 FPS

    # --- Shutdown ---
    print("[Main] Shutdown signal received. Stopping threads...")
    with state_lock:
        shared_state["app_quit"] = True

    detection_thread.join(timeout=5)
    hand_thread.join(timeout=5)
    camera.stop()
    
    pygame.quit()
    print("[Main] Aletheia OS has shut down cleanly.")

if __name__ == "__main__":
    main()