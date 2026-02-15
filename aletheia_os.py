# aletheia_os.py
# Core foundation for Project Aletheia - A Standalone AR OS for Raspberry Pi
# SFHacks 2026

import pygame
import threading
import time
import math
import os
import sys
from aletheia_gui import SpiritCompanion, GreyFog, DetectionOverlay, HealthBar, MissionTracker

# RPi-specific camera controller
from camera_rpi import RPiCamera

import cv2
import mediapipe as mp
import numpy as np

# Add meta-yolo to path for YOLO imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta-yolo"))
from yolo_live import DetectionThread, get_cpu_temp

# --- Global Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
BLACK = (0, 0, 0)
VERSION = "Aletheia OS v0.3.0 RPi"

# Path to YOLO model
YOLO_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "meta-yolo", "yolo26n_xnnpack.pte"
)

# --- Shared State & Thread Safety ---
shared_state = {
    "carbon_velocity": 0.0,
    "index_finger_tip": (0, 0),
    "is_pinching": False,
    "detected_objects": [],
    "app_quit": False,
    "cpu_temp": 0.0,
    "inference_ms": 0.0,
    "detection_count": 0,
    "health": 100,
    "missions_completed": 0,
    "missions_total": 5
    }
state_lock = threading.Lock()


# --- Background Hand Tracking Thread ---

class HandTrackingThread(threading.Thread):
    """
    Handles hand tracking using MediaPipe on frames from the shared camera.
    """
    def __init__(self, camera, shared_state, state_lock):
        super().__init__(daemon=True)
        self.camera = camera
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

    def run(self):
        print("[HandTracking] Thread started, waiting for camera...")
        while True:
            with self.state_lock:
                if self.shared_state["app_quit"]:
                    break
            
            frame_rgb = self.camera.get_frame()
            if frame_rgb is None:
                time.sleep(0.01) # Wait for frames
                continue
            
            # Flip for mirror view and process
            frame_rgb = cv2.flip(frame_rgb, 1)
            results = self.hands.process(frame_rgb)
            
            is_pinching_now = False
            finger_tip_pos = (0, 0)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)
                if distance < 0.05:
                    is_pinching_now = True

                hud_x = int(index_tip.x * SCREEN_WIDTH)
                hud_y = int(index_tip.y * SCREEN_HEIGHT)
                finger_tip_pos = (hud_x, hud_y)

            with self.state_lock:
                self.shared_state["index_finger_tip"] = finger_tip_pos
                self.shared_state["is_pinching"] = is_pinching_now
        
        self.hands.close()
        print("[HandTracking] Thread finished.")


# --- Main Application Logic ---

def main():
    pygame.init()
    SpiritCompanion.time = time # Patch time module for SpiritCompanion
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
    #experience_bar = ExperienceBar(shared_state, state_lock)
    mission_tracker = MissionTracker(shared_state, state_lock)


    # --- Start RPi Camera Controller ---
    print("[Main] Initializing RPi Camera...")
    rpi_camera = RPiCamera()
    rpi_camera.start()

    # --- Start Worker Threads ---
    # Start YOLO Detection Thread, passing the camera object
    detection_thread = DetectionThread(
        model_path=YOLO_MODEL_PATH,
        camera=rpi_camera,
        shared_state=shared_state,
        state_lock=state_lock
    )
    detection_thread.start()

    # Start Hand Tracking Thread, passing the same camera object
    hand_thread = HandTrackingThread(
        camera=rpi_camera,
        shared_state=shared_state,
        state_lock=state_lock
    )
    hand_thread.start()

    # --- Main Loop ---
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        all_sprites.update()
        screen.fill(BLACK)

        # Draw all GUI elements
        grey_fog.draw(screen)
        all_sprites.draw(screen)
        detection_overlay.draw(screen)
        health_bar.draw(screen)
        mission_tracker.draw(screen)

        # Draw hand cursor
        with state_lock:
            cursor_pos = shared_state["index_finger_tip"]
            is_pinching = shared_state["is_pinching"]
        
        cursor_img = pygame.Surface((20, 20), pygame.SRCALPHA)
        if is_pinching:
            pygame.draw.circle(cursor_img, (50, 255, 50, 220), (10, 10), 10, width=4)
        else:
            pygame.draw.circle(cursor_img, (255, 255, 255, 200), (10, 10), 10)
        screen.blit(cursor_img, cursor_pos)

        # --- Debug / Status Bar ---
        with state_lock:
            carbon_v = shared_state["carbon_velocity"]
            det_count = shared_state["detection_count"]
            cpu_temp = shared_state["cpu_temp"]
            inference_ms = shared_state["inference_ms"]

        temp_str = f"{cpu_temp:.0f}°C" if cpu_temp > 0 else "N/A"
        temp_color = (255, 180, 0) if cpu_temp >= 70 else (255, 50, 50) if cpu_temp >= 80 else (255, 255, 255)
        
        debug_text = f"{VERSION} | HUD FPS: {clock.get_fps():.0f} | Carbon: {carbon_v:.2f} | Objects: {det_count}"
        text_surface = font.render(debug_text, True, (255, 255, 255))
        screen.blit(text_surface, (20, 20))

        bottom_text = f"CPU: {temp_str} | YOLO: {inference_ms:.0f}ms"
        bottom_surface = small_font.render(bottom_text, True, temp_color)
        screen.blit(bottom_surface, (20, SCREEN_HEIGHT - 40))

        if cpu_temp >= 80:
            warn_surface = font.render("⚠ THERMAL THROTTLE", True, (255, 50, 50))
            screen.blit(warn_surface, (SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT - 40))

        pygame.display.flip()
        clock.tick(60)

    # --- Shutdown ---
    print("[Main] Shutdown signal received. Stopping threads...")
    with state_lock:
        shared_state["app_quit"] = True

    rpi_camera.stop()
    detection_thread.join(timeout=2)
    hand_thread.join(timeout=2)
    
    pygame.quit()
    print("[Main] Aletheia OS has shut down cleanly.")

if __name__ == "__main__":
    main()
