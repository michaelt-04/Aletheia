# hand_tracking_test.py
# Standalone test for MediaPipe hand tracking with pinch-to-drag.
# RPi Version: Uses the RPiCamera controller.
#
# Controls: ESC or Ctrl+X to quit

import pygame
import cv2
import mediapipe as mp
import math
import sys
import os
import threading
import time

# Import the RPi camera controller
from camera_rpi import get_camera_manager

# --- Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
CAM_WIDTH, CAM_HEIGHT = 1280, 720 # Camera resolution
PINCH_THRESHOLD = 0.05
OBJECT_RADIUS = 30
FPS = 60

# Path to hand landmarker model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# MediaPipe Tasks API aliases
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand landmark indices
THUMB_TIP = 4
INDEX_FINGER_TIP = 8

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (50, 255, 50)
CYAN = (0, 220, 255)
RED = (255, 60, 60)
YELLOW = (255, 220, 50)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Hand landmarker model not found at {MODEL_PATH}")
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Aletheia - RPi Hand Tracking Test")
    clock = pygame.time.Clock()
    small_font = pygame.font.Font(None, 24)

    # --- Camera setup ---
    camera = get_camera_manager(width=CAM_WIDTH, height=CAM_HEIGHT)
    camera.start()
    print("[TestScript] Waiting for camera to provide frames...")
    while camera.get_frame() is None:
        time.sleep(0.1)
    print("[TestScript] Camera ready.")

    # --- Shared tracking state (updated async from MediaPipe callback) ---
    tracking = { "finger_pos": None, "pinch_pos": None, "is_pinching": False, "pinch_distance": 1.0 }
    tracking_lock = threading.Lock()

    def on_result(result, output_image, timestamp_ms):
        """Callback for LIVE_STREAM mode"""
        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            thumb, index = hand[THUMB_TIP], hand[INDEX_FINGER_TIP]
            
            pinch_dist = math.hypot(index.x - thumb.x, index.y - thumb.y)
            ix, iy = int(index.x * SCREEN_WIDTH), int(index.y * SCREEN_HEIGHT)
            is_pinching = pinch_dist < PINCH_THRESHOLD
            
            with tracking_lock:
                tracking["finger_pos"] = (ix, iy)
                tracking["is_pinching"] = is_pinching
                tracking["pinch_distance"] = pinch_dist
                if is_pinching:
                    mx = int(((thumb.x + index.x) / 2) * SCREEN_WIDTH)
                    my = int(((thumb.y + index.y) / 2) * SCREEN_HEIGHT)
                    tracking["pinch_pos"] = (mx, my)
                else:
                    tracking["pinch_pos"] = None
        else: # No hand detected
             with tracking_lock:
                tracking["finger_pos"] = None
                tracking["pinch_pos"] = None
                tracking["is_pinching"] = False

    # --- MediaPipe Hand Landmarker (async LIVE_STREAM mode) ---
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        result_callback=on_result,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # --- Draggable object state ---
    obj_pos = pygame.Vector2(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    is_grabbed = False
    grab_offset = pygame.Vector2(0, 0)
    frame_timestamp = 0

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # --- Get frame from RPiCamera ---
        frame_rgb = camera.get_frame()
        if frame_rgb is None:
            continue

        frame_rgb = cv2.flip(frame_rgb, 1)  # Mirror for natural interaction

        # --- Send frame to MediaPipe async (non-blocking) ---
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        frame_timestamp += 1
        landmarker.detect_async(mp_image, frame_timestamp)

        with tracking_lock:
            pinch_pos = tracking["pinch_pos"]
            is_pinching = tracking["is_pinching"]

        # --- Grab logic ---
        if is_pinching and pinch_pos:
            if not is_grabbed and obj_pos.distance_to(pinch_pos) < OBJECT_RADIUS + 40:
                is_grabbed = True
                grab_offset = obj_pos - pinch_pos
            if is_grabbed:
                obj_pos = pygame.Vector2(pinch_pos) + grab_offset
        else:
            is_grabbed = False
            
        obj_pos.x = max(OBJECT_RADIUS, min(SCREEN_WIDTH - OBJECT_RADIUS, obj_pos.x))
        obj_pos.y = max(OBJECT_RADIUS, min(SCREEN_HEIGHT - OBJECT_RADIUS, obj_pos.y))

        # --- Drawing ---
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        frame_surface = pygame.surfarray.make_surface(frame_bgr.swapaxes(0, 1))
        dark_overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        dark_overlay.fill((0, 0, 0, 140))
        screen.blit(frame_surface, (0, 0))
        screen.blit(dark_overlay, (0, 0))

        # Draw draggable object
        obj_color = GREEN if is_grabbed else CYAN
        pygame.draw.circle(screen, obj_color, (int(obj_pos.x), int(obj_pos.y)), OBJECT_RADIUS)
        pygame.draw.circle(screen, WHITE, (int(obj_pos.x), int(obj_pos.y)), OBJECT_RADIUS, 2)
        
        # Draw cursor
        with tracking_lock:
            finger_pos = tracking["finger_pos"]
        if finger_pos:
            if is_pinching:
                pygame.draw.circle(screen, GREEN, pinch_pos, 12, 3)
            else:
                pygame.draw.circle(screen, WHITE, finger_pos, 8, 2)

        pygame.display.flip()
        clock.tick(FPS)

    # --- Cleanup ---
    camera.stop()
    landmarker.close()
    pygame.quit()
    print("[TestScript] Test finished cleanly.")

if __name__ == "__main__":
    main()
