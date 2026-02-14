# hand_tracking_test.py
# Standalone test for MediaPipe hand tracking with pinch-to-drag
# Uses Mac camera + pygame display (MediaPipe Tasks API)
# Controls: ESC or Ctrl+X to quit

import pygame
import cv2
import mediapipe as mp
import math
import sys
import os
import threading
import time

# --- Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
CAM_WIDTH, CAM_HEIGHT = 1280, 720
PINCH_THRESHOLD = 0.05       # Normalized distance to trigger pinch
OBJECT_RADIUS = 30
FPS = 60

# Path to hand landmarker model (download from MediaPipe if missing)
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
        print("Download it with:")
        print('  curl -L -o hand_landmarker.task "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"')
        sys.exit(1)

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Aletheia - Hand Tracking Test")
    clock = pygame.time.Clock()
    small_font = pygame.font.Font(None, 24)

    # --- Camera setup ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        print("ERROR: Could not open camera.")
        sys.exit(1)

    # --- Shared tracking state (updated async from MediaPipe callback) ---
    tracking = {
        "finger_pos": None,
        "pinch_pos": None,
        "is_pinching": False,
        "pinch_distance": 1.0,
    }
    tracking_lock = threading.Lock()

    def on_result(result, output_image, timestamp_ms):
        """Callback for LIVE_STREAM mode — runs on MediaPipe's thread."""
        _finger_pos = None
        _pinch_pos = None
        _is_pinching = False
        _pinch_distance = 1.0

        if result.hand_landmarks:
            hand = result.hand_landmarks[0]
            thumb = hand[THUMB_TIP]
            index = hand[INDEX_FINGER_TIP]

            _pinch_distance = math.hypot(index.x - thumb.x, index.y - thumb.y)

            ix, iy = int(index.x * SCREEN_WIDTH), int(index.y * SCREEN_HEIGHT)
            _finger_pos = (ix, iy)

            if _pinch_distance < PINCH_THRESHOLD:
                _is_pinching = True
                mx = int(((thumb.x + index.x) / 2) * SCREEN_WIDTH)
                my = int(((thumb.y + index.y) / 2) * SCREEN_HEIGHT)
                _pinch_pos = (mx, my)

        with tracking_lock:
            tracking["finger_pos"] = _finger_pos
            tracking["pinch_pos"] = _pinch_pos
            tracking["is_pinching"] = _is_pinching
            tracking["pinch_distance"] = _pinch_distance

    # --- MediaPipe Hand Landmarker (async LIVE_STREAM mode) ---
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
        result_callback=on_result,
    )
    landmarker = HandLandmarker.create_from_options(options)

    # --- Draggable object state ---
    obj_x, obj_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
    is_grabbed = False
    grab_offset_x = 0
    grab_offset_y = 0
    frame_timestamp = 0

    running = True
    while running:
        # --- Events ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_x and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    running = False

        # --- Read camera frame ---
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # Mirror for natural interaction
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Send frame to MediaPipe async (non-blocking) ---
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        frame_timestamp += 1
        landmarker.detect_async(mp_image, frame_timestamp)

        # --- Read latest tracking results ---
        with tracking_lock:
            finger_pos = tracking["finger_pos"]
            pinch_pos = tracking["pinch_pos"]
            is_pinching = tracking["is_pinching"]
            pinch_distance = tracking["pinch_distance"]

        # --- Grab logic ---
        if is_pinching and pinch_pos:
            if not is_grabbed:
                dx = pinch_pos[0] - obj_x
                dy = pinch_pos[1] - obj_y
                dist_to_obj = math.hypot(dx, dy)
                if dist_to_obj < OBJECT_RADIUS + 40:
                    is_grabbed = True
                    grab_offset_x = obj_x - pinch_pos[0]
                    grab_offset_y = obj_y - pinch_pos[1]
            else:
                obj_x = pinch_pos[0] + grab_offset_x
                obj_y = pinch_pos[1] + grab_offset_y
        else:
            is_grabbed = False

        # Clamp object to screen
        obj_x = max(OBJECT_RADIUS, min(SCREEN_WIDTH - OBJECT_RADIUS, obj_x))
        obj_y = max(OBJECT_RADIUS, min(SCREEN_HEIGHT - OBJECT_RADIUS, obj_y))

        # --- Drawing ---
        # Convert camera frame to pygame surface (dim it so UI pops)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        dark_overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        dark_overlay.fill((0, 0, 0, 140))

        screen.blit(frame_surface, (0, 0))
        screen.blit(dark_overlay, (0, 0))

        # Draw the draggable object
        obj_color = GREEN if is_grabbed else CYAN
        glow_alpha = pygame.Surface((OBJECT_RADIUS * 4, OBJECT_RADIUS * 4), pygame.SRCALPHA)
        glow_color = (*obj_color, 60)
        pygame.draw.circle(glow_alpha, glow_color, (OBJECT_RADIUS * 2, OBJECT_RADIUS * 2), OBJECT_RADIUS + 12)
        screen.blit(glow_alpha, (obj_x - OBJECT_RADIUS * 2, obj_y - OBJECT_RADIUS * 2))
        pygame.draw.circle(screen, obj_color, (obj_x, obj_y), OBJECT_RADIUS)
        pygame.draw.circle(screen, WHITE, (obj_x, obj_y), OBJECT_RADIUS, 2)

        # Draw cursor / finger indicator
        if finger_pos:
            if is_pinching:
                pygame.draw.circle(screen, GREEN, pinch_pos, 12, 3)
                pygame.draw.circle(screen, GREEN, pinch_pos, 4)
            else:
                pygame.draw.circle(screen, WHITE, finger_pos, 8, 2)

        # --- HUD text ---
        status = "GRABBED" if is_grabbed else ("PINCHING" if is_pinching else "OPEN")
        status_color = GREEN if is_grabbed else (YELLOW if is_pinching else WHITE)
        hud_lines = [
            (f"Hand Tracking Test | FPS: {clock.get_fps():.0f}", WHITE),
            (f"Pinch distance: {pinch_distance:.3f}  (threshold: {PINCH_THRESHOLD})", WHITE),
            (f"Status: {status}", status_color),
        ]
        for i, (text, color) in enumerate(hud_lines):
            surf = small_font.render(text, True, color)
            screen.blit(surf, (16, 16 + i * 24))

        hint = "Pinch near the orb to grab and drag it"
        hint_surf = small_font.render(hint, True, (180, 180, 180))
        screen.blit(hint_surf, (SCREEN_WIDTH // 2 - hint_surf.get_width() // 2, SCREEN_HEIGHT - 36))

        pygame.display.flip()
        clock.tick(FPS)

    # Cleanup
    cap.release()
    landmarker.close()
    pygame.quit()


if __name__ == "__main__":
    main()
