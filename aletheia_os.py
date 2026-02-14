# aletheia_os.py
# Core foundation for Project Aletheia - A Standalone AR OS
# SFHacks 2026

import pygame
import threading
import time
import math
import os
import sys

# from picamera2 import Picamera2 # Uncomment when on Raspberry Pi
import cv2
import mediapipe as mp
import numpy as np

# Add meta-yolo to path for YOLO imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta-yolo"))
from yolo_live import DetectionThread, get_cpu_temp

# --- Global Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
BLACK = (0, 0, 0)
VERSION = "Aletheia OS v0.2.0"

# Path to YOLO model
YOLO_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "meta-yolo", "yolo26n_xnnpack.pte"
)

# --- Shared State & Thread Safety ---
shared_state = {
    "carbon_velocity": 0.0,       # Range 0.0 to 1.0; affects fog and sprite
    "index_finger_tip": (0, 0),   # In HUD coordinates
    "is_pinching": False,
    "detected_objects": [],       # List of {"label": str, "box": (x1,y1,x2,y2), ...}
    "app_quit": False,
    "cpu_temp": 0.0,              # RPi CPU temperature
    "inference_ms": 0.0,          # Last YOLO inference time
    "detection_count": 0,         # Number of objects detected
}
state_lock = threading.Lock()


# --- AR HUD Components ---

class EcoSprite(pygame.sprite.Sprite):
    """
    A floating entity whose appearance and behavior are tied to environmental data.
    """
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((50, 50), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (0, 255, 150), (25, 25), 25)
        self.rect = self.image.get_rect(center=(SCREEN_WIDTH - 150, 150))

        # Bobbing animation state
        self.bob_angle = 0
        self.bob_speed = 0.02
        self.bob_amplitude = 10
        self.base_y = self.rect.y

        # State management
        self.state = "calm"  # "calm", "agitated", "critical"

    def update(self):
        # 1. Bobbing Animation
        self.bob_angle += self.bob_speed
        if self.bob_angle > 2 * math.pi:
            self.bob_angle -= 2 * math.pi
        self.rect.y = self.base_y + int(self.bob_amplitude * math.sin(self.bob_angle))

        # 2. State change based on Carbon Velocity
        with state_lock:
            carbon_v = shared_state["carbon_velocity"]

        new_state = "calm"
        if 0.3 <= carbon_v < 0.7:
            new_state = "agitated"
        elif carbon_v >= 0.7:
            new_state = "critical"

        if new_state != self.state:
            self.state = new_state
            self.update_appearance()

    def update_appearance(self):
        """Update sprite visuals based on its current state."""
        self.image.fill((0, 0, 0, 0))  # Clear
        if self.state == "calm":
            pygame.draw.circle(self.image, (0, 255, 150), (25, 25), 25)
        elif self.state == "agitated":
            pygame.draw.circle(self.image, (255, 180, 0), (25, 25), 25)
        elif self.state == "critical":
            pygame.draw.circle(self.image, (255, 50, 50), (25, 25), 25)
            pygame.draw.circle(self.image, (255, 255, 255), (25, 25), 25, width=3)


class GreyFog:
    """
    An overlay that represents the carbon impact, becoming more opaque
    as the 'carbon_velocity' increases.
    """
    def __init__(self):
        self.surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.current_alpha = 0

    def draw(self, screen):
        with state_lock:
            carbon_v = shared_state["carbon_velocity"]

        target_alpha = int(carbon_v * 200)
        self.current_alpha = self.current_alpha * 0.95 + target_alpha * 0.05

        self.surface.fill((50, 50, 55, int(self.current_alpha)))
        screen.blit(self.surface, (0, 0))


class DetectionOverlay:
    """
    Draws detected objects and their carbon impact labels on the HUD.
    Shows what the YOLO model is currently seeing.
    """
    def __init__(self):
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.impact_colors = {
            "high": (255, 50, 50),       # Red
            "medium": (255, 180, 0),     # Orange
            "low": (0, 220, 100),        # Green
            "unknown": (180, 180, 180),  # Grey
        }

    def draw(self, screen):
        with state_lock:
            detections = shared_state["detected_objects"]

        for det in detections:
            label = det["label"]
            conf = det["confidence"]
            impact = det.get("carbon_impact", "unknown")
            color = self.impact_colors.get(impact, (180, 180, 180))

            # Draw object label tag (floating near top-left of screen for AR overlay)
            # In a full AR system these would be positioned in 3D space
            # For now, show as a list on the left side
            # (Box drawing is skipped since the camera feed isn't shown on the HUD)

        # Draw detection summary panel on the left
        if detections:
            # Background panel
            panel_h = min(len(detections), 8) * 30 + 50
            panel_surface = pygame.Surface((320, panel_h), pygame.SRCALPHA)
            panel_surface.fill((0, 0, 0, 140))
            screen.blit(panel_surface, (20, 70))

            # Title
            title = self.font.render("Detected Objects", True, (255, 255, 255))
            screen.blit(title, (30, 78))

            # List objects (max 8)
            y = 108
            for i, det in enumerate(detections[:8]):
                color = self.impact_colors.get(det.get("carbon_impact", "unknown"), (180, 180, 180))

                # Impact dot
                pygame.draw.circle(screen, color, (40, y + 8), 5)

                # Label + confidence
                text = f"{det['label']} ({det['confidence']:.0%})"
                text_surf = self.small_font.render(text, True, (255, 255, 255))
                screen.blit(text_surf, (52, y))

                # Carbon impact tag
                impact_text = det.get("carbon_impact", "?")
                impact_surf = self.small_font.render(impact_text, True, color)
                screen.blit(impact_surf, (260, y))

                y += 30

            if len(detections) > 8:
                more = self.small_font.render(
                    f"+{len(detections) - 8} more...", True, (150, 150, 150))
                screen.blit(more, (52, y))


# --- Background Threads ---

def hand_tracking_thread():
    """
    Handles camera input and hand tracking (MediaPipe).
    Runs on the same camera as YOLO but processes every frame for smooth tracking.
    """
    print("[HandTracking] Thread started.")

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[HandTracking] ERROR: Could not open camera.")
        return

    while True:
        with state_lock:
            if shared_state["app_quit"]:
                break

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(image_rgb)
        is_pinching_now = False
        finger_tip_pos = (0, 0)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

            if distance < 0.05:
                is_pinching_now = True

            hud_x = int(index_tip.x * SCREEN_WIDTH)
            hud_y = int(index_tip.y * SCREEN_HEIGHT)
            finger_tip_pos = (hud_x, hud_y)

        with state_lock:
            shared_state["index_finger_tip"] = finger_tip_pos
            shared_state["is_pinching"] = is_pinching_now

    cap.release()
    hands.close()
    print("[HandTracking] Thread finished.")


# --- Main Application Logic ---

def main():
    """
    Main function to initialize Pygame and run the AR HUD loop.
    """
    pygame.init()

    # Setup the display - borderless fullscreen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.NOFRAME)
    pygame.display.set_caption("Aletheia OS")
    pygame.mouse.set_visible(False)

    # --- AR Components ---
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()
    eco_sprite = EcoSprite()
    all_sprites = pygame.sprite.Group(eco_sprite)
    grey_fog = GreyFog()
    detection_overlay = DetectionOverlay()

    # --- Cursor for hand tracking ---
    cursor_img = pygame.Surface((20, 20), pygame.SRCALPHA)
    pygame.draw.circle(cursor_img, (255, 255, 255, 200), (10, 10), 10)

    # --- Start YOLO Detection Thread ---
    print(f"[Main] Starting YOLO detection thread...")
    print(f"[Main] Model: {YOLO_MODEL_PATH}")

    detection_thread = DetectionThread(
        model_path=YOLO_MODEL_PATH,
        shared_state=shared_state,
        state_lock=state_lock,
        camera_index=0,
        base_interval=2.5,     # Run detection every 2.5 seconds
        confidence=0.25,
    )
    detection_thread.start()

    # --- Start Hand Tracking Thread ---
    # NOTE: If using the same camera for both YOLO and hand tracking,
    # you may need to use a single camera thread and share frames.
    # For now, YOLO uses its own camera capture in DetectionThread.
    # Uncomment below if you have a second camera or want hand tracking:
    #
    # hand_thread = threading.Thread(target=hand_tracking_thread, daemon=True)
    # hand_thread.start()

    # --- Main Loop ---
    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Update Step ---
        all_sprites.update()

        # --- Draw Step ---
        screen.fill(BLACK)

        # Draw Grey Fog overlay (carbon impact atmosphere)
        grey_fog.draw(screen)

        # Draw AR sprites
        all_sprites.draw(screen)

        # Draw detection overlay (object list)
        detection_overlay.draw(screen)

        # Get latest state
        with state_lock:
            cursor_pos = shared_state["index_finger_tip"]
            is_pinching = shared_state["is_pinching"]
            carbon_v = shared_state["carbon_velocity"]
            cpu_temp = shared_state.get("cpu_temp", 0)
            inference_ms = shared_state.get("inference_ms", 0)
            det_count = shared_state.get("detection_count", 0)

        # Draw the hand cursor
        cursor_img.fill((0, 0, 0, 0))
        if is_pinching:
            pygame.draw.circle(cursor_img, (50, 255, 50, 220), (10, 10), 10, width=4)
        else:
            pygame.draw.circle(cursor_img, (255, 255, 255, 200), (10, 10), 10)
        screen.blit(cursor_img, cursor_pos)

        # --- Debug / Status Bar ---
        temp_str = f"{cpu_temp:.0f}°C" if cpu_temp > 0 else "N/A"
        temp_color = (255, 255, 255)
        if cpu_temp >= 80:
            temp_color = (255, 50, 50)
        elif cpu_temp >= 70:
            temp_color = (255, 180, 0)

        # Top status line
        debug_text = (f"{VERSION} | HUD FPS: {clock.get_fps():.0f} | "
                      f"Carbon: {carbon_v:.2f} | Objects: {det_count}")
        text_surface = font.render(debug_text, True, (255, 255, 255))
        screen.blit(text_surface, (20, 20))

        # Bottom status line (thermal + inference)
        bottom_text = f"CPU: {temp_str} | YOLO: {inference_ms:.0f}ms"
        bottom_surface = small_font.render(bottom_text, True, temp_color)
        screen.blit(bottom_surface, (20, SCREEN_HEIGHT - 40))

        # Thermal warning
        if cpu_temp >= 80:
            warn_surface = font.render("⚠ THERMAL THROTTLE", True, (255, 50, 50))
            screen.blit(warn_surface, (SCREEN_WIDTH // 2 - 120, SCREEN_HEIGHT - 40))

        # --- Final Flip ---
        pygame.display.flip()
        clock.tick(60)

    # Signal threads to quit
    with state_lock:
        shared_state["app_quit"] = True

    print("[Main] Waiting for threads to finish...")
    detection_thread.join(timeout=5)
    pygame.quit()
    print("[Main] Aletheia OS has shut down cleanly.")


if __name__ == "__main__":
    main()
