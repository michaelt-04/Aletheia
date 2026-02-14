# aletheia_os.py
# Core foundation for Project Aletheia - A Standalone AR OS
# SFHacks 2026

import pygame
import threading
import time
import math
import random
# from picamera2 import Picamera2 # Uncomment when on Raspberry Pi
import cv2
import mediapipe as mp
import numpy as np

# --- Global Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
BLACK = (0, 0, 0)
VERSION = "Aletheia OS v0.1.0"

# --- Shared State & Thread Safety ---
# This dictionary will hold shared data between threads.
# It's crucial to use locks when accessing shared data to prevent race conditions.
shared_state = {
    "carbon_velocity": 0.0,  # Range 0.0 to 1.0; affects fog and sprite
    "index_finger_tip": (0, 0),  # In HUD coordinates
    "is_pinching": False,
    "detected_objects": [], # List of {"label": "name", "box": (x,y,w,h)}
    "app_quit": False
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
        pygame.draw.circle(self.image, (0, 255, 150), (25, 25), 25) # Default "calm" state
        self.rect = self.image.get_rect(center=(SCREEN_WIDTH - 150, 150))
        
        # Bobbing animation state
        self.bob_angle = 0
        self.bob_speed = 0.02
        self.bob_amplitude = 10
        self.base_y = self.rect.y

        # State management
        self.state = "calm" # "calm", "agitated", "critical"

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
        """ Update sprite visuals based on its current state. """
        if self.state == "calm":
            pygame.draw.circle(self.image, (0, 255, 150), (25, 25), 25) # Green
        elif self.state == "agitated":
            pygame.draw.circle(self.image, (255, 180, 0), (25, 25), 25) # Orange
        elif self.state == "critical":
            pygame.draw.circle(self.image, (255, 50, 50), (25, 25), 25) # Red
            # Add more visual flair, like a pulsing border
            pygame.draw.circle(self.image, (255,255,255), (25, 25), 25, width=3)


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
        
        # Map carbon_velocity (0.0-1.0) to alpha (0-200)
        target_alpha = int(carbon_v * 200)
        
        # Smoothly interpolate to the target alpha
        self.current_alpha = self.current_alpha * 0.95 + target_alpha * 0.05
        
        self.surface.fill((50, 50, 55, self.current_alpha))
        screen.blit(self.surface, (0, 0))

# --- Background Threads ---

def vision_processing_thread():
    """
    Handles camera input, hand tracking (MediaPipe), and object detection (ExecuTorch).
    """
    print("Vision thread started.")
    
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    mp_drawing = mp.solutions.drawing_utils

    # TODO: Initialize ExecuTorch YOLOv11 Model
    # yolo_model = load_yolo_v11_executorch_model("path/to/yolov11.pte")

    # Initialize Camera
    # Using OpenCV for compatibility, switch to Picamera2 for performance on Pi
    cap = cv2.VideoCapture(0) # Use 0 for default camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)

    while True:
        with state_lock:
            if shared_state["app_quit"]: break

        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # For performance, process a smaller image and flip
        frame = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- 1. MediaPipe Hand Tracking ---
        results = hands.process(image_rgb)
        is_pinching_now = False
        finger_tip_pos = (0, 0)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0] # Single hand
            
            # Get landmarks for thumb tip and index finger tip
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate distance between them
            distance = math.hypot(index_tip.x - thumb_tip.x, index_tip.y - thumb_tip.y)

            # Detect "pinch"
            if distance < 0.05: # Threshold may need tuning
                is_pinching_now = True

            # Map index finger position to HUD space
            h, w, _ = image_rgb.shape
            hud_x = int(index_tip.x * SCREEN_WIDTH)
            hud_y = int(index_tip.y * SCREEN_HEIGHT)
            finger_tip_pos = (hud_x, hud_y)

        # --- 2. ExecuTorch Object Detection ---
        # Placeholder for YOLOv11 object detection logic
        # This would be a heavy operation, run it periodically (e.g., every N frames)
        # detected_objects_list = yolo_model.predict(image_rgb)
        detected_objects_list = [] # Placeholder

        # --- 3. Update Shared State ---
        with state_lock:
            shared_state["index_finger_tip"] = finger_tip_pos
            shared_state["is_pinching"] = is_pinching_now
            shared_state["detected_objects"] = detected_objects_list
            # Demo: Link carbon velocity to pinch for testing
            if is_pinching_now:
                 shared_state["carbon_velocity"] = min(1.0, shared_state["carbon_velocity"] + 0.01)
            else:
                 shared_state["carbon_velocity"] = max(0.0, shared_state["carbon_velocity"] - 0.005)


    # Cleanup
    cap.release()
    hands.close()
    print("Vision thread finished.")


# --- Main Application Logic ---

def main():
    """
    Main function to initialize Pygame and run the AR HUD loop.
    """
    pygame.init()

    # Setup the display - borderless fullscreen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.NOFRAME)
    pygame.display.set_caption("Aletheia OS")
    
    # Hide mouse cursor
    pygame.mouse.set_visible(False)

    # --- AR Components ---
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()
    eco_sprite = EcoSprite()
    all_sprites = pygame.sprite.Group(eco_sprite)
    grey_fog = GreyFog()
    
    # --- Cursor for hand tracking ---
    cursor_img = pygame.Surface((20, 20), pygame.SRCALPHA)
    pygame.draw.circle(cursor_img, (255, 255, 255, 200), (10, 10), 10)
    
    # --- Start Vision Thread ---
    vision_thread = threading.Thread(target=vision_processing_thread, daemon=True)
    vision_thread.start()

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
        # The background is "transparent" by simply not drawing anything (black)
        screen.fill(BLACK)
        
        # Draw AR elements
        all_sprites.draw(screen)
        grey_fog.draw(screen)
        
        # Get latest state from vision thread
        with state_lock:
            cursor_pos = shared_state["index_finger_tip"]
            is_pinching = shared_state["is_pinching"]
            carbon_v = shared_state["carbon_velocity"]

        # Draw the hand cursor
        if is_pinching:
             pygame.draw.circle(cursor_img, (50, 255, 50, 220), (10, 10), 10, width=4)
        else:
             pygame.draw.circle(cursor_img, (255, 255, 255, 200), (10, 10), 10)
        screen.blit(cursor_img, cursor_pos)

        # Draw debug info
        debug_text = f"{VERSION} | FPS: {clock.get_fps():.1f} | Carbon: {carbon_v:.2f}"
        text_surface = font.render(debug_text, True, (255, 255, 255))
        screen.blit(text_surface, (20, 20))

        # --- Final Flip ---
        pygame.display.flip()
        clock.tick(60)

    # Signal threads to quit
    with state_lock:
        shared_state["app_quit"] = True
    
    print("Waiting for threads to finish...")
    vision_thread.join() # Wait for the vision thread to clean up
    pygame.quit()
    print("Aletheia OS has shut down cleanly.")

if __name__ == "__main__":
    main()
