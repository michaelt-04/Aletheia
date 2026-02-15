# spirit_test.py
# GUI Test Harness for Aletheia OS Components
#
# Usage:
# python spirit_test.py
#
# This script is designed to run on a development machine (like macOS)
# to test all GUI elements from aletheia_gui.py without needing a camera,
# a working YOLO model, or Raspberry Pi-specific hardware.

import pygame
import threading
import time
import math
import random  # SpiritCompanion uses this but doesn't import it, so we do.
from aletheia_gui import SpiritCompanion, GreyFog, DetectionOverlay, HealthBar, MissionTracker

# --- Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
BLACK = (0, 0, 0)
VERSION = "GUI Test Harness v1.0"

# --- Mock Shared State & Lock ---
# This dictionary simulates the main shared_state from aletheia_os.py
# We will manipulate it in the main loop to test the GUI's reaction.
shared_state = {
    "carbon_velocity": 0.0,
    "index_finger_tip": (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2),
    "is_pinching": False,
    "detected_objects": [],
    "app_quit": False,
    "cpu_temp": 45.0,
    "inference_ms": 35.0,
    "detection_count": 0,
    "health": 100,
    "experience": 0,
}
state_lock = threading.Lock()

# --- Main Test Application ---

def main():
    pygame.init()
    # Import time for SpiritCompanion jump timer initialization
    # This patches the missing import in the class itself
    SpiritCompanion.time = time

    # Setup display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(VERSION)

    # --- Instantiate all GUI Components ---
    # We pass them our mock shared state and lock
    spirit_companion = SpiritCompanion(shared_state, state_lock)
    all_sprites = pygame.sprite.Group(spirit_companion)
    grey_fog = GreyFog(shared_state, state_lock)
    detection_overlay = DetectionOverlay(shared_state, state_lock)
    health_bar = HealthBar(shared_state, state_lock)
    mission_tracker = MissionTracker(shared_state, state_lock)
    
    # Fonts for test info display
    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()

    # --- Main Test Loop ---
    running = True
    start_time = time.time()
    
    # Mock data for cycling
    mock_detections = [
        {"label": "laptop", "confidence": 0.88, "carbon_impact": "high"},
        {"label": "bottle", "confidence": 0.75, "carbon_impact": "medium"},
        {"label": "potted plant", "confidence": 0.92, "carbon_impact": "low"},
        {"label": "cell phone", "confidence": 0.65, "carbon_impact": "high"},
        {"label": "book", "confidence": 0.81, "carbon_impact": "unknown"},
    ]

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # --- Simulate State Changes ---
        elapsed_time = time.time() - start_time

        # 1. Cycle carbon_velocity from 0.0 to 1.0 and back every 10 seconds
        carbon_cycle = (math.sin(elapsed_time * (2 * math.pi / 10)) + 1) / 2
        
        # 2. Cycle health and experience
        health_cycle = (math.cos(elapsed_time * 0.5) + 1) / 2 * 100
        xp_cycle = int((elapsed_time * 5) % 100)
        
        # 3. Simulate mouse as hand cursor and pinch on click
        mouse_pos = pygame.mouse.get_pos()
        is_pinching_now = pygame.mouse.get_pressed()[0]
        
        # 4. Cycle through mock detections
        num_dets = int(((math.sin(elapsed_time * 0.8) + 1) / 2) * (len(mock_detections) + 1))
        
        # Update the shared state dictionary (must use lock)
        with state_lock:
            shared_state["carbon_velocity"] = carbon_cycle
            shared_state["health"] = health_cycle
            shared_state["experience"] = xp_cycle
            shared_state["index_finger_tip"] = mouse_pos
            shared_state["is_pinching"] = is_pinching_now
            shared_state["detected_objects"] = mock_detections[:num_dets]
            shared_state["detection_count"] = len(mock_detections[:num_dets])

        # --- Pygame Update and Draw ---
        all_sprites.update()
        
        screen.fill(BLACK)
        
        # Draw all GUI components
        grey_fog.draw(screen)
        all_sprites.draw(screen)
        detection_overlay.draw(screen)
        health_bar.draw(screen)
        mission_tracker.draw(screen)

        # Draw a mock cursor (since the real one is in the OS)
        cursor_color = (50, 255, 50) if is_pinching_now else (255, 255, 255)
        pygame.draw.circle(screen, cursor_color, mouse_pos, 10, 3)

        # Draw test info
        info_text = f"Carbon Velocity: {carbon_cycle:.2f} | Detections: {num_dets}"
        info_surface = font.render(info_text, True, (255, 255, 255))
        screen.blit(info_surface, (20, 20))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("GUI test harness shut down.")

if __name__ == "__main__":
    main()
