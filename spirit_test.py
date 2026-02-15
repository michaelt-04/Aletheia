# spirit_test.py
# GUI Test Harness for Aletheia OS Components
#
# Usage:
# python spirit_test.py

import pygame
import threading
import time
import math
import random

from aletheia_gui import (
    SpiritCompanion,
    DetectionOverlay,
    HealthBar,
    MissionTracker,
    CarbonSavingsWidget,
)

# --- Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
BLACK = (0, 0, 0)
VERSION = "GUI Test Harness v1.4 (Calm -> Angry -> Pristine -> Calm)"

# --- Mock Shared State & Lock ---
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

    # Carbon savings tracker state
    "carbon_saved_g": 0.0,
    "last_savings_event": "",
    "last_savings_event_time": 0.0,  # ✅ reliable pristine trigger

    "missions_completed": 2,
    "missions_total": 5,

    # Optional future hook (won’t hurt if GUI ignores it)
    "energy_waste_count": 0,
}
state_lock = threading.Lock()


def main():
    pygame.init()

    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(VERSION)

    spirit_companion = SpiritCompanion(shared_state, state_lock)
    all_sprites = pygame.sprite.Group(spirit_companion)

    detection_overlay = DetectionOverlay(shared_state, state_lock)
    health_bar = HealthBar(shared_state, state_lock)
    mission_tracker = MissionTracker(shared_state, state_lock)
    carbon_widget = CarbonSavingsWidget(shared_state, state_lock)

    font = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()

    running = True
    start_time = time.time()

    # Mock detections for overlay visuals (optional)
    mock_detections = [
        {"label": "laptop", "confidence": 0.88, "carbon_impact": "high"},
        {"label": "bottle", "confidence": 0.75, "carbon_impact": "medium"},
        {"label": "potted plant", "confidence": 0.92, "carbon_impact": "low"},
        {"label": "cell phone", "confidence": 0.65, "carbon_impact": "high"},
        {"label": "book", "confidence": 0.81, "carbon_impact": "unknown"},
    ]

    # Ensures we trigger pristine only once per cycle
    pristine_fired = False

    while running:
        # --- Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        elapsed = time.time() - start_time

        # --- State order: calm -> angry -> pristine -> calm (repeat) ---
        # calm:   0-5s
        # angry:  5-9s
        # pristine trigger moment: at 9s (once)
        # resolved/calm after: 9-14s (pristine animation runs in GUI, then eases back)
        cycle_len = 14.0
        t = elapsed % cycle_len

        if t < 5.0:
            phase = "calm"
            pristine_fired = False
            carbon_v = 0.10
            waste_count = 0

        elif t < 9.0:
            phase = "angry"
            carbon_v = 0.85          # > 0.3 so GUI shows angry :contentReference[oaicite:1]{index=1}
            waste_count = 3

        else:
            phase = "resolved->pristine"
            carbon_v = 0.05
            waste_count = 0

        # Other simulated values
        health_cycle = (math.cos(elapsed * 0.5) + 1) / 2 * 100
        xp_cycle = int((elapsed * 5) % 100)
        mouse_pos = pygame.mouse.get_pos()
        is_pinching_now = pygame.mouse.get_pressed()[0]

        # Mock detection list (for overlay only)
        num_dets = int(((math.sin(elapsed * 0.8) + 1) / 2) * (len(mock_detections) + 1))
        det_list = mock_detections[:num_dets]

        with state_lock:
            # Inputs
            shared_state["carbon_velocity"] = carbon_v
            shared_state["energy_waste_count"] = waste_count

            # Fire quest completion ONCE at start of resolved phase
            if phase == "resolved->pristine" and not pristine_fired:
                pristine_fired = True
                saved = random.randint(60, 260)
                shared_state["carbon_saved_g"] += saved
                shared_state["last_savings_event"] = f"Saved {saved}g from eco swap"
                shared_state["last_savings_event_time"] = time.time()  # ✅ key: timestamp trigger
                shared_state["missions_completed"] += 1

            # Other state
            shared_state["health"] = health_cycle
            shared_state["experience"] = xp_cycle
            shared_state["index_finger_tip"] = mouse_pos
            shared_state["is_pinching"] = is_pinching_now
            shared_state["detected_objects"] = det_list
            shared_state["detection_count"] = len(det_list)

        # --- Update and Draw ---
        all_sprites.update()

        screen.fill(BLACK)

        all_sprites.draw(screen)
        carbon_widget.draw(screen)
        detection_overlay.draw(screen)
        health_bar.draw(screen)
        mission_tracker.draw(screen)

        # Cursor
        cursor_color = (50, 255, 50) if is_pinching_now else (255, 255, 255)
        pygame.draw.circle(screen, cursor_color, mouse_pos, 10, 3)

        # Debug info
        info_text = f"Phase(test): {phase} | carbon_velocity={carbon_v:.2f} | waste_count={waste_count}"
        screen.blit(font.render(info_text, True, (255, 255, 255)), (20, 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    print("GUI test harness shut down.")


if __name__ == "__main__":
    main()
