import pygame
import math
import threading

# --- Global Configuration (needed for GUI elements) ---
# These should ideally be passed in or derived from a config object
# For now, replicate to avoid circular imports / over-complication
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
BLACK = (0, 0, 0)

# Shared state and lock are passed to GUI elements for drawing logic
# but not defined here to avoid circular imports.

class EcoSprite(pygame.sprite.Sprite):
    """
    A floating entity whose appearance and behavior are tied to environmental data.
    """
    def __init__(self, shared_state, state_lock):
        super().__init__()
        self.image = pygame.Surface((50, 50), pygame.SRCALPHA)
        pygame.draw.circle(self.image, (0, 255, 150), (25, 25), 25)
        self.rect = self.image.get_rect(center=(SCREEN_WIDTH - 150, 150))
        
        self.shared_state = shared_state
        self.state_lock = state_lock

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
        with self.state_lock:
            carbon_v = self.shared_state["carbon_velocity"]

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
    def __init__(self, shared_state, state_lock):
        self.surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.current_alpha = 0
        self.shared_state = shared_state
        self.state_lock = state_lock

    def draw(self, screen):
        with self.state_lock:
            carbon_v = self.shared_state["carbon_velocity"]

        target_alpha = int(carbon_v * 200)
        self.current_alpha = self.current_alpha * 0.95 + target_alpha * 0.05

        self.surface.fill((50, 50, 55, int(self.current_alpha)))
        screen.blit(self.surface, (0, 0))


class DetectionOverlay:
    """
    Draws detected objects and their carbon impact labels on the HUD.
    Shows what the YOLO model is currently seeing.
    """
    def __init__(self, shared_state, state_lock):
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.impact_colors = {
            "high": (255, 50, 50),       # Red
            "medium": (255, 180, 0),     # Orange
            "low": (0, 220, 100),        # Green
            "unknown": (180, 180, 180),  # Grey
        }
        self.shared_state = shared_state
        self.state_lock = state_lock

    def draw(self, screen):
        with self.state_lock:
            detections = self.shared_state["detected_objects"]

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

class HealthBar:
    """
    Displays the user's health in the bottom right corner.
    """
    def __init__(self, shared_state, state_lock):
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.font = pygame.font.Font(None, 24)
        self.width = 200
        self.height = 20
        self.padding = 10
        self.border_color = (255, 255, 255)
        self.fill_color = (0, 255, 0) # Green for health
        self.bg_color = (50, 50, 50)

    def draw(self, screen):
        with self.state_lock:
            current_health = self.shared_state["health"]

        # Calculate bar position
        x = SCREEN_WIDTH - self.width - self.padding
        y = SCREEN_HEIGHT - self.height - self.padding

        # Draw background
        pygame.draw.rect(screen, self.bg_color, (x, y, self.width, self.height))

        # Draw fill
        fill_width = (current_health / 100) * self.width
        pygame.draw.rect(screen, self.fill_color, (x, y, fill_width, self.height))

        # Draw border
        pygame.draw.rect(screen, self.border_color, (x, y, self.width, self.height), 2)

        # Draw text
        health_text = self.font.render(f"Health: {current_health}%", True, (255, 255, 255))
        text_rect = health_text.get_rect(center=(x + self.width / 2, y + self.height / 2))
        screen.blit(health_text, text_rect)

class ExperienceBar:
    """
    Displays the user's experience points below the health bar.
    """
    def __init__(self, shared_state, state_lock):
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.font = pygame.font.Font(None, 24)
        self.width = 200
        self.height = 15
        self.padding = 10
        self.border_color = (255, 255, 255)
        self.fill_color = (0, 150, 255) # Blue for experience
        self.bg_color = (50, 50, 50)
        self.offset_y = 5 # Offset from health bar

    def draw(self, screen):
        with self.state_lock:
            current_experience = self.shared_state["experience"]

        # Calculate bar position (below health bar)
        health_bar_y = SCREEN_HEIGHT - 20 - self.padding # Assuming health bar height is 20
        x = SCREEN_WIDTH - self.width - self.padding
        y = health_bar_y + self.height + self.offset_y # Position below health bar

        # Draw background
        pygame.draw.rect(screen, self.bg_color, (x, y, self.width, self.height))

        # Draw fill (assuming max experience for a level is 100 for now)
        # TODO: Implement proper level/max_xp logic
        fill_width = (min(current_experience, 100) / 100) * self.width
        pygame.draw.rect(screen, self.fill_color, (x, y, fill_width, self.height))

        # Draw border
        pygame.draw.rect(screen, self.border_color, (x, y, self.width, self.height), 1)

        # Draw text
        exp_text = self.font.render(f"XP: {current_experience}", True, (255, 255, 255))
        text_rect = exp_text.get_rect(center=(x + self.width / 2, y + self.height / 2))
        screen.blit(exp_text, text_rect)

