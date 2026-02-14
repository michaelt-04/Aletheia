import pygame
import math
import threading
import time
import random

# --- Global Configuration (needed for GUI elements) ---
# These should ideally be passed in or derived from a config object
# For now, replicate to avoid circular imports / over-complication
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
BLACK = (0, 0, 0)

# Shared state and lock are passed to GUI elements for drawing logic
# but not defined here to avoid circular imports.


class OrbitParticle:
    """Small circles that orbit around the spirit core and fade out."""
    def __init__(self, angle, radius, ang_speed, size, life, color):
        self.angle = angle
        self.radius = radius
        self.ang_speed = ang_speed
        self.size = size
        self.life = life
        self.color = color  # pygame.Vector3 for smooth blending

    def update(self):
        self.angle += self.ang_speed
        self.life -= 0.015  # fade rate

class Particle:
    """Sparkle particles emitted from the spirit core."""
    def __init__(self, x, y, color):
        self.x, self.y = float(x), float(y)
        self.color = color
        self.size = random.randint(2, 5)
        self.life = 1.0
        self.decay = random.uniform(0.02, 0.05)
        self.vel_x = random.uniform(-1.5, 1.5)
        self.vel_y = random.uniform(-1.5, 1.5)

    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y
        self.life -= self.decay


class SpiritCompanion(pygame.sprite.Sprite):
    def __init__(self, shared_state, state_lock):
        super().__init__()
        self.image = pygame.Surface((600, 600), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=(300, 540))  # SCREEN_HEIGHT // 2

        self.shared_state = shared_state
        self.state_lock = state_lock

        self.home_pos = pygame.Vector2(300, 540)
        self.pos = self.home_pos.copy()

        self.hover_angle = 0
        self.wing_angle = 0

        # Particles
        self.particles = []
        self.orbit_particles = []
        self._last_orbit_spawn = 0.0

        # --- State management ---
        self.current_state = "calm"
        self.state_timer = 0.0
        self.elapsed_in_state = 0.0
        self.celebration_phase = "idle"
        self.celebration_progress = 0.0
        self.circle_center = self.home_pos.copy()
        self.current_color = pygame.Vector3(80, 255, 150)  # starting calm color

        # Angry star logic
        self.star_angle = 0

        #Transition Flag
        self.transitioning_to_pristine = False
        self.transition_start_pos = self.pos.copy()
        self.transition_progress = 0.0


    def draw_ethereal_wing(self, surf, center, angle_offset, width, height, color, is_left=True):
        """Draws soft light-based wings."""
        for i in range(5, 0, -1):
            w, h = width + (i * 12), height + (i * 6)
            wing_surf = pygame.Surface((w * 2, h * 2), pygame.SRCALPHA)
            alpha = 50 // i
            pygame.draw.ellipse(wing_surf, (*color, alpha), (0, 0, w, h))
            rot_angle = angle_offset + (math.sin(self.wing_angle) * 15)
            if not is_left:
                rot_angle = -rot_angle
            rotated = pygame.transform.rotate(wing_surf, rot_angle)
            surf.blit(rotated, rotated.get_rect(center=center), special_flags=pygame.BLEND_ADD)

    def update(self):
        self.image.fill((0, 0, 0, 0))
        now = pygame.time.get_ticks() / 1000.0

        # --- State Durations --- TODO, TRIGGER ON SPECIFIC EVENTS
        durations = {"calm": 4.0, "angry": 1.2, "pristine": 1.5}
        elapsed = now - self.state_timer

        # --- State Transitions ---
        if elapsed >= durations[self.current_state]:
            if self.current_state == "calm":
                self.current_state = "angry"
            elif self.current_state == "angry":
                self.current_state = "pristine"
                self.state_timer = now
                # Start smooth transition
                self.transitioning_to_pristine = True
                self.transition_progress = 0.0
                self.transition_start_pos = self.pos.copy()
                self.circle_center = self.home_pos.copy()
                self.celebration_phase = "circle"
                self.celebration_progress = 0.0
            elif self.current_state == "pristine":
                self.current_state = "calm"
                self.celebration_phase = "idle"
            self.state_timer = now
            self.elapsed_in_state = 0.0
        else:
            self.elapsed_in_state = elapsed

        # --- Target Colors ---
        target_colors = {
            "calm": pygame.Vector3(80, 255, 150),
            "angry": pygame.Vector3(255, 40, 40),
            "pristine": pygame.Vector3(180, 255, 200),
        }

        # --- Smooth color transition ---
        lerp_speed = 3.0  # adjust for faster/slower blending
        target_color = target_colors[self.current_state]
        self.current_color += (target_color - self.current_color) * (lerp_speed * (1/60))  # assuming 60 FPS
        color = (int(self.current_color.x), int(self.current_color.y), int(self.current_color.z))

        # --- Wing speed & breath ---
        if self.current_state == "calm":
            wing_speed = 0.18
            breath = 1.0 + math.sin(now * math.tau * 1.6) * 0.045
        elif self.current_state == "angry":
            wing_speed = 1.2
            breath = 1.0
        else:  # pristine
            wing_speed = 0.08
            breath = 1.0 + math.sin(now * math.tau * 1.2) * 0.06

        self.wing_angle += wing_speed
        self.hover_angle += 0.08

        # --- Movement ---
        if self.current_state == "angry":
            jitter = pygame.Vector2(random.randint(-8, 8), random.randint(-8, 8))
            self.pos = self.home_pos + jitter
        elif self.current_state == "pristine":
            circle_radius = 80
            circle_speed = 0.025

            if self.transitioning_to_pristine:
                # Smoothly move from last angry position to circle center
                self.transition_progress += 0.02  # adjust speed here (smaller = slower)
                t = min(self.transition_progress, 1.0)
                target_pos = self.circle_center + pygame.Vector2(circle_radius, 0)  # start at right side of circle
                self.pos = self.transition_start_pos.lerp(target_pos, t)
                if t >= 1.0:
                    self.transitioning_to_pristine = False
            else:
                # Circle motion after transition
                self.celebration_progress += circle_speed
                angle = self.celebration_progress * 2 * math.pi
                self.pos = self.circle_center + pygame.Vector2(math.cos(angle) * circle_radius,
                                                            math.sin(angle) * circle_radius)

        else:  # calm
            drift = pygame.Vector2(math.sin(now * 1.2) * 30, math.cos(now * 0.8) * 25)
            self.pos = self.pos.lerp(self.home_pos + drift, 0.08)

        self.rect.center = (int(self.pos.x), int(self.pos.y))

        # --- Orbit particles ---
        if self.current_state in ("calm", "pristine"):
            if now - getattr(self, "_last_orbit_spawn", 0) > 0.06 and len(self.orbit_particles) < 100:
                self._last_orbit_spawn = now
                ang = random.uniform(0, math.tau)
                r = random.uniform(40, 120)
                speed = random.uniform(0.02, 0.06) * (1 if random.random() > 0.5 else -1)
                self.orbit_particles.append(OrbitParticle(ang, r, speed, random.randint(2, 5), 1.0, color))

        for p in self.orbit_particles:
            p.update()
        self.orbit_particles = [p for p in self.orbit_particles if p.life > 0]

        # --- Draw Wings ---
        center = (300, 300)
        for p in self.orbit_particles:
            ox = center[0] + math.cos(p.angle) * (p.radius * breath)
            oy = center[1] + math.sin(p.angle) * (p.radius * breath)
            alpha = int(180 * max(0.0, min(1.0, p.life)))
            pygame.draw.circle(self.image, (*p.color, alpha), (int(ox), int(oy)), p.size)

        self.draw_ethereal_wing(self.image, (240, 270), 30, 120, 35, color, True)
        self.draw_ethereal_wing(self.image, (360, 270), 30, 120, 35, color, False)
        self.draw_ethereal_wing(self.image, (250, 310), -20, 90, 25, color, True)
        self.draw_ethereal_wing(self.image, (350, 310), -20, 90, 25, color, False)

        # --- Draw Core ---
        for i in range(6, 0, -1):
            r = int((10 + (i * 9)) * breath)
            pygame.draw.circle(self.image, (*color, 170 // i), center, r)
        pygame.draw.circle(self.image, (255, 255, 255), center, int(14 * breath))
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
            for det in detections[:8]:
                color = self.impact_colors.get(det.get("carbon_impact", "unknown"), (180, 180, 180))

                # Impact dot
                pygame.draw.circle(screen, color, (40, y + 8), 5)

                # Label + confidence (defensive defaults)
                label = det.get("label", "?")
                conf = float(det.get("confidence", 0.0))
                text = f"{label} ({conf:.0%})"
                text_surf = self.small_font.render(text, True, (255, 255, 255))
                screen.blit(text_surf, (52, y))

                # Carbon impact tag
                impact_text = det.get("carbon_impact", "?")
                impact_surf = self.small_font.render(impact_text, True, color)
                screen.blit(impact_surf, (260, y))

                y += 30

            if len(detections) > 8:
                more = self.small_font.render(f"+{len(detections) - 8} more...", True, (150, 150, 150))
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
        self.fill_color = (0, 255, 0)  # Green for health
        self.bg_color = (50, 50, 50)

    def draw(self, screen):
        with self.state_lock:
            current_health = float(self.shared_state.get("health", 0))

        # Lock to bottom-right
        x = SCREEN_WIDTH - self.width - self.padding
        y = SCREEN_HEIGHT - self.height - self.padding

        # Draw health bar
        pygame.draw.rect(screen, self.bg_color, (x, y, self.width, self.height))
        fill_width = (max(0.0, min(current_health, 100.0)) / 100.0) * self.width
        pygame.draw.rect(screen, self.fill_color, (x, y, fill_width, self.height))
        pygame.draw.rect(screen, self.border_color, (x, y, self.width, self.height), 2)

        # Draw health text
        health_text = self.font.render(f"Health: {int(current_health)}%", True, (255, 255, 255))
        text_rect = health_text.get_rect(center=(x + self.width / 2, y + self.height / 2))
        screen.blit(health_text, text_rect)


class ExperienceBar:
    """
    Displays the user's experience points stacked **above** the health bar.
    """
    def __init__(self, shared_state, state_lock):
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.font = pygame.font.Font(None, 24)
        self.width = 200
        self.height = 15
        self.padding = 10
        self.offset_y = 6  # spacing above health bar
        self.border_color = (255, 255, 255)
        self.fill_color = (0, 150, 255)
        self.bg_color = (50, 50, 50)
        self.health_bar_height = 20  # should match HealthBar height

    def draw(self, screen):
        with self.state_lock:
            current_experience = float(self.shared_state.get("experience", 0))

        x = SCREEN_WIDTH - self.width - self.padding
        # Stack **above the health bar** in bottom-right
        y = SCREEN_HEIGHT - self.health_bar_height - self.offset_y - self.height - self.padding

        # Draw XP bar
        pygame.draw.rect(screen, self.bg_color, (x, y, self.width, self.height))
        fill_width = (max(0.0, min(current_experience, 100.0)) / 100.0) * self.width
        pygame.draw.rect(screen, self.fill_color, (x, y, fill_width, self.height))
        pygame.draw.rect(screen, self.border_color, (x, y, self.width, self.height), 1)

        # Draw XP text
        xp_text = self.font.render(f"XP: {int(current_experience)}", True, (255, 255, 255))
        text_rect = xp_text.get_rect(center=(x + self.width / 2, y + self.height / 2))
        screen.blit(xp_text, text_rect)

