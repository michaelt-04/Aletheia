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

class SpiritCompanion(pygame.sprite.Sprite):
    def __init__(self, shared_state, state_lock):
        super().__init__()
        self.image = pygame.Surface((600, 600), pygame.SRCALPHA)
        # Initial position, which will be updated by its movement logic
        self.rect = self.image.get_rect(center=(300, SCREEN_HEIGHT // 2))
        
        self.shared_state = shared_state
        self.state_lock = state_lock

        self.home_pos = pygame.Vector2(300, SCREEN_HEIGHT // 2)
        self.pos = pygame.Vector2(300, SCREEN_HEIGHT // 2)
        
        self.hover_angle = 0
        self.wing_angle = 0
        self.particles = []
        
        # Happy Arc Logic
        self.is_jumping = False
        self.jump_progress = 0
        self.jump_timer = time.time() # This needs to be imported, will add later
        
        # Angry Star Logic
        self.star_angle = 0

    def draw_ethereal_wing(self, surf, center, angle_offset, width, height, color, is_left=True):
        """Draws soft light-based wings."""
        for i in range(5, 0, -1):
            w, h = width + (i * 12), height + (i * 6)
            wing_surf = pygame.Surface((w * 2, h * 2), pygame.SRCALPHA)
            alpha = 50 // i
            pygame.draw.ellipse(wing_surf, color + (alpha,), (0, 0, w, h))
            rot_angle = angle_offset + (math.sin(self.wing_angle) * 15)
            if not is_left: rot_angle = -rot_angle
            rotated = pygame.transform.rotate(wing_surf, rot_angle)
            surf.blit(rotated, rotated.get_rect(center=center), special_flags=pygame.BLEND_ADD)

    def update(self):
        self.image.fill((0, 0, 0, 0))
        now = pygame.time.get_ticks() / 1000.0 # Use pygame.time for consistency
        
        with self.state_lock:
            carbon_v = self.shared_state["carbon_velocity"]

        # State Mapping
        if carbon_v <= 0.26:
            state, color, wing_speed = "pristine", (180, 255, 200), 0.08
        elif carbon_v <= 0.6:
            state, color, wing_speed = "calm", (80, 255, 150), 0.18
        else:
            state, color, wing_speed = "angry", (255, 40, 40), 1.2

        self.wing_angle += wing_speed
        self.hover_angle += 0.08

        # --- MOVEMENT BEHAVIORS ---
        if state == "angry":
            # VIOLENT STAR ANIMATION
            self.star_angle += 1 
            r = 0 # Radius of the star points
            points = [0, 144, 288, 72, 216]
            idx = int(self.star_angle % 5)
            target_angle = math.radians(points[idx])
            
            star_offset = pygame.Vector2(math.cos(target_angle) * r, math.sin(target_angle) * r)
            jitter = pygame.Vector2(random.randint(-5, 5), random.randint(-5, 5))
            self.pos = self.home_pos + star_offset + jitter
            self.is_jumping = False
            
        elif state == "pristine":
            # SPEEDY 180-DEGREE ARC JUMP
            if not self.is_jumping and now - self.jump_timer > random.uniform(3, 5):
                self.is_jumping = True
                self.jump_progress = 0
                self.jump_timer = now # Reset timer when jump starts
            
            if self.is_jumping:
                self.jump_progress += 0.035 
                
                angle = self.jump_progress * math.pi 
                radius = 250
                
                offset_x = math.sin(angle) * radius
                offset_y = (1 - math.cos(angle)) * radius
                
                self.pos = self.home_pos + pygame.Vector2(offset_x, -offset_y)

                if self.jump_progress >= 1.0:
                    self.is_jumping = False
                    self.jump_timer = now
            else:
                # Freedom Drift (High magnitude)
                drift = pygame.Vector2(math.sin(now * 1.5) * 60, math.cos(now * 1.2) * 45)
                self.pos += (self.home_pos + drift - self.pos) * 0.08
        
        else: # Calm state
            drift = pygame.Vector2(math.sin(now * 1.2) * 40, math.cos(now * 0.8) * 30)
            self.pos += (self.home_pos + drift - self.pos) * 0.1
            self.is_jumping = False

        self.rect.center = (int(self.pos.x), int(self.pos.y))

        # --- Render Spirit ---
        center = (300, 300)
        self.draw_ethereal_wing(self.image, (240, 270), 30, 120, 35, color, True)
        self.draw_ethereal_wing(self.image, (360, 270), 30, 120, 35, color, False)
        self.draw_ethereal_wing(self.image, (250, 310), -20, 90, 25, color, True)
        self.draw_ethereal_wing(self.image, (350, 310), -20, 90, 25, color, False)
        
        for i in range(6, 0, -1):
            pygame.draw.circle(self.image, color + (170 // i,), center, 10 + (i * 9))
        pygame.draw.circle(self.image, (255, 255, 255), center, 14) # WHITE

        # Sparkles
        if random.random() > 0.4:
            self.particles.append(Particle(self.rect.centerx, self.rect.centery, color))

class Particle: # Moved Particle class definition here, it is used by SpiritCompanion
    def __init__(self, x, y, color):
        self.x, self.y = x, y
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

