import pygame
import math
import random
from typing import List, Tuple

# --- Global Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
BLACK = (0, 0, 0)

# Pre-calculate common math constants
TWO_PI = 2 * math.pi
TAU = math.tau
PI_HALF = math.pi / 2


class OrbitParticle:
    """Small circles that orbit around the spirit core and fade out."""
    __slots__ = ('angle', 'radius', 'ang_speed', 'size', 'life', 'color')
    
    def __init__(self, angle, radius, ang_speed, size, life, color):
        self.angle = angle
        self.radius = radius
        self.ang_speed = ang_speed
        self.size = size
        self.life = life
        self.color = color

    def update(self):
        self.angle += self.ang_speed
        self.life -= 0.015


class Particle:
    """Sparkle particles emitted from the spirit core."""
    __slots__ = ('x', 'y', 'color', 'size', 'life', 'decay', 'vel_x', 'vel_y')
    
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
    # Class-level cache for wing surfaces (shared across instances if needed)
    _wing_cache_global = {}
    
    def __init__(self, shared_state, state_lock):
        super().__init__()
        self.image = pygame.Surface((600, 600), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=(300, 540))

        self.shared_state = shared_state
        self.state_lock = state_lock

        self.home_pos = pygame.Vector2(300, 540)
        self.pos = self.home_pos.copy()

        self.hover_angle = 0.0
        self.wing_angle = 0.0

        # Particles - preallocate lists with initial capacity hint
        self.particles: List[Particle] = []
        self.orbit_particles: List[OrbitParticle] = []
        self._last_orbit_spawn = 0.0

        # State management
        self.current_state = "calm"
        self.state_timer = 0.0
        self.elapsed_in_state = 0.0
        self.celebration_phase = "idle"
        self.celebration_progress = 0.0
        self.circle_center = self.home_pos.copy()
        self.current_color = pygame.Vector3(80, 255, 150)

        # Angry star
        self.star_angle = 0

        # Transition
        self.transitioning_to_pristine = False
        self.transition_start_pos = self.pos.copy()
        self.transition_progress = 0.0
        
        # Constants
        self.TWO_PI = TWO_PI
        self.TAU = TAU
        
        # State durations (avoid dict lookup every frame)
        self.duration_calm = 4.0
        self.duration_angry = 1.2
        self.duration_pristine = 1.5
        
        # Target colors as Vector3 for faster lerping
        self.color_calm = pygame.Vector3(80, 255, 150)
        self.color_angry = pygame.Vector3(255, 40, 40)
        self.color_pristine = pygame.Vector3(180, 255, 200)
        
        # Pre-create core circle radii for breath effect (6 layers)
        self.core_base_radii = [10 + (i * 9) for i in range(6, 0, -1)]
        self.core_alphas = [170 // i for i in range(6, 0, -1)]
        
        # Center position (constant)
        self.center = (300, 300)

    def _get_state_duration(self):
        """Inline duration lookup - faster than dict"""
        if self.current_state == "calm":
            return self.duration_calm
        elif self.current_state == "angry":
            return self.duration_angry
        else:
            return self.duration_pristine

    def draw_ethereal_wing(self, surf, center, angle_offset, width, height, color, is_left=True):
        """Optimized wing drawing with global caching."""
        cache_key = (width, height, is_left)
        
        if cache_key not in SpiritCompanion._wing_cache_global:
            wing_layers = []
            for i in range(5, 0, -1):
                w, h = width + (i * 12), height + (i * 6)
                wing_surf = pygame.Surface((w * 2, h * 2), pygame.SRCALPHA)
                wing_layers.append((wing_surf, w, h, 50 // i))
            SpiritCompanion._wing_cache_global[cache_key] = wing_layers
        
        wing_layers = SpiritCompanion._wing_cache_global[cache_key]
        
        # Pre-calculate rotation angle once
        rot_angle = angle_offset + (math.sin(self.wing_angle) * 15)
        if not is_left:
            rot_angle = -rot_angle
        
        for wing_surf, w, h, alpha in wing_layers:
            wing_surf.fill((0, 0, 0, 0))
            pygame.draw.ellipse(wing_surf, (*color, alpha), (0, 0, w, h))
            rotated = pygame.transform.rotate(wing_surf, rot_angle)
            surf.blit(rotated, rotated.get_rect(center=center), special_flags=pygame.BLEND_ADD)

    def update(self):
        self.image.fill((0, 0, 0, 0))
        now = pygame.time.get_ticks() * 0.001  # Multiply is faster than divide

        # State transitions
        elapsed = now - self.state_timer
        if elapsed >= self._get_state_duration():
            if self.current_state == "calm":
                self.current_state = "angry"
                self.state_timer = now
            elif self.current_state == "angry":
                self.current_state = "pristine"
                self.state_timer = now
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

        # Smooth color transition (use Vector3 directly)
        lerp_speed = 0.05
        if self.current_state == "calm":
            target = self.color_calm
        elif self.current_state == "angry":
            target = self.color_angry
        else:
            target = self.color_pristine
        
        # Direct vector lerp is faster
        self.current_color.x += (target.x - self.current_color.x) * lerp_speed
        self.current_color.y += (target.y - self.current_color.y) * lerp_speed
        self.current_color.z += (target.z - self.current_color.z) * lerp_speed
        color = (int(self.current_color.x), int(self.current_color.y), int(self.current_color.z))

        # Wing speed & breath (avoid repeated dict lookups)
        if self.current_state == "calm":
            wing_speed = 0.18
            breath = 1.0 + math.sin(now * TAU * 1.6) * 0.045
        elif self.current_state == "angry":
            wing_speed = 1.2
            breath = 1.0
        else:  # pristine
            wing_speed = 0.08
            breath = 1.0 + math.sin(now * TAU * 1.2) * 0.06

        self.wing_angle += wing_speed
        self.hover_angle += 0.08

        # Movement
        if self.current_state == "angry":
            # Use randint less frequently if possible, or cache jitter
            jitter_x = random.randint(-8, 8)
            jitter_y = random.randint(-8, 8)
            self.pos.x = self.home_pos.x + jitter_x
            self.pos.y = self.home_pos.y + jitter_y
            
        elif self.current_state == "pristine":
            circle_radius = 80
            circle_speed = 0.025

            if self.transitioning_to_pristine:
                self.transition_progress += 0.02
                if self.transition_progress >= 1.0:
                    self.transition_progress = 1.0
                    self.transitioning_to_pristine = False
                    
                # Manual lerp is faster than .lerp()
                t = self.transition_progress
                target_x = self.circle_center.x + circle_radius
                target_y = self.circle_center.y
                self.pos.x = self.transition_start_pos.x + (target_x - self.transition_start_pos.x) * t
                self.pos.y = self.transition_start_pos.y + (target_y - self.transition_start_pos.y) * t
            else:
                self.celebration_progress += circle_speed
                angle = self.celebration_progress * TWO_PI
                # Direct calculation instead of Vector2 addition
                self.pos.x = self.circle_center.x + math.cos(angle) * circle_radius
                self.pos.y = self.circle_center.y + math.sin(angle) * circle_radius
                
        else:  # calm
            drift_x = math.sin(now * 1.2) * 30
            drift_y = math.cos(now * 0.8) * 25
            target_x = self.home_pos.x + drift_x
            target_y = self.home_pos.y + drift_y
            self.pos.x += (target_x - self.pos.x) * 0.08
            self.pos.y += (target_y - self.pos.y) * 0.08

        self.rect.center = (int(self.pos.x), int(self.pos.y))

        # Orbit particles (batch creation, reduced frequency check)
        if self.current_state in ("calm", "pristine"):
            time_since_spawn = now - self._last_orbit_spawn
            if time_since_spawn > 0.06 and len(self.orbit_particles) < 100:
                self._last_orbit_spawn = now
                # Create 2 at once to reduce overhead
                for _ in range(2):
                    self.orbit_particles.append(OrbitParticle(
                        random.uniform(0, TAU),
                        random.uniform(40, 120),
                        random.uniform(0.02, 0.06) * random.choice((-1, 1)),
                        random.randint(2, 5),
                        1.0,
                        color
                    ))

        # Update particles in-place
        for p in self.orbit_particles:
            p.update()
        
        # Filter dead particles (list comprehension is faster than loop)
        self.orbit_particles = [p for p in self.orbit_particles if p.life > 0]

        # Draw orbit particles (avoid tuple unpacking in loop)
        center_x, center_y = self.center
        cos_fn = math.cos
        sin_fn = math.sin
        draw_circle = pygame.draw.circle
        
        for p in self.orbit_particles:
            ox = int(center_x + cos_fn(p.angle) * (p.radius * breath))
            oy = int(center_y + sin_fn(p.angle) * (p.radius * breath))
            alpha = int(180 * min(1.0, max(0.0, p.life)))
            draw_circle(self.image, (*p.color, alpha), (ox, oy), p.size)

        # Draw wings
        self.draw_ethereal_wing(self.image, (240, 270), 30, 120, 35, color, True)
        self.draw_ethereal_wing(self.image, (360, 270), 30, 120, 35, color, False)
        self.draw_ethereal_wing(self.image, (250, 310), -20, 90, 25, color, True)
        self.draw_ethereal_wing(self.image, (350, 310), -20, 90, 25, color, False)

        # Draw core (use pre-calculated values)
        for i, (base_r, alpha) in enumerate(zip(self.core_base_radii, self.core_alphas)):
            r = int(base_r * breath)
            draw_circle(self.image, (*color, alpha), self.center, r)
        draw_circle(self.image, (255, 255, 255), self.center, int(14 * breath))


class GreyFog:
    """Optimized fog overlay."""
    __slots__ = ('surface', 'current_alpha', 'shared_state', 'state_lock')
    
    def __init__(self, shared_state, state_lock):
        self.surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.current_alpha = 0.0
        self.shared_state = shared_state
        self.state_lock = state_lock

    def draw(self, screen):
        with self.state_lock:
            carbon_v = self.shared_state["carbon_velocity"]

        target_alpha = carbon_v * 200  # Avoid int() until final use
        self.current_alpha += (target_alpha - self.current_alpha) * 0.05

        self.surface.fill((50, 50, 55, int(self.current_alpha)))
        screen.blit(self.surface, (0, 0))


class DetectionOverlay:
    """Optimized detection overlay."""
    __slots__ = ('font', 'small_font', 'impact_colors', 'shared_state', 'state_lock', 'title_surface')
    
    def __init__(self, shared_state, state_lock):
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        self.impact_colors = {
            "high": (255, 50, 50),
            "medium": (255, 180, 0),
            "low": (0, 220, 100),
            "unknown": (180, 180, 180),
        }
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.title_surface = self.font.render("Detected Objects", True, (255, 255, 255))

    def draw(self, screen):
        with self.state_lock:
            detections = self.shared_state["detected_objects"]

        if not detections:
            return

        panel_h = min(len(detections), 8) * 30 + 50
        panel_surface = pygame.Surface((320, panel_h), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 140))
        screen.blit(panel_surface, (20, 70))
        screen.blit(self.title_surface, (30, 78))

        y = 108
        white = (255, 255, 255)
        
        for det in detections[:8]:
            color = self.impact_colors.get(det.get("carbon_impact", "unknown"), (180, 180, 180))
            pygame.draw.circle(screen, color, (40, y + 8), 5)

            label = det.get("label", "?")
            conf = float(det.get("confidence", 0.0))
            text_surf = self.small_font.render(f"{label} ({conf:.0%})", True, white)
            screen.blit(text_surf, (52, y))

            impact_surf = self.small_font.render(det.get("carbon_impact", "?"), True, color)
            screen.blit(impact_surf, (260, y))
            y += 30

        if len(detections) > 8:
            more = self.small_font.render(f"+{len(detections) - 8} more...", True, (150, 150, 150))
            screen.blit(more, (52, y))


class HealthBar:
    """Optimized HP bar."""
    __slots__ = ('shared_state', 'state_lock', 'width', 'height', 'font', 
                 'fill_color', 'text_color', 'bg_panel', 'x', 'y')
    
    def __init__(self, shared_state, state_lock):
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.width = 200
        self.height = 20
        self.font = pygame.font.Font(None, 20)
        self.fill_color = (140, 255, 180)
        self.text_color = (255, 255, 255)
        
        self.bg_panel = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self.bg_panel.fill((255, 255, 255, 25))
        
        self.x = SCREEN_WIDTH - self.width - 10
        self.y = SCREEN_HEIGHT - self.height - 10

    def draw(self, screen):
        with self.state_lock:
            hp = self.shared_state.get("health", 0)

        screen.blit(self.bg_panel, (self.x, self.y))
        
        fill_width = max(0.0, min(hp, 100.0)) * 0.01 * self.width  # Avoid division
        pygame.draw.rect(screen, self.fill_color, (self.x, self.y, fill_width, self.height))

        hp_text = self.font.render(f"HP {int(hp)}%", True, self.text_color)
        screen.blit(hp_text, (self.x + 5, self.y + 2))


class MissionTracker:
    """
    Displays daily carbon mission progress.
    Example: Daily Mission: 2/5 Completed
    """
    def __init__(self, shared_state, state_lock):
        self.shared_state = shared_state
        self.state_lock = state_lock
        self.padding = 10
        self.font = pygame.font.Font(None, 22)

        self.bg_color = (255, 255, 255, 25)
        self.text_color = (255, 255, 255)

    def draw(self, screen):
        with self.state_lock:
            completed = int(self.shared_state.get("missions_completed", 0))
            total = int(self.shared_state.get("missions_total", 5))

        text = f"Daily Mission: {completed}/{total} Completed"

        text_surface = self.font.render(text, True, self.text_color)

        width = text_surface.get_width() + 20
        height = text_surface.get_height() + 10

        x = SCREEN_WIDTH - width - self.padding
        y = SCREEN_HEIGHT - 60  # Slightly above health bar

        # Frosted background
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill(self.bg_color)
        screen.blit(panel, (x, y))

        screen.blit(text_surface, (x + 10, y + 5))
