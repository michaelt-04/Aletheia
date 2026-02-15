import pygame
import math
import random
import time
from typing import List, Tuple

# --- Global Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
BLACK = (0, 0, 0)

# Pre-calculate common math constants
TWO_PI = 2 * math.pi
TAU = math.tau
PI_HALF = math.pi / 2


# =========================
# Carbon Tracker
# =========================
class CarbonSavingsWidget:
    """
    HUD widget that shows total CO2e reduced and the last savings event.
    Top-right, more transparent background, resolution-adaptive.
    """
    __slots__ = (
        "shared_state", "state_lock",
        "font_large", "font_small",
        "padding", "width", "height",
        "displayed_value", "panel_alpha"
    )

    def __init__(self, shared_state, state_lock, *, panel_alpha: int = 90):
        self.shared_state = shared_state
        self.state_lock = state_lock

        self.font_large = pygame.font.Font(None, 42)
        self.font_small = pygame.font.Font(None, 24)

        self.padding = 20
        self.width = 360
        self.height = 120

        self.displayed_value = 0.0  # smooth count-up
        self.panel_alpha = panel_alpha

    def draw(self, screen):
        with self.state_lock:
            total_saved = float(self.shared_state.get("carbon_saved_g", 0.0))
            last_event = self.shared_state.get("last_savings_event", "")

        # Smooth animated count-up
        self.displayed_value += (total_saved - self.displayed_value) * 0.08
        total_saved_kg = self.displayed_value / 1000.0

        # Resolution-adaptive positioning (TOP-RIGHT)
        screen_width, _ = screen.get_size()
        x = screen_width - self.width - self.padding
        y = self.padding

        # Background panel (more transparent)
        panel = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, self.panel_alpha))
        screen.blit(panel, (x, y))

        # Title
        title = self.font_small.render("Carbon Reduced", True, (120, 255, 160))
        screen.blit(title, (x + 15, y + 10))

        # Big number (use CO2e for glyph compatibility)
        value = self.font_large.render(f"{total_saved_kg:.2f} kg CO2e", True, (255, 255, 255))
        screen.blit(value, (x + 15, y + 40))

        # Last event line
        if last_event:
            event_text = self.font_small.render(f"+ {last_event}", True, (180, 255, 200))
            screen.blit(event_text, (x + 15, y + 85))


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
    _star_points_cache = {}  # left in place, but star drawing removed below

    def __init__(self, shared_state, state_lock):
        super().__init__()
        self.image = pygame.Surface((600, 600), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2))

        self.shared_state = shared_state
        self.state_lock = state_lock

        self.home_pos = pygame.Vector2(SCREEN_WIDTH // 10, SCREEN_HEIGHT // 2)
        self.pos = self.home_pos.copy()

        self.hover_angle = 0.0
        self.wing_angle = 0.0

        self.particles: List[Particle] = []
        self.orbit_particles: List[OrbitParticle] = []
        self._last_orbit_spawn = 0.0

        # State management (driven by carbon_velocity)
        self.current_state = "calm"  # 'calm', 'angry', 'pristine'
        self.fsm_state = "calm"  # strict FSM: calm -> angry -> pristine -> calm
        self._pending_quest_complete = False

        # Calm wander (non-circular): choose random waypoints and ease toward them
        self._calm_waypoint = self.home_pos.copy()
        self._next_waypoint_epoch = 0.0

        # Angry shake (deterministic + smooth amplitude)
        self._angry_amp = 0.0


        # Jitter frequency controls (tune these to adjust how fast angry shakes)
        self.angry_jitter_rate_x = 19.0
        self.angry_jitter_rate_y = 23.0

        # How long calm stays after pristine (seconds). Increase to make the green calm linger longer.
        self.post_pristine_calm_seconds = 2.0

        # After pristine finishes, hold calm for a short cooldown so it never snaps back to angry immediately
        self._post_pristine_cooldown_until = 0.0

        self.current_color = pygame.Vector3(80, 255, 150)

        # Angry pulse timing (kept)
        self.star_angle = 0.0
        self.star_pulse_t = 0.0

        # Pristine celebration (kept)
        self.celebration_phase = "idle"
        self.celebration_progress = 0.0
        self.circle_center = self.home_pos.copy()

        # Transition vars (these must exist)
        self.transitioning_to_pristine = False
        self.transition_start_pos = self.pos.copy()
        self.transition_progress = 0.0

        # Track last carbon-savings event we've reacted to (prevents resetting circle every frame)
        self._last_seen_savings_event_time = 0.0
        # Fallback: track last event text if timestamps aren't provided
        self._last_seen_savings_event_text = ""

        # Pristine celebration sequence state (transition in -> circle -> return home)
        self._pristine_active = False
        self._pristine_phase = "idle"  # "in", "circle", "return", "idle"
        self._return_progress = 0.0

        # Target colors
        self.color_calm = pygame.Vector3(80, 255, 150)
        self.color_angry = pygame.Vector3(255, 40, 40)
        self.color_pristine = pygame.Vector3(180, 255, 200)

        # Core rings
        self.core_base_radii = [10 + (i * 9) for i in range(6, 0, -1)]
        self.core_alphas = [170 // i for i in range(6, 0, -1)]

        # Center position relative to sprite surface
        self.center = (self.image.get_width() // 2, self.image.get_height() // 2)

    def _get_star_points(self, radius, num_points, inner_ratio, angle_offset=0):
        """Left in place for compatibility (not used once star removed)."""
        key = (radius, num_points, inner_ratio, angle_offset)
        if key in SpiritCompanion._star_points_cache:
            return SpiritCompanion._star_points_cache[key]

        points = []
        for i in range(num_points * 2):
            r = radius * (inner_ratio if i % 2 == 1 else 1)
            angle = math.pi / num_points * i + angle_offset
            x = r * math.sin(angle)
            y = r * math.cos(angle)
            points.append((x, y))
        SpiritCompanion._star_points_cache[key] = points
        return points

    def draw_ethereal_wing(self, surf, center, angle_offset, width, height, color, is_left=True):
        """Optimized wing drawing with caching."""
        cache_key = (width, height, is_left)

        if cache_key not in SpiritCompanion._wing_cache_global:
            wing_layers = []
            for i in range(5, 0, -1):
                w, h = width + (i * 12), height + (i * 6)
                wing_surf = pygame.Surface((w * 2, h * 2), pygame.SRCALPHA)
                wing_layers.append((wing_surf, w, h, 50 // i))
            SpiritCompanion._wing_cache_global[cache_key] = wing_layers

        wing_layers = SpiritCompanion._wing_cache_global[cache_key]

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
        now = pygame.time.get_ticks() * 0.001
        epoch_now = time.time()

        # --- State determination (STRICT ORDER) ---
        # Desired order:
        #   calm (default) -> angry (when waste detected) -> pristine (only on quest complete)
        #   pristine ALWAYS returns to calm (never directly to angry)
        with self.state_lock:
            carbon_saved_event_time = float(self.shared_state.get("last_savings_event_time", 0.0))
            carbon_saved_event_text = str(self.shared_state.get("last_savings_event", ""))
            energy_waste_count = int(self.shared_state.get("energy_waste_count", 0))
            detections = self.shared_state.get("detected_objects", [])

        # Waste detected signal (future-proof):
        waste_present = energy_waste_count > 0 or any(d.get("carbon_impact") == "high" for d in detections)

        # Detect a NEW quest completion / savings event (timestamp preferred; text fallback)
        quest_complete = False
        if carbon_saved_event_time > 0.0:
            quest_complete = carbon_saved_event_time > self._last_seen_savings_event_time
        elif carbon_saved_event_text:
            quest_complete = carbon_saved_event_text != self._last_seen_savings_event_text

        if quest_complete:
            if carbon_saved_event_time > 0.0:
                self._last_seen_savings_event_time = carbon_saved_event_time
            if carbon_saved_event_text:
                self._last_seen_savings_event_text = carbon_saved_event_text

        # --- STRICT FSM transitions ---
        if self.fsm_state == "calm":
            # calm can ONLY go to angry (but never immediately after pristine)
            if waste_present and epoch_now >= self._post_pristine_cooldown_until:
                self.fsm_state = "angry"

        elif self.fsm_state == "angry":
            # angry can ONLY go to pristine
            if quest_complete:
                # Start pristine celebration exactly once per quest completion
                self._pristine_active = True
                self._pristine_phase = "in"
                self.celebration_progress = 0.0
                self.circle_center = self.home_pos.copy()

                self.transitioning_to_pristine = True
                self.transition_progress = 0.0
                self.transition_start_pos = self.pos.copy()

                self.fsm_state = "pristine"

        elif self.fsm_state == "pristine":
            # locked until pristine sequence finishes (movement section will set _pristine_active False)
            if not self._pristine_active:
                # pristine ALWAYS returns to calm, never directly to angry
                self.fsm_state = "calm"
                # hold calm briefly so it doesn't snap back to angry in the same moment
                self._post_pristine_cooldown_until = epoch_now + self.post_pristine_calm_seconds

        # Map FSM to render state
        self.current_state = self.fsm_state

        # --- Smooth color transition ---
        lerp_speed = 0.05
        if self.current_state == "calm":
            target = self.color_calm
        elif self.current_state == "angry":
            target = self.color_angry
        else:
            target = self.color_pristine

        self.current_color.x += (target.x - self.current_color.x) * lerp_speed
        self.current_color.y += (target.y - self.current_color.y) * lerp_speed
        self.current_color.z += (target.z - self.current_color.z) * lerp_speed
        color = (int(self.current_color.x), int(self.current_color.y), int(self.current_color.z))

        # --- Wing speed, breath & movement ---
        draw_wings = True  # ensure defined for all states

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

        # --- Movement ---
        if self.current_state == "angry":
            # Deterministic, strong shake with smooth amplitude (consistent feel)
            target_amp = 22.0
            self._angry_amp += (target_amp - self._angry_amp) * 0.10
            self.pos.x = self.home_pos.x + math.sin(now * self.angry_jitter_rate_x) * self._angry_amp
            self.pos.y = self.home_pos.y + math.cos(now * self.angry_jitter_rate_y) * self._angry_amp

        elif self.current_state == "pristine":
            # Celebration: transition to circle, do ONE full loop, then return home.
            circle_radius = 80
            circle_speed = 0.018  # ~1 loop in ~0.9s at 60fps

            if self._pristine_phase == "in":
                # Lerp into the circle start point (right side of the circle)
                self.transition_progress += 0.02
                if self.transition_progress >= 1.0:
                    self.transition_progress = 1.0
                    self.transitioning_to_pristine = False
                    self._pristine_phase = "circle"

                t = self.transition_progress
                target_x = self.circle_center.x + circle_radius
                target_y = self.circle_center.y
                self.pos.x = self.transition_start_pos.x + (target_x - self.transition_start_pos.x) * t
                self.pos.y = self.transition_start_pos.y + (target_y - self.transition_start_pos.y) * t

            elif self._pristine_phase == "circle":
                # Circle once; when completed, begin returning home
                self.celebration_progress += circle_speed
                if self.celebration_progress >= 1.0:
                    self.celebration_progress = 1.0
                    self._pristine_phase = "return"
                    self._return_progress = 0.0
                    self._return_start_pos = self.pos.copy()

                angle = self.celebration_progress * TWO_PI
                self.pos.x = self.circle_center.x + math.cos(angle) * circle_radius
                self.pos.y = self.circle_center.y + math.sin(angle) * circle_radius

            elif self._pristine_phase == "return":
                # Lerp from current position back to home
                self._return_progress += 0.03
                if self._return_progress >= 1.0:
                    self._return_progress = 1.0
                    self._pristine_phase = "idle"
                    self._pristine_active = False  # celebration complete

                t = self._return_progress
                start = getattr(self, "_return_start_pos", self.pos)
                self.pos.x = start.x + (self.home_pos.x - start.x) * t
                self.pos.y = start.y + (self.home_pos.y - start.y) * t

            else:
                # Safety fallback: end pristine if phase is unknown
                self._pristine_active = False
                self._pristine_phase = "idle"

        else:  # calm
            # Non-circular calm wander: pick a waypoint occasionally and ease toward it
            if epoch_now >= self._next_waypoint_epoch:
                self._next_waypoint_epoch = epoch_now + random.uniform(2.0, 4.0)
                self._calm_waypoint = self.home_pos + pygame.Vector2(
                    random.uniform(-45, 45),
                    random.uniform(-30, 30),
                )
            self.pos.x += (self._calm_waypoint.x - self.pos.x) * 0.03
            self.pos.y += (self._calm_waypoint.y - self.pos.y) * 0.03

        self.rect.center = (int(self.pos.x), int(self.pos.y))

        # --- Orbit particles ---
        if self.current_state in ("calm", "pristine"):
            time_since_spawn = now - self._last_orbit_spawn
            if time_since_spawn > 0.06 and len(self.orbit_particles) < 100:
                self._last_orbit_spawn = now
                for _ in range(2):
                    self.orbit_particles.append(OrbitParticle(
                        random.uniform(0, TAU),
                        random.uniform(40, 120),
                        random.uniform(0.02, 0.06) * random.choice((-1, 1)),
                        random.randint(2, 5),
                        1.0,
                        color
                    ))

        for p in self.orbit_particles:
            p.update()
        self.orbit_particles = [p for p in self.orbit_particles if p.life > 0]

        # --- Drawing ---
        center_x, center_y = self.center
        draw_circle = pygame.draw.circle

        # Draw orbit particles
        for p in self.orbit_particles:
            ox = int(center_x + math.cos(p.angle) * (p.radius * breath))
            oy = int(center_y + math.sin(p.angle) * (p.radius * breath))
            alpha = int(180 * min(1.0, max(0.0, p.life)))
            draw_circle(self.image, (*p.color, alpha), (ox, oy), p.size)

        if draw_wings:
            self.draw_ethereal_wing(self.image, (center_x - 60, center_y - 30), 30, 120, 35, color, True)
            self.draw_ethereal_wing(self.image, (center_x + 60, center_y - 30), 30, 120, 35, color, False)
            self.draw_ethereal_wing(self.image, (center_x - 50, center_y + 10), -20, 90, 25, color, True)
            self.draw_ethereal_wing(self.image, (center_x + 50, center_y + 10), -20, 90, 25, color, False)

        # ✅ STAR REMOVED: Draw core for ALL states (including angry)
        for base_r, alpha in zip(self.core_base_radii, self.core_alphas):
            r = int(base_r * breath)
            draw_circle(self.image, (*color, alpha), self.center, r)
        draw_circle(self.image, (255, 255, 255), self.center, int(14 * breath))





class DetectionOverlay:
    """Optimized detection overlay."""
    __slots__ = ('font', 'small_font', 'impact_colors', 'shared_state', 'state_lock', 'title_surface',
                 'x0', 'y0')

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

        # Keep down from the top; carbon widget is top-right now, so left panel can be higher
        self.x0 = 20
        self.y0 = 70

    def draw(self, screen):
        with self.state_lock:
            detections = self.shared_state["detected_objects"]

        if not detections:
            return

        panel_h = min(len(detections), 8) * 30 + 50
        panel_surface = pygame.Surface((320, panel_h), pygame.SRCALPHA)
        panel_surface.fill((0, 0, 0, 140))
        screen.blit(panel_surface, (self.x0, self.y0))
        screen.blit(self.title_surface, (self.x0 + 10, self.y0 + 8))

        y = self.y0 + 38
        white = (255, 255, 255)

        for det in detections[:8]:
            color = self.impact_colors.get(det.get("carbon_impact", "unknown"), (180, 180, 180))
            pygame.draw.circle(screen, color, (self.x0 + 20, y + 10), 5)

            label = det.get("label", "?")
            conf = float(det.get("confidence", 0.0))
            text_surf = self.small_font.render(f"{label} ({conf:.0%})", True, white)
            screen.blit(text_surf, (self.x0 + 32, y + 2))

            impact_surf = self.small_font.render(det.get("carbon_impact", "?"), True, color)
            screen.blit(impact_surf, (self.x0 + 240, y + 2))
            y += 30

        if len(detections) > 8:
            more = self.small_font.render(f"+{len(detections) - 8} more...", True, (150, 150, 150))
            screen.blit(more, (self.x0 + 32, y + 2))


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

        fill_width = max(0.0, min(hp, 100.0)) * 0.01 * self.width
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

        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill(self.bg_color)
        screen.blit(panel, (x, y))

        screen.blit(text_surface, (x + 10, y + 5))