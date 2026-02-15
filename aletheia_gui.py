# aletheia_gui.py - Big Buttons & Fixed Coordinates
import pygame
import math
import random
import time
import os
import array
from typing import List

# --- Global Configuration ---
CAMERA_WIDTH, CAMERA_HEIGHT = 1280, 720
YOLO_SIZE = 640
BLACK = (0, 0, 0)

# Pre-calculate common math constants
TWO_PI = 2 * math.pi
TAU = math.tau
PI_HALF = math.pi / 2

# --- Trig lookup table ---
_TRIG_N = 360
_TRIG_SCALE = _TRIG_N / TWO_PI
_SIN_TABLE = array.array('f', [math.sin(i * TWO_PI / _TRIG_N) for i in range(_TRIG_N)])
_COS_TABLE = array.array('f', [math.cos(i * TWO_PI / _TRIG_N) for i in range(_TRIG_N)])

def _fast_sin(angle): return _SIN_TABLE[int((angle % TWO_PI) * _TRIG_SCALE) % _TRIG_N]
def _fast_cos(angle): return _COS_TABLE[int((angle % TWO_PI) * _TRIG_SCALE) % _TRIG_N]


# =========================
# Quest Manager
# =========================
class QuestManager:
    """
    Manages Quests with Coordinate Fix + Big Buttons.
    """
    __slots__ = ("shared_state", "state_lock", "font_title", "font_body", "font_timer", "fog_sprite", "active_target", "quest_state", "timer_start", "timer_duration", "was_pinching", "carbon_table", "tracked_targets")

    def __init__(self, shared_state, state_lock):
        self.shared_state = shared_state; self.state_lock = state_lock
        self.font_title = pygame.font.Font(None, 40) # Bigger Font
        self.font_body = pygame.font.Font(None, 30)
        self.font_timer = pygame.font.Font(None, 50)

        self.fog_sprite = pygame.Surface((60, 60), pygame.SRCALPHA)
        for r in range(30, 0, -5):
            alpha = 5 + (30 - r) * 2
            pygame.draw.circle(self.fog_sprite, (255, 50, 50, alpha), (30, 30), r)

        self.active_target = None; self.quest_state = "IDLE"
        self.timer_start = 0.0; self.timer_duration = 30.0
        self.was_pinching = False
        self.tracked_targets = []
        self.carbon_table = {"high": 500.0, "medium": 150.0, "low": 10.0}

    def _is_clicked(self, cursor, is_pinching):
        return is_pinching and not self.was_pinching

    def _map_coords(self, box, sw, sh):
        x1, y1, x2, y2 = box
        raw_cx = (x1 + x2) / 2
        raw_cy = (y1 + y2) / 2
        scale = YOLO_SIZE / max(CAMERA_WIDTH, CAMERA_HEIGHT)
        img_h_in_yolo = CAMERA_HEIGHT * scale
        pad_y = (YOLO_SIZE - img_h_in_yolo) / 2
        norm_x = raw_cx / YOLO_SIZE
        norm_y = (raw_cy - pad_y) / img_h_in_yolo
        return int(norm_x * sw), int(norm_y * sh)

    def _update_tracked_targets(self, detections, sw, sh, dt):
        now = time.time()
        for det in detections:
            if det.get("carbon_impact", "low") != "high": continue
            box = det.get("box", (0, 0, 0, 0))
            cx, cy = self._map_coords(box, sw, sh)
            
            best_match, best_dist = None, 150.0
            for target in self.tracked_targets:
                d = math.hypot(target['x'] - cx, target['y'] - cy)
                if d < best_dist: best_match, best_dist = target, d
            
            if best_match:
                best_match['x'] += (cx - best_match['x']) * 0.2
                best_match['y'] += (cy - best_match['y']) * 0.2
                best_match['last_seen'] = now
                best_match['data'] = det
            else:
                self.tracked_targets.append({'x': cx, 'y': cy, 'last_seen': now, 'data': det})
        self.tracked_targets = [t for t in self.tracked_targets if now - t['last_seen'] < 0.8]

    def draw(self, screen, state_snapshot, dt=0.016):
        is_pinching = state_snapshot.get("is_pinching", False)
        cursor = state_snapshot.get("index_finger_tip", (0, 0))
        detections = state_snapshot.get("detected_objects", [])
        clicked = self._is_clicked(cursor, is_pinching)
        current_time = time.time()
        sw, sh = screen.get_size()
        
        self._update_tracked_targets(detections, sw, sh, dt)

        if self.quest_state == "IDLE":
            hovering = False
            for target in self.tracked_targets:
                cx, cy = int(target['x']), int(target['y'])
                self._draw_fog(screen, cx, cy, dt)
                if math.hypot(cursor[0]-cx, cursor[1]-cy) < 100:
                    hovering = True
                    if is_pinching:
                        self.active_target = target['data']
                        self.quest_state = "OFFER"
                        self.was_pinching = True 
            self._draw_cursor(screen, cursor, is_pinching, hovering)

        elif self.quest_state == "OFFER":
            if self.active_target:
                label = self.active_target.get("label", "Device")
                res = self._draw_popup(screen, "Vampire Power Detected!",
                                       f"This {label} is wasting energy.\nStart Unplug Challenge?",
                                       ["Accept Quest", "Ignore"], cursor, clicked)
                if res == "Accept Quest":
                    self.quest_state = "ACTIVE"; self.timer_start = current_time
                elif res == "Ignore":
                    self.quest_state = "IDLE"; self.active_target = None
            else: self.quest_state = "IDLE"
            self._draw_cursor(screen, cursor, is_pinching, False)

        elif self.quest_state == "ACTIVE":
            rem = max(0.0, self.timer_duration - (current_time - self.timer_start))
            res = self._draw_popup(screen, f"Challenge Active: {rem:.1f}s",
                                   "1. Unplug the device.\n2. Confirm below.",
                                   ["I Did It!", "Cancel"], cursor, clicked, (255, 100, 100))
            if res == "I Did It!": self._complete_quest()
            elif res == "Cancel": self.quest_state = "IDLE"; self.active_target = None
            self._draw_cursor(screen, cursor, is_pinching, False)

        self.was_pinching = is_pinching

    def _draw_cursor(self, screen, cursor, is_pinching, hovering):
        cx, cy = cursor
        color = (255, 255, 0) if hovering else ((50, 255, 100) if is_pinching else (200, 200, 200))
        radius = 12 if hovering or is_pinching else 8
        pygame.draw.circle(screen, color, (cx, cy), radius)
        pygame.draw.circle(screen, (0,0,0), (cx, cy), radius + 2, 2)

    def _draw_fog(self, screen, x, y, dt):
        t = time.time() * 5.0
        for i in range(5):
            ox = math.sin(t + i) * 30; oy = math.cos(t * 1.3 + i) * 30
            screen.blit(self.fog_sprite, (x + ox - 30, y + oy - 30))

    def _draw_popup(self, screen, title, body, options, cursor, clicked, highlight_color=(100, 200, 255)):
        sw, sh = screen.get_size()
        # BIGGER PANEL
        w, h = 500, 300 
        x, y = (sw - w) // 2, (sh - h) // 2
        
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        s.fill((20, 25, 30, 235))
        pygame.draw.rect(s, highlight_color, (0, 0, w, h), 3, border_radius=15)
        screen.blit(s, (x, y))
        
        t_surf = self.font_title.render(title, True, highlight_color)
        screen.blit(t_surf, (x + 25, y + 25))
        
        lines = body.split('\n')
        by = y + 80
        for line in lines:
            b_surf = self.font_body.render(line, True, (220, 220, 220))
            screen.blit(b_surf, (x + 25, by))
            by += 35
            
        # BIGGER BUTTONS
        btn_w, btn_h = 220, 80 
        bx = x + 20
        by = y + h - 100
        
        result = None
        for opt in options:
            btn_rect = pygame.Rect(bx, by, btn_w, btn_h)
            is_hover = btn_rect.collidepoint(cursor)
            
            bg = (60, 90, 110) if not is_hover else (100, 140, 120)
            border = (120, 120, 120) if not is_hover else highlight_color
            
            pygame.draw.rect(screen, bg, btn_rect, border_radius=12)
            pygame.draw.rect(screen, border, btn_rect, 3, border_radius=12)
            
            txt = self.font_body.render(opt, True, (255, 255, 255))
            tr = txt.get_rect(center=btn_rect.center)
            screen.blit(txt, tr)
            
            if is_hover and clicked: result = opt
            bx += btn_w + 20
        return result

    def _complete_quest(self):
        if not self.active_target: return
        impact = self.active_target.get("carbon_impact", "low")
        saved = self.carbon_table.get(impact, 10.0)
        label = self.active_target.get("label", "Device")
        with self.state_lock:
            self.shared_state["carbon_saved_g"] = float(self.shared_state.get("carbon_saved_g", 0.0)) + saved
            cur = int(self.shared_state.get("missions_completed", 0))
            tot = int(self.shared_state.get("missions_total", 5))
            if cur < tot: self.shared_state["missions_completed"] = cur + 1
            self.shared_state["last_savings_event"] = f"Unplugged {label}"
            self.shared_state["last_savings_event_time"] = time.time()
            self.shared_state["energy_waste_count"] = 0
        self.quest_state = "IDLE"; self.active_target = None


# =========================
# Standard Widgets (Unchanged)
# =========================
class CarbonSavingsWidget:
    __slots__ = ("shared_state", "state_lock", "font_large", "font_small", "padding", "width", "height", "displayed_value", "panel_alpha", "_panel", "_title_surf", "_cached_value_text", "_cached_value_surf", "_cached_event_text", "_cached_event_surf")
    def __init__(self, shared_state, state_lock, *, panel_alpha: int = 90):
        self.shared_state = shared_state; self.state_lock = state_lock
        self.font_large = pygame.font.Font(None, 42); self.font_small = pygame.font.Font(None, 24)
        self.padding = 20; self.width = 360; self.height = 120; self.displayed_value = 0.0; self.panel_alpha = panel_alpha
        self._panel = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        self._title_surf = self.font_small.render("Carbon Reduced", True, (120, 255, 160))
        self._cached_value_text = ""; self._cached_value_surf = None; self._cached_event_text = ""; self._cached_event_surf = None
    def draw(self, screen, state_snapshot=None):
        if state_snapshot:
            total_saved = float(state_snapshot.get("carbon_saved_g", 0.0))
            last_event = state_snapshot.get("last_savings_event", "")
        else:
            with self.state_lock:
                total_saved = float(self.shared_state.get("carbon_saved_g", 0.0))
                last_event = self.shared_state.get("last_savings_event", "")
        self.displayed_value += (total_saved - self.displayed_value) * 0.08
        total_kg = self.displayed_value / 1000.0
        sw, sh = screen.get_size()
        x = sw - self.width - self.padding; y = self.padding
        self._panel.fill((0, 0, 0, self.panel_alpha))
        screen.blit(self._panel, (x, y))
        screen.blit(self._title_surf, (x + 15, y + 10))
        val_txt = f"{total_kg:.2f} kg CO2e"
        if val_txt != self._cached_value_text:
            self._cached_value_text = val_txt; self._cached_value_surf = self.font_large.render(val_txt, True, (255, 255, 255))
        screen.blit(self._cached_value_surf, (x + 15, y + 40))
        if last_event:
            evt_txt = f"+ {last_event}"
            if evt_txt != self._cached_event_text:
                self._cached_event_text = evt_txt; self._cached_event_surf = self.font_small.render(evt_txt, True, (180, 255, 200))
            screen.blit(self._cached_event_surf, (x + 15, y + 85))

class OrbitParticle:
    __slots__ = ('angle', 'radius', 'ang_speed', 'size', 'life', 'color')
    def __init__(self, angle, radius, ang_speed, size, life, color):
        self.angle = angle; self.radius = radius; self.ang_speed = ang_speed
        self.size = size; self.life = life; self.color = color
    def update(self, dt60=1.0):
        self.angle += self.ang_speed * dt60; self.life -= 0.015 * dt60

class Particle:
    __slots__ = ('x', 'y', 'color', 'size', 'life', 'decay', 'vel_x', 'vel_y')
    def __init__(self, x, y, color):
        self.x, self.y = float(x), float(y); self.color = color
        self.size = random.randint(2, 5); self.life = 1.0; self.decay = random.uniform(0.02, 0.05)
        self.vel_x = random.uniform(-1.5, 1.5); self.vel_y = random.uniform(-1.5, 1.5)
    def update(self, dt60=1.0):
        self.x += self.vel_x * dt60; self.y += self.vel_y * dt60; self.life -= self.decay * dt60

class SpiritCompanion(pygame.sprite.Sprite):
    _wing_cache_global = {}; _star_points_cache = {}; _rotated_cache = {}; _ROTATED_CACHE_MAX = 256
    def __init__(self, shared_state, state_lock):
        super().__init__()
        self._render_surf = pygame.Surface((200, 200), pygame.SRCALPHA)
        self.image = pygame.Surface((400, 400), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=(320, 240))
        self.shared_state = shared_state; self.state_lock = state_lock
        try:
            info = pygame.display.Info(); sw, sh = info.current_w, info.current_h
            if sw < 100: sw, sh = 1280, 720
        except: sw, sh = 1280, 720
        self.home_pos = pygame.Vector2(sw * 0.15, sh * 0.5); self.pos = self.home_pos.copy()
        self.hover_angle = 0.0; self.wing_angle = 0.0
        self.particles = []; self.orbit_particles = []; self._last_orbit_spawn = 0.0
        self.current_state = "calm"; self.fsm_state = "calm"; self._pending_quest_complete = False
        self._calm_waypoint = self.home_pos.copy(); self._next_waypoint_epoch = 0.0
        self._angry_amp = 0.0; self.angry_jitter_rate_x = 19.0; self.angry_jitter_rate_y = 23.0
        self.post_pristine_calm_seconds = 2.0; self._post_pristine_cooldown_until = 0.0
        self.current_color = pygame.Vector3(80, 255, 150)
        self.celebration_phase = "idle"; self.celebration_progress = 0.0; self.circle_center = self.home_pos.copy()
        self.transitioning_to_pristine = False; self.transition_start_pos = self.pos.copy()
        self.transition_progress = 0.0; self._last_seen_savings_event_time = 0.0
        self._last_seen_savings_event_text = ""; self._pristine_active = False
        self._pristine_phase = "idle"; self._return_progress = 0.0
        self.waste_timeout_seconds = 10.0; self._last_waste_seen_epoch = 0.0
        self.color_calm = pygame.Vector3(80, 255, 150); self.color_angry = pygame.Vector3(255, 40, 40)
        self.color_pristine = pygame.Vector3(180, 255, 200)
        self.core_base_radii = [int((10 + (i * 9)) * 0.5) for i in range(6, 0, -1)]
        self.core_alphas = [170 // i for i in range(6, 0, -1)]
        self.center = (100, 100)
        self._max_particles = int(os.getenv("ALETHEIA_MAX_PARTICLES", "40"))
        self._wings_enabled = os.getenv("ALETHEIA_LITE_MODE", "0") != "1"

    def draw_ethereal_wing(self, surf, center, angle_offset, width, height, color, is_left=True):
        cache_key = (width, height, is_left)
        if cache_key not in SpiritCompanion._wing_cache_global:
            wing_layers = []
            for i in range(3, 0, -1):
                w, h = width + (i * 12), height + (i * 6)
                wing_surf = pygame.Surface((w * 2, h * 2), pygame.SRCALPHA)
                wing_layers.append((wing_surf, w, h, 50 // i))
            SpiritCompanion._wing_cache_global[cache_key] = wing_layers
        wing_layers = SpiritCompanion._wing_cache_global[cache_key]
        rot_angle = angle_offset + (_fast_sin(self.wing_angle) * 15)
        if not is_left: rot_angle = -rot_angle
        rot_q = round(rot_angle / 3.0) * 3.0
        color_q = (color[0] >> 4, color[1] >> 4, color[2] >> 4)
        for wing_surf, w, h, alpha in wing_layers:
            rc_key = (w, h, color_q, alpha, rot_q)
            rotated = SpiritCompanion._rotated_cache.get(rc_key)
            if rotated is None:
                wing_surf.fill((0, 0, 0, 0))
                pygame.draw.ellipse(wing_surf, (*color, alpha), (0, 0, w, h))
                rotated = pygame.transform.rotate(wing_surf, rot_q)
                SpiritCompanion._rotated_cache[rc_key] = rotated
                if len(SpiritCompanion._rotated_cache) > SpiritCompanion._ROTATED_CACHE_MAX:
                    for k in list(SpiritCompanion._rotated_cache.keys())[:64]: del SpiritCompanion._rotated_cache[k]
            surf.blit(rotated, rotated.get_rect(center=center), special_flags=pygame.BLEND_ADD)

    def update(self, state_snapshot=None, dt=None):
        if dt is None: dt60 = 1.0
        else: dt60 = min(dt * 60.0, 3.0)
        self._render_surf.fill((0, 0, 0, 0))
        now = pygame.time.get_ticks() * 0.001; epoch_now = time.time()
        if state_snapshot:
            saved_time = float(state_snapshot.get("last_savings_event_time", 0.0))
            saved_text = str(state_snapshot.get("last_savings_event", ""))
            waste_count = int(state_snapshot.get("energy_waste_count", 0))
            detections = state_snapshot.get("detected_objects", [])
        else:
            with self.state_lock:
                saved_time = float(self.shared_state.get("last_savings_event_time", 0.0))
                saved_text = str(self.shared_state.get("last_savings_event", ""))
                waste_count = int(self.shared_state.get("energy_waste_count", 0))
                detections = self.shared_state.get("detected_objects", [])
        
        waste_present = waste_count > 0 or any(d.get("carbon_impact") == "high" for d in detections)
        if waste_present: self._last_waste_seen_epoch = epoch_now
        waste_effective = (epoch_now - self._last_waste_seen_epoch) <= self.waste_timeout_seconds
        
        quest_complete = False
        if saved_time > 0.0: quest_complete = saved_time > self._last_seen_savings_event_time
        elif saved_text: quest_complete = saved_text != self._last_seen_savings_event_text
        if quest_complete:
            if saved_time > 0.0: self._last_seen_savings_event_time = saved_time
            if saved_text: self._last_seen_savings_event_text = saved_text

        if self.fsm_state == "calm":
            if waste_effective and epoch_now >= self._post_pristine_cooldown_until: self.fsm_state = "angry"
        elif self.fsm_state == "angry":
            if not waste_effective: self.fsm_state = "calm"
            elif quest_complete:
                self._pristine_active = True; self._pristine_phase = "in"; self.celebration_progress = 0.0
                self.circle_center = self.home_pos.copy(); self.transitioning_to_pristine = True
                self.transition_progress = 0.0; self.transition_start_pos = self.pos.copy(); self.fsm_state = "pristine"
        elif self.fsm_state == "pristine":
            if not self._pristine_active:
                self.fsm_state = "calm"; self._post_pristine_cooldown_until = epoch_now + self.post_pristine_calm_seconds
        self.current_state = self.fsm_state

        lerp_speed = 0.05 * dt60
        if self.current_state == "calm": target = self.color_calm
        elif self.current_state == "angry": target = self.color_angry
        else: target = self.color_pristine
        self.current_color.x += (target.x - self.current_color.x) * lerp_speed
        self.current_color.y += (target.y - self.current_color.y) * lerp_speed
        self.current_color.z += (target.z - self.current_color.z) * lerp_speed
        color = (int(self.current_color.x), int(self.current_color.y), int(self.current_color.z))

        draw_wings = True
        if self.current_state == "calm": wing_speed = 0.18 * dt60; breath = 1.0 + _fast_sin(now * TAU * 1.6) * 0.045
        elif self.current_state == "angry": wing_speed = 1.2 * dt60; breath = 1.0
        else: wing_speed = 0.08 * dt60; breath = 1.0 + _fast_sin(now * TAU * 1.2) * 0.06
        self.wing_angle += wing_speed; self.hover_angle += 0.08 * dt60

        if self.current_state == "angry":
            self._angry_amp += (22.0 - self._angry_amp) * 0.10 * dt60
            self.pos.x = self.home_pos.x + _fast_sin(now * self.angry_jitter_rate_x) * self._angry_amp
            self.pos.y = self.home_pos.y + _fast_cos(now * self.angry_jitter_rate_y) * self._angry_amp
        elif self.current_state == "pristine":
            circle_radius = 80; circle_speed = 0.018 * dt60
            if self._pristine_phase == "in":
                self.transition_progress += 0.02 * dt60
                if self.transition_progress >= 1.0:
                    self.transition_progress = 1.0; self.transitioning_to_pristine = False; self._pristine_phase = "circle"
                t = self.transition_progress; tx = self.circle_center.x + circle_radius; ty = self.circle_center.y
                self.pos.x = self.transition_start_pos.x + (tx - self.transition_start_pos.x) * t
                self.pos.y = self.transition_start_pos.y + (ty - self.transition_start_pos.y) * t
            elif self._pristine_phase == "circle":
                self.celebration_progress += circle_speed
                if self.celebration_progress >= 1.0:
                    self.celebration_progress = 1.0; self._pristine_phase = "return"; self._return_progress = 0.0; self._return_start_pos = self.pos.copy()
                angle = self.celebration_progress * TWO_PI
                self.pos.x = self.circle_center.x + _fast_cos(angle) * circle_radius
                self.pos.y = self.circle_center.y + _fast_sin(angle) * circle_radius
            elif self._pristine_phase == "return":
                self._return_progress += 0.03 * dt60
                if self._return_progress >= 1.0: self._return_progress = 1.0; self._pristine_phase = "idle"; self._pristine_active = False
                t = self._return_progress; start = getattr(self, "_return_start_pos", self.pos)
                self.pos.x = start.x + (self.home_pos.x - start.x) * t
                self.pos.y = start.y + (self.home_pos.y - start.y) * t
        else:
            if epoch_now >= self._next_waypoint_epoch:
                self._next_waypoint_epoch = epoch_now + random.uniform(2.0, 4.0)
                self._calm_waypoint = self.home_pos + pygame.Vector2(random.uniform(-45, 45), random.uniform(-30, 30))
            self.pos.x += (self._calm_waypoint.x - self.pos.x) * 0.03 * dt60
            self.pos.y += (self._calm_waypoint.y - self.pos.y) * 0.03 * dt60
        self.rect.center = (int(self.pos.x), int(self.pos.y))

        if self.current_state in ("calm", "pristine"):
            if now - self._last_orbit_spawn > 0.06 and len(self.orbit_particles) < self._max_particles:
                self._last_orbit_spawn = now
                for _ in range(2): self.orbit_particles.append(OrbitParticle(random.uniform(0, TAU), random.uniform(20, 60), random.uniform(0.02, 0.06) * random.choice((-1, 1)), random.randint(1, 3), 1.0, color))
        for p in self.orbit_particles: p.update(dt60)
        self.orbit_particles = [p for p in self.orbit_particles if p.life > 0]
        
        cx, cy = self.center; rs = self._render_surf
        for p in self.orbit_particles:
            ox = int(cx + _fast_cos(p.angle) * (p.radius * breath))
            oy = int(cy + _fast_sin(p.angle) * (p.radius * breath))
            pygame.draw.circle(rs, (*p.color, int(180 * min(1.0, max(0.0, p.life)))), (ox, oy), p.size)
        if draw_wings and self._wings_enabled:
            self.draw_ethereal_wing(rs, (cx - 30, cy - 15), 30, 60, 18, color, True)
            self.draw_ethereal_wing(rs, (cx + 30, cy - 15), 30, 60, 18, color, False)
            self.draw_ethereal_wing(rs, (cx - 25, cy + 5), -20, 45, 13, color, True)
            self.draw_ethereal_wing(rs, (cx + 25, cy + 5), -20, 45, 13, color, False)
        for base_r, alpha in zip(self.core_base_radii, self.core_alphas):
            pygame.draw.circle(rs, (*color, alpha), self.center, int(base_r * breath))
        pygame.draw.circle(rs, (255, 255, 255), self.center, int(7 * breath))
        pygame.transform.smoothscale(self._render_surf, (400, 400), self.image)

class DetectionOverlay:
    __slots__ = ('font', 'small_font', 'impact_colors', 'shared_state', 'state_lock', 'title_surface', 'x0', 'y0', '_panel', '_text_cache')
    def __init__(self, shared_state, state_lock):
        self.font = pygame.font.Font(None, 28); self.small_font = pygame.font.Font(None, 22)
        self.impact_colors = {"high": (255, 50, 50), "medium": (255, 180, 0), "low": (0, 220, 100), "unknown": (180, 180, 180)}
        self.shared_state = shared_state; self.state_lock = state_lock
        self.title_surface = self.font.render("Detected Objects", True, (255, 255, 255))
        self.x0 = 20; self.y0 = 70; self._panel = pygame.Surface((320, 300), pygame.SRCALPHA); self._text_cache = {}
    def _get_text(self, text, color):
        key = (text, color)
        if key not in self._text_cache: self._text_cache[key] = self.small_font.render(text, True, color)
        return self._text_cache[key]
    def draw(self, screen, state_snapshot=None):
        if state_snapshot: detections = state_snapshot.get("detected_objects", [])
        else:
            with self.state_lock: detections = self.shared_state["detected_objects"]
        if not detections: return
        panel_h = min(len(detections), 8) * 30 + 50
        self._panel.fill((0, 0, 0, 0)); pygame.draw.rect(self._panel, (0, 0, 0, 140), (0, 0, 320, panel_h))
        screen.blit(self._panel, (self.x0, self.y0), area=(0, 0, 320, panel_h))
        screen.blit(self.title_surface, (self.x0 + 10, self.y0 + 8))
        y = self.y0 + 38
        for det in detections[:8]:
            color = self.impact_colors.get(det.get("carbon_impact", "unknown"), (180, 180, 180))
            pygame.draw.circle(screen, color, (self.x0 + 20, y + 10), 5)
            screen.blit(self._get_text(f"{det.get('label','?')} ({float(det.get('confidence',0)):.0%})", (255,255,255)), (self.x0 + 32, y + 2))
            screen.blit(self._get_text(det.get("carbon_impact", "?"), color), (self.x0 + 240, y + 2))
            y += 30

class HealthBar:
    __slots__ = ('shared_state', 'state_lock', 'width', 'height', 'font', 'fill_color', 'text_color', 'bg_panel', '_cached_hp', '_cached_hp_surf')
    def __init__(self, shared_state, state_lock):
        self.shared_state = shared_state; self.state_lock = state_lock
        self.width = 200; self.height = 20; self.font = pygame.font.Font(None, 20)
        self.fill_color = (140, 255, 180); self.text_color = (255, 255, 255)
        self.bg_panel = pygame.Surface((self.width, self.height), pygame.SRCALPHA); self.bg_panel.fill((255, 255, 255, 25))
        self._cached_hp = -1; self._cached_hp_surf = None
    def draw(self, screen, state_snapshot=None):
        if state_snapshot: hp = state_snapshot.get("health", 0)
        else:
            with self.state_lock: hp = self.shared_state.get("health", 0)
        sw, sh = screen.get_size(); x = sw - self.width - 10; y = sh - self.height - 10
        screen.blit(self.bg_panel, (x, y))
        pygame.draw.rect(screen, self.fill_color, (x, y, max(0.0, min(hp, 100.0)) * 0.01 * self.width, self.height))
        if int(hp) != self._cached_hp:
            self._cached_hp = int(hp); self._cached_hp_surf = self.font.render(f"HP {int(hp)}%", True, self.text_color)
        screen.blit(self._cached_hp_surf, (x + 5, y + 2))

class MissionTracker:
    def __init__(self, shared_state, state_lock):
        self.shared_state = shared_state; self.state_lock = state_lock
        self.padding = 10; self.font = pygame.font.Font(None, 22)
        self.bg_color = (255, 255, 255, 25); self.text_color = (255, 255, 255)
        self._panel = pygame.Surface((300, 40), pygame.SRCALPHA)
        self._cached_text = ""; self._cached_text_surf = None; self._cached_width = 0; self._cached_height = 0
    def draw(self, screen, state_snapshot=None):
        if state_snapshot: cur = int(state_snapshot.get("missions_completed", 0)); tot = int(state_snapshot.get("missions_total", 5))
        else:
            with self.state_lock: cur = int(self.shared_state.get("missions_completed", 0)); tot = int(self.shared_state.get("missions_total", 5))
        txt = f"Daily Mission: {cur}/{tot} Completed"
        if txt != self._cached_text:
            self._cached_text = txt; self._cached_text_surf = self.font.render(txt, True, self.text_color)
            self._cached_width = self._cached_text_surf.get_width() + 20; self._cached_height = self._cached_text_surf.get_height() + 10
        sw, sh = screen.get_size(); x = sw - self._cached_width - self.padding; y = sh - 60
        self._panel.fill((0,0,0,0)); pygame.draw.rect(self._panel, self.bg_color, (0, 0, self._cached_width, self._cached_height))
        screen.blit(self._panel, (x, y), area=(0, 0, self._cached_width, self._cached_height))
        screen.blit(self._cached_text_surf, (x + 10, y + 5))