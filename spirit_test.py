import pygame
import math
import random
import time
import sys

# --- Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

class Particle:
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

class SpiritCompanion(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((600, 600), pygame.SRCALPHA)
        self.rect = self.image.get_rect(center=(300, SCREEN_HEIGHT // 2))
        
        self.home_pos = pygame.Vector2(300, SCREEN_HEIGHT // 2)
        self.pos = pygame.Vector2(300, SCREEN_HEIGHT // 2)
        
        self.hover_angle = 0
        self.wing_angle = 0
        self.carbon_v = 0.27 
        self.particles = []
        
        # Happy Arc Logic
        self.is_jumping = False
        self.jump_progress = 0
        self.jump_timer = time.time()
        
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
        now = time.time()

        # State Mapping
        if self.carbon_v <= 0.26:
            state, color, wing_speed = "pristine", (180, 255, 200), 0.08
        elif self.carbon_v <= 0.6:
            state, color, wing_speed = "calm", (80, 255, 150), 0.18
        else:
            state, color, wing_speed = "angry", (255, 40, 40), 1.2

        self.wing_angle += wing_speed
        self.hover_angle += 0.08

        # --- MOVEMENT BEHAVIORS ---
        if state == "angry":
            # VIOLENT STAR ANIMATION
            # Rapidly cycle through a 5-pointed star path
            self.star_angle += 1.5 
            r = 50 # Radius of the star points
            # Traditional star vertex math
            # We jump between outer points in a star pattern (0 -> 144 -> 288 -> 72 -> 216 -> 0 deg)
            points = [0, 144, 288, 72, 216]
            idx = int(self.star_angle % 5)
            target_angle = math.radians(points[idx])
            
            star_offset = pygame.Vector2(math.cos(target_angle) * r, math.sin(target_angle) * r)
            jitter = pygame.Vector2(random.randint(-15, 15), random.randint(-15, 15))
            self.pos = self.home_pos + star_offset + jitter
            self.is_jumping = False
            
        elif state == "pristine":
            # SPEEDY 180-DEGREE ARC JUMP
            if not self.is_jumping and now - self.jump_timer > random.uniform(3, 5):
                self.is_jumping = True
                self.jump_progress = 0
            
            if self.is_jumping:
                # Faster jump velocity
                self.jump_progress += 0.035 
                
                # Semicircle math for 180 deg radius
                # Ease-in-out for the arc: progress starts fast, slows at peak (90deg), ends slow
                angle = self.jump_progress * math.pi 
                radius = 250
                
                # Path: Semicircle arc that ends where it started
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
        
        else:
            # Base State (0.27): Calm Floating with High Freedom
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
        pygame.draw.circle(self.image, WHITE, center, 14)

        # Sparkles
        if random.random() > 0.4:
            self.particles.append(Particle(self.rect.centerx, self.rect.centery, color))

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    spirit = SpiritCompanion()
    sim_v = 0.27

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return
            if event.type == pygame.KEYDOWN:
                # Quit with Control + X
                if event.key == pygame.K_x and (pygame.key.get_mods() & pygame.KMOD_CTRL):
                    return

        # Controls
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: sim_v = min(1.0, sim_v + 0.008)
        if keys[pygame.K_DOWN]: sim_v = max(0.0, sim_v - 0.008)

        spirit.carbon_v = sim_v
        spirit.update()
        screen.fill(BLACK)
        
        for p in spirit.particles[:]:
            p.update()
            if p.life <= 0: spirit.particles.remove(p)
            else:
                pygame.draw.circle(screen, p.color + (int(255 * p.life),), (int(p.x), int(p.y)), int(p.size * p.life))
            
        screen.blit(spirit.image, spirit.rect)
        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()