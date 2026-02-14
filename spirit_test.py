import pygame
import math
import random
import time
import sys

# --- Configuration ---
SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)



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
                if event.key == pygame.K_w:
                    sim_v = min(1.0, sim_v + 0.05)
                if event.key == pygame.K_s:
                    sim_v = max(0.0, sim_v - 0.05)

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