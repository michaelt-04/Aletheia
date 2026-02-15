import pygame


class CarbonSavingsWidget:
    def __init__(self, shared_state, state_lock):
        self.shared_state = shared_state
        self.state_lock = state_lock

        self.font_large = pygame.font.Font(None, 42)
        self.font_small = pygame.font.Font(None, 24)

        self.padding = 20
        self.width = 360
        self.height = 120

        self.displayed_value = 0.0  # smooth animation

    def draw(self, screen):
        with self.state_lock:
            total_saved = self.shared_state.get("carbon_saved_g", 0.0)
            last_event = self.shared_state.get("last_savings_event", "")

        # Smooth animated count-up
        self.displayed_value += (total_saved - self.displayed_value) * 0.08
        total_saved_kg = self.displayed_value / 1000.0

        screen_width, screen_height = screen.get_size()

        # --- LEFT SIDE POSITION ---
        x = self.padding
        y = self.padding

        # --- MORE TRANSPARENT PANEL ---
        panel = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 70))  # lower alpha (was 150)
        screen.blit(panel, (x, y))

        # Title
        title = self.font_small.render("Carbon Reduced", True, (120, 255, 160))
        screen.blit(title, (x + 15, y + 10))

        # Big number (CO2e with normal 2)
        value = self.font_large.render(
            f"{total_saved_kg:.2f} kg CO2e",
            True,
            (255, 255, 255),
        )
        screen.blit(value, (x + 15, y + 40))

        # Last savings event
        if last_event:
            event_text = self.font_small.render(
                f"+ {last_event}",
                True,
                (180, 255, 200),
            )
            screen.blit(event_text, (x + 15, y + 85))
