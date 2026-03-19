import pygame
import sys
import math
import random

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 800
FPS = 60

WHITE = (255, 255, 255)

# Colori scarpa
SHOE_DARK   = ( 80,  45,  15)   # suola / contorno
SHOE_UPPER  = (139,  90,  43)   # tomaia principale
SHOE_MID    = (160, 110,  55)   # highlight centrale
SHOE_SOLE   = ( 55,  30,  10)   # bordo suola

STRIDE_LENGTH = 120.0
HIP_WIDTH     = 10.0
AGENT_SPEED   = 150.0

SHOE_LENGTH = 30.0
SHOE_WIDTH  = 12.0


def get_relative_position(phase: float) -> float:
    p = phase % 1.0
    if p < 0.5:
        return 1.0 - 4.0 * p
    else:
        return 4.0 * p - 3.0


def draw_shoe(screen, cx: float, cy: float, dir_x: float, dir_y: float, is_left: bool):
    """
    Disegna una scarpa marrone orientata nella direzione di marcia.
    Forma: punta arrotondata davanti, tallone squadrato dietro.
    """
    half_l = SHOE_LENGTH / 2.0
    half_w = SHOE_WIDTH  / 2.0

    norm_x = -dir_y
    norm_y =  dir_x

    # ── Suola (polygon più largo, più scuro) ─────────────────────────────
    sl = half_l * 1.10
    sw = half_w * 1.25

    sole_pts = [
        (cx + dir_x * sl        + norm_x * sw,        cy + dir_y * sl        + norm_y * sw),
        (cx + dir_x * sl        - norm_x * sw,        cy + dir_y * sl        - norm_y * sw),
        (cx - dir_x * sl * 0.9  - norm_x * sw * 0.9,  cy - dir_y * sl * 0.9  - norm_y * sw * 0.9),
        (cx - dir_x * sl * 0.9  + norm_x * sw * 0.9,  cy - dir_y * sl * 0.9  + norm_y * sw * 0.9),
    ]
    pygame.draw.polygon(screen, SHOE_SOLE, sole_pts)

    # ── Tomaia principale ──────────────────────────────────────────────
    # Poligono a 6 punti: punta rastremata davanti, tallone rettangolare dietro
    w_front = half_w * 0.55   # punta stretta
    w_mid   = half_w * 1.05
    w_back  = half_w * 0.90

    pts = [
        # punta (davanti)
        (cx + dir_x * half_l,                              cy + dir_y * half_l),
        # lato destro: fronte → centro → tallone
        (cx + dir_x * half_l * 0.5  - norm_x * w_front,   cy + dir_y * half_l * 0.5  - norm_y * w_front),
        (cx - dir_x * half_l * 0.15 - norm_x * w_mid,     cy - dir_y * half_l * 0.15 - norm_y * w_mid),
        (cx - dir_x * half_l        - norm_x * w_back,     cy - dir_y * half_l        - norm_y * w_back),
        # tallone
        (cx - dir_x * half_l        + norm_x * w_back,     cy - dir_y * half_l        + norm_y * w_back),
        # lato sinistro: tallone → centro → fronte
        (cx - dir_x * half_l * 0.15 + norm_x * w_mid,     cy - dir_y * half_l * 0.15 + norm_y * w_mid),
        (cx + dir_x * half_l * 0.5  + norm_x * w_front,   cy + dir_y * half_l * 0.5  + norm_y * w_front),
    ]
    pygame.draw.polygon(screen, SHOE_UPPER, pts)

    # ── Striscia highlight centrale ────────────────────────────────────
    hw = half_w * 0.30
    hi_pts = [
        (cx + dir_x * half_l * 0.85,               cy + dir_y * half_l * 0.85),
        (cx + dir_x * half_l * 0.20 - norm_x * hw, cy + dir_y * half_l * 0.20 - norm_y * hw),
        (cx - dir_x * half_l * 0.50 - norm_x * hw, cy - dir_y * half_l * 0.50 - norm_y * hw),
        (cx - dir_x * half_l * 0.50 + norm_x * hw, cy - dir_y * half_l * 0.50 + norm_y * hw),
        (cx + dir_x * half_l * 0.20 + norm_x * hw, cy + dir_y * half_l * 0.20 + norm_y * hw),
    ]
    pygame.draw.polygon(screen, SHOE_MID, hi_pts)

    # ── Contorno ───────────────────────────────────────────────────────
    pygame.draw.polygon(screen, SHOE_DARK, pts, 2)

    # ── Laccio / dettaglio punta ───────────────────────────────────────
    tip_x = cx + dir_x * (half_l * 0.75)
    tip_y = cy + dir_y * (half_l * 0.75)
    pygame.draw.circle(screen, SHOE_DARK, (int(tip_x), int(tip_y)), 2)


class HumanKinematics:
    def __init__(self, width: int, height: int):
        self.env_width  = width
        self.env_height = height
        self.cx = 0.0
        self.cy = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.phase  = 0.0
        self.dir_x  = 0.0
        self.dir_y  = -1.0
        self.respawn()

    def respawn(self):
        edge = random.randint(0, 3)
        if edge == 0:
            self.cx = random.uniform(0, self.env_width)
            self.cy = 0.0
            raw_vx = random.uniform(-1.0, 1.0)
            raw_vy = random.uniform(0.1, 1.0)
        elif edge == 1:
            self.cx = self.env_width
            self.cy = random.uniform(0, self.env_height)
            raw_vx = random.uniform(-1.0, -0.1)
            raw_vy = random.uniform(-1.0, 1.0)
        elif edge == 2:
            self.cx = random.uniform(0, self.env_width)
            self.cy = self.env_height
            raw_vx = random.uniform(-1.0, 1.0)
            raw_vy = random.uniform(-1.0, -0.1)
        else:
            self.cx = 0.0
            self.cy = random.uniform(0, self.env_height)
            raw_vx = random.uniform(0.1, 1.0)
            raw_vy = random.uniform(-1.0, 1.0)

        length     = math.hypot(raw_vx, raw_vy)
        self.vx    = (raw_vx / length) * AGENT_SPEED
        self.vy    = (raw_vy / length) * AGENT_SPEED
        self.dir_x = raw_vx / length
        self.dir_y = raw_vy / length

    def update(self, dt: float):
        self.cx += self.vx * dt
        self.cy += self.vy * dt
        if (self.cx < -50 or self.cx > self.env_width  + 50 or
            self.cy < -50 or self.cy > self.env_height + 50):
            self.respawn()
        self.phase += (AGENT_SPEED * dt) / (2.0 * STRIDE_LENGTH)

    def get_foot_positions(self):
        rel_l = (STRIDE_LENGTH / 2.0) * get_relative_position(self.phase)
        rel_r = (STRIDE_LENGTH / 2.0) * get_relative_position(self.phase + 0.5)
        norm_x = -self.dir_y
        norm_y =  self.dir_x
        lx = self.cx + self.dir_x * rel_l + norm_x * HIP_WIDTH
        ly = self.cy + self.dir_y * rel_l + norm_y * HIP_WIDTH
        rx = self.cx + self.dir_x * rel_r - norm_x * HIP_WIDTH
        ry = self.cy + self.dir_y * rel_r - norm_y * HIP_WIDTH
        return (lx, ly), (rx, ry)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Walking Shoes")
    clock = pygame.time.Clock()

    human = HumanKinematics(WINDOW_WIDTH, WINDOW_HEIGHT)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        human.update(dt)
        (lx, ly), (rx, ry) = human.get_foot_positions()

        screen.fill(WHITE)

        # Determina quale scarpa è "davanti" (fase) e disegnala per ultima
        rel_l = get_relative_position(human.phase)
        rel_r = get_relative_position(human.phase + 0.5)
        if rel_l > rel_r:
            draw_shoe(screen, rx, ry, human.dir_x, human.dir_y, is_left=False)
            draw_shoe(screen, lx, ly, human.dir_x, human.dir_y, is_left=True)
        else:
            draw_shoe(screen, lx, ly, human.dir_x, human.dir_y, is_left=True)
            draw_shoe(screen, rx, ry, human.dir_x, human.dir_y, is_left=False)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()