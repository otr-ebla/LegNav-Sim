"""
Simulazione Camminata Umana 2D — Vista dall'Alto
Scenari multipli con navigazione, ostacoli e interazione tra persone.

MODELLO DEI PIEDI (onda triangolare):
  y_foot_l(t) = y_body + (L/2) * Λ(t/T)
  y_foot_r(t) = y_body + (L/2) * Λ(t/T + 0.5)
  con coordinate ruotate nella direzione di marcia.

SCENARI:
  1 - Moto rettilineo libero
  2 - Moto curvilineo (percorso a S)
  3 - Ostacoli statici con path-finding locale
  4 - Due persone che si avvicinano e si schivano
  5 - Folla — più persone con collision avoidance (steering behaviors)

Tasti: 1-5 seleziona scenario, ESC esci, SPAZIO pausa
"""

import pygame
import math
import random
import sys
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# ── COSTANTI ─────────────────────────────────────────────────────────────────
WIDTH,  HEIGHT  = 1100, 700
FPS             = 60
ROOM_MARGIN     = 60

# Modello piedi
STRIDE_L   = 36.0   # px  semi-ampiezza oscillazione (L/2)
STEP_T     = 0.55   # s   periodo falcata
HIP_W      = 11.0   # px  semi-larghezza fianchi

# Agenti
AGENT_RADIUS   = 14
AGENT_SPEED    = 90.0    # px/s
TURN_SPEED     = 3.5     # rad/s   velocità di sterzata
TRAIL_LEN      = 180
AVOIDANCE_DIST = 80      # px  distanza entro cui scansare

# ── COLORI ───────────────────────────────────────────────────────────────────
BG          = (12,  16,  22)
ROOM_BG     = (17,  23,  33)
GRID_C      = (22,  30,  44)
WALL_C      = (40,  55,  80)
OBSTACLE_C  = (35,  50,  75)
OBSTACLE_HL = (55,  75, 110)
TEXT_C      = (160, 185, 220)
DIM_C       = ( 70,  95, 130)
ACCENT_C    = (255, 200,  55)
PANEL_C     = ( 8,  12,  20)

PERSON_PALETTE = [
    ((74, 158, 255), (38, 192, 106), (255, 112,  67)),  # body, left, right
    ((220, 80, 160), (255, 180,  50), (50, 220, 200)),
    ((120, 220,  80), (255, 100, 100), (100, 160, 255)),
    ((255, 160,  40), (80, 200, 230), (220, 100, 200)),
    ((180, 255, 130), (255, 120, 80), (80, 180, 255)),
    ((255, 100, 130), (100, 230, 180), (200, 160, 50)),
]


# ── GEOMETRIA ────────────────────────────────────────────────────────────────
def triangle(phi: float) -> float:
    phi = phi % 1.0
    return 4.0*phi - 1.0 if phi < 0.5 else -4.0*phi + 3.0

def vec2_len(x, y):    return math.hypot(x, y)
def vec2_norm(x, y):
    d = math.hypot(x, y)
    return (x/d, y/d) if d > 1e-6 else (0.0, 0.0)
def vec2_dot(ax,ay,bx,by): return ax*bx + ay*by
def angle_diff(a, b):
    d = (b - a) % (2*math.pi)
    return d - 2*math.pi if d > math.pi else d


# ── OSTACOLO ─────────────────────────────────────────────────────────────────
@dataclass
class Obstacle:
    x: float; y: float; w: float; h: float; label: str = ""

    def rect(self): return pygame.Rect(int(self.x), int(self.y), int(self.w), int(self.h))
    def center(self): return (self.x + self.w/2, self.y + self.h/2)
    def nearest_point(self, px, py):
        cx = max(self.x, min(px, self.x + self.w))
        cy = max(self.y, min(py, self.y + self.h))
        return cx, cy
    def dist_to(self, px, py):
        npx, npy = self.nearest_point(px, py)
        return math.hypot(px - npx, py - npy)


# ── AGENTE ───────────────────────────────────────────────────────────────────
class Agent:
    def __init__(self, x, y, angle, speed=AGENT_SPEED, color_idx=0, label=""):
        self.x      = float(x)
        self.y      = float(y)
        self.angle  = float(angle)   # direzione attuale (rad)
        self.target_angle = float(angle)
        self.speed  = speed
        self.t      = random.uniform(0, STEP_T)
        self.colors = PERSON_PALETTE[color_idx % len(PERSON_PALETTE)]
        self.label  = label

        self.trail: List[Tuple] = []
        self.foot_trail_l: List[Tuple] = []
        self.foot_trail_r: List[Tuple] = []

        # waypoints per lo scenario
        self.waypoints: List[Tuple] = []
        self.wp_idx   = 0
        self.active   = True
        self.paused   = False

        # per collision avoidance
        self.avoidance_angle = 0.0
        self.avoiding = False

    @property
    def pos(self): return (self.x, self.y)

    def set_waypoints(self, wps, loop=False):
        self.waypoints = list(wps)
        self.wp_idx    = 0
        self.loop      = loop

    def _foot_pos(self, side_sign):
        """Posizione del piede in coord mondo"""
        phase_offset = 0.0 if side_sign > 0 else 0.5
        lam = triangle(self.t / STEP_T + phase_offset)
        # asse avanzamento e laterale
        fx = math.cos(self.angle)
        fy = math.sin(self.angle)
        lx = -math.sin(self.angle)
        ly =  math.cos(self.angle)
        foot_x = self.x + lx * (HIP_W * side_sign) + fx * STRIDE_L * lam
        foot_y = self.y + ly * (HIP_W * side_sign) + fy * STRIDE_L * lam
        return foot_x, foot_y

    def update(self, dt, obstacles: List[Obstacle], agents: List['Agent']):
        if not self.active or self.paused:
            return

        self.t += dt

        # ── naviga verso waypoint ──────────────────────────────────────────
        desired_angle = self.target_angle
        if self.waypoints and self.wp_idx < len(self.waypoints):
            wx, wy = self.waypoints[self.wp_idx]
            dx, dy = wx - self.x, wy - self.y
            dist   = math.hypot(dx, dy)
            if dist < 18:
                self.wp_idx += 1
                if self.loop:
                    self.wp_idx %= len(self.waypoints)
                elif self.wp_idx >= len(self.waypoints):
                    self.active = False
                    return
            else:
                desired_angle = math.atan2(dy, dx)

        # ── ostacoli: steering ────────────────────────────────────────────
        obs_steer = 0.0
        for obs in obstacles:
            d = obs.dist_to(self.x, self.y)
            if d < AVOIDANCE_DIST + 20:
                npx, npy = obs.nearest_point(self.x, self.y)
                away_x, away_y = vec2_norm(self.x - npx, self.y - npy)
                avoid_ang = math.atan2(away_y, away_x)
                diff = angle_diff(self.angle, avoid_ang)
                weight = max(0, 1 - d / (AVOIDANCE_DIST + 20))
                obs_steer += diff * weight * 2.5

        # ── agenti: collision avoidance ───────────────────────────────────
        agent_steer = 0.0
        for other in agents:
            if other is self or not other.active:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            d  = math.hypot(dx, dy)
            if d < AVOIDANCE_DIST and d > 1:
                # quanto è "davanti" l'altro?
                forward_dot = vec2_dot(math.cos(self.angle), math.sin(self.angle),
                                       dx/d, dy/d)
                if forward_dot > -0.3:  # sta davanti a me
                    # sterza a destra o sinistra in base alla posizione relativa
                    side = dx * math.sin(self.angle) - dy * math.cos(self.angle)
                    sign = -1 if side >= 0 else 1   # sterza opposto al lato
                    weight = max(0, 1 - d / AVOIDANCE_DIST)
                    agent_steer += sign * weight * 2.0

        # angolo desiderato totale
        total_desired = desired_angle
        diff_to_des = angle_diff(self.angle, total_desired)
        steer = diff_to_des * 4.0 + obs_steer + agent_steer
        steer = max(-TURN_SPEED, min(TURN_SPEED, steer))
        self.angle += steer * dt

        # ── muovi ─────────────────────────────────────────────────────────
        self.x += math.cos(self.angle) * self.speed * dt
        self.y += math.sin(self.angle) * self.speed * dt

        # rimbalza sulle pareti della stanza
        margin = AGENT_RADIUS + 5
        self.x = max(ROOM_MARGIN + margin, min(WIDTH  - ROOM_MARGIN - margin, self.x))
        self.y = max(ROOM_MARGIN + margin, min(HEIGHT - ROOM_MARGIN - margin, self.y))

        # ── aggiorna tracce ───────────────────────────────────────────────
        self.trail.append((self.x, self.y))
        if len(self.trail) > TRAIL_LEN: self.trail.pop(0)

        fl = self._foot_pos(-1)
        fr = self._foot_pos(+1)
        self.foot_trail_l.append(fl)
        self.foot_trail_r.append(fr)
        if len(self.foot_trail_l) > TRAIL_LEN: self.foot_trail_l.pop(0)
        if len(self.foot_trail_r) > TRAIL_LEN: self.foot_trail_r.pop(0)

    def draw(self, surf):
        if not self.active: return
        bc, lc, rc = self.colors

        # ── tracce corpo ──────────────────────────────────────────────────
        n = len(self.trail)
        for i in range(1, n):
            alpha = int((i / n) * 55)
            col   = (*bc, alpha)
            s = pygame.Surface((4,4), pygame.SRCALPHA)
            pygame.draw.circle(s, col, (2,2), 2)
            surf.blit(s, (int(self.trail[i][0])-2, int(self.trail[i][1])-2))

        # ── tracce impronte ───────────────────────────────────────────────
        for trail, col in [(self.foot_trail_l, lc), (self.foot_trail_r, rc)]:
            n = len(trail)
            for i in range(1, n):
                alpha = int((i / n) * 90)
                cs = (*col, alpha)
                s = pygame.Surface((6,6), pygame.SRCALPHA)
                pygame.draw.ellipse(s, cs, (0,0,6,6))
                surf.blit(s, (int(trail[i][0])-3, int(trail[i][1])-3))

        # ── connettori corpo-piedi ─────────────────────────────────────────
        fl = self._foot_pos(-1)
        fr = self._foot_pos(+1)
        for fp in [fl, fr]:
            pygame.draw.line(surf, (30, 45, 70),
                (int(self.x), int(self.y)), (int(fp[0]), int(fp[1])), 1)

        # ── piedi ─────────────────────────────────────────────────────────
        lam_l = triangle(self.t / STEP_T)
        lam_r = triangle(self.t / STEP_T + 0.5)

        for fp, col, lam in [(fl, lc, lam_l), (fr, rc, lam_r)]:
            sc = 0.75 + 0.25 * (1 - abs(lam))
            fw = max(2, int(8 * sc))
            fh = max(3, int(14 * sc))
            # ombra
            s = pygame.Surface((fw*2+4, fh*2+4), pygame.SRCALPHA)
            pygame.draw.ellipse(s, (*col, 40), (2,2, fw*2, fh*2))
            surf.blit(s, (int(fp[0])-fw-2+2, int(fp[1])-fh-2+2))
            # piede
            rect = pygame.Rect(int(fp[0])-fw, int(fp[1])-fh, fw*2, fh*2)
            pygame.draw.ellipse(surf, col, rect)
            bright = tuple(min(255, c + 70) for c in col)
            pygame.draw.ellipse(surf, bright, rect, 1)

        # ── corpo ─────────────────────────────────────────────────────────
        bx, by = int(self.x), int(self.y)
        pygame.draw.circle(surf, (20, 28, 42), (bx+2, by+2), AGENT_RADIUS)
        pygame.draw.circle(surf, bc, (bx, by), AGENT_RADIUS)
        bright_bc = tuple(min(255, c+80) for c in bc)
        pygame.draw.circle(surf, bright_bc, (bx, by), AGENT_RADIUS, 2)
        # direzione
        ex = int(bx + math.cos(self.angle) * (AGENT_RADIUS + 8))
        ey = int(by + math.sin(self.angle) * (AGENT_RADIUS + 8))
        pygame.draw.line(surf, ACCENT_C, (bx, by), (ex, ey), 2)
        pygame.draw.circle(surf, ACCENT_C, (ex, ey), 3)

        # etichetta
        if self.label:
            try:
                f = pygame.font.SysFont("consolas", 11)
            except:
                f = pygame.font.SysFont(None, 13)
            s = f.render(self.label, True, TEXT_C)
            surf.blit(s, (bx - s.get_width()//2, by - AGENT_RADIUS - 16))


# ── SCENARI ──────────────────────────────────────────────────────────────────
def make_scenario(idx):
    """Restituisce (agents, obstacles, title, description)"""
    cx, cy = WIDTH//2, HEIGHT//2
    rm = ROOM_MARGIN

    if idx == 1:
        # ── MOTO RETTILINEO ───────────────────────────────────────────────
        agents = []
        rows = 3
        for r in range(rows):
            y  = cy - 80 + r * 80
            a  = Agent(rm + 40, y, 0.0, AGENT_SPEED * random.uniform(0.85,1.15),
                       color_idx=r, label=f"P{r+1}")
            a.set_waypoints([(WIDTH-rm-40, y)], loop=False)
            agents.append(a)
        return agents, [], "1 — Moto rettilineo", \
               "Tre persone camminano in linea retta parallela"

    elif idx == 2:
        # ── MOTO CURVILINEO (percorso a S + cerchio) ──────────────────────
        agents = []
        # percorso a S
        wps_s = []
        for i in range(20):
            frac = i / 19
            x = rm+60 + frac*(WIDTH-2*rm-120)
            y = cy + math.sin(frac * math.pi * 2) * 150
            wps_s.append((x, y))
        a1 = Agent(rm+60, cy, 0.0, AGENT_SPEED, color_idx=0, label="S-curve")
        a1.set_waypoints(wps_s, loop=False)
        agents.append(a1)

        # percorso a cerchio
        wps_c = []
        n_pts = 24
        for i in range(n_pts+1):
            ang = i / n_pts * 2 * math.pi
            wps_c.append((cx + math.cos(ang)*160, cy + math.sin(ang)*130))
        a2 = Agent(cx+160, cy, math.pi/2, AGENT_SPEED*0.9, color_idx=2, label="cerchio")
        a2.set_waypoints(wps_c, loop=True)
        agents.append(a2)

        # zig-zag
        wps_z = [(rm+60 + i*90, cy - 200 + (i%2)*100) for i in range(10)]
        a3 = Agent(rm+60, cy-200, 0.0, AGENT_SPEED*1.1, color_idx=4, label="zig-zag")
        a3.set_waypoints(wps_z, loop=False)
        agents.append(a3)

        return agents, [], "2 — Moto curvilineo", \
               "S-curve, cerchio continuo, zig-zag"

    elif idx == 3:
        # ── OSTACOLI ─────────────────────────────────────────────────────
        obs = [
            Obstacle(cx-200, cy-120, 80, 80, "A"),
            Obstacle(cx-30,  cy-60,  80, 120, "B"),
            Obstacle(cx+120, cy-100, 90, 70, "C"),
            Obstacle(cx-120, cy+60,  70, 80, "D"),
            Obstacle(cx+60,  cy+80,  100, 60, "E"),
        ]
        agents = []
        starts = [
            (rm+40, rm+80, WIDTH-rm-40, HEIGHT-rm-80),
            (rm+40, HEIGHT-rm-80, WIDTH-rm-40, rm+80),
            (rm+40, cy, WIDTH-rm-40, cy+50),
        ]
        for i, (sx,sy,ex,ey) in enumerate(starts):
            a = Agent(sx, sy, 0.0, AGENT_SPEED, color_idx=i, label=f"P{i+1}")
            a.set_waypoints([(ex,ey)], loop=False)
            agents.append(a)
        return agents, obs, "3 — Ostacoli", \
               "Le persone navigano attorno agli ostacoli"

    elif idx == 4:
        # ── DUE PERSONE CHE SI INCONTRANO ────────────────────────────────
        mid_y = cy
        y_off = [-80, 0, 80]
        agents = []
        for i, dy in enumerate(y_off):
            # da sinistra a destra
            a = Agent(rm+50, mid_y+dy, 0.0, AGENT_SPEED, color_idx=i, label=f"A{i+1}")
            a.set_waypoints([(WIDTH-rm-50, mid_y+dy + random.randint(-30,30))], loop=False)
            agents.append(a)
            # da destra a sinistra
            b = Agent(WIDTH-rm-50, mid_y+dy, math.pi, AGENT_SPEED, color_idx=i+3, label=f"B{i+1}")
            b.set_waypoints([(rm+50, mid_y+dy + random.randint(-30,30))], loop=False)
            agents.append(b)
        return agents, [], "4 — Incontro e schivata", \
               "Due gruppi che si avvicinano e si adattano"

    elif idx == 5:
        # ── FOLLA ─────────────────────────────────────────────────────────
        agents = []
        obs = [
            Obstacle(cx-50, cy-50, 100, 100, "blocco"),
        ]
        # persone in arrivo da tutti i lati verso punti random
        for i in range(12):
            side = i % 4
            if side == 0:  sx,sy,ang = rm+40, random.randint(rm+60,HEIGHT-rm-60), 0.0
            elif side==1:  sx,sy,ang = random.randint(rm+60,WIDTH-rm-60), rm+40, math.pi/2
            elif side==2:  sx,sy,ang = WIDTH-rm-40, random.randint(rm+60,HEIGHT-rm-60), math.pi
            else:          sx,sy,ang = random.randint(rm+60,WIDTH-rm-60), HEIGHT-rm-40, -math.pi/2

            gx = random.randint(rm+80, WIDTH-rm-80)
            gy = random.randint(rm+80, HEIGHT-rm-80)
            a  = Agent(sx, sy, ang, AGENT_SPEED*random.uniform(0.7,1.3),
                       color_idx=i, label=f"{i+1}")
            a.set_waypoints([(gx,gy)], loop=False)
            agents.append(a)
        return agents, obs, "5 — Folla con collision avoidance", \
               "12 persone da direzioni diverse si evitano"

    return [], [], "?", ""


# ── DISEGNO STANZA ────────────────────────────────────────────────────────────
def draw_room(surf, obstacles):
    rm = ROOM_MARGIN
    rw = WIDTH  - 2*rm
    rh = HEIGHT - 2*rm

    # pavimento
    pygame.draw.rect(surf, ROOM_BG, (rm, rm, rw, rh))

    # griglia interna
    for gx in range(rm, WIDTH-rm+1, 50):
        pygame.draw.line(surf, GRID_C, (gx, rm), (gx, HEIGHT-rm), 1)
    for gy in range(rm, HEIGHT-rm+1, 50):
        pygame.draw.line(surf, GRID_C, (rm, gy), (WIDTH-rm, gy), 1)

    # muri
    pygame.draw.rect(surf, WALL_C, (rm, rm, rw, rh), 3)
    # angoli decorativi
    for cx2, cy2 in [(rm,rm),(WIDTH-rm,rm),(rm,HEIGHT-rm),(WIDTH-rm,HEIGHT-rm)]:
        pygame.draw.circle(surf, WALL_C, (cx2, cy2), 6)

    # ostacoli
    for obs in obstacles:
        r = obs.rect()
        pygame.draw.rect(surf, OBSTACLE_C, r, border_radius=6)
        pygame.draw.rect(surf, OBSTACLE_HL, r, 2, border_radius=6)
        if obs.label:
            try: f = pygame.font.SysFont("consolas", 13, bold=True)
            except: f = pygame.font.SysFont(None, 15)
            s = f.render(obs.label, True, (80, 110, 150))
            surf.blit(s, (r.centerx - s.get_width()//2,
                          r.centery - s.get_height()//2))


def draw_hud(surf, title, desc, scenario_idx, paused, font_lg, font_md, font_sm):
    # titolo scenario
    panel_w = 420
    draw_rounded_rect_alpha(surf, PANEL_C, (ROOM_MARGIN, 4, panel_w, 50), 200, 8)
    pygame.draw.rect(surf, WALL_C, (ROOM_MARGIN, 4, panel_w, 50), border_radius=8, width=1)
    s = font_lg.render(title, True, ACCENT_C)
    surf.blit(s, (ROOM_MARGIN + 12, 10))
    s2 = font_sm.render(desc, True, DIM_C)
    surf.blit(s2, (ROOM_MARGIN + 12, 32))

    # tasti scenario
    keys_x = WIDTH - ROOM_MARGIN - 260
    draw_rounded_rect_alpha(surf, PANEL_C, (keys_x, 4, 258, 50), 200, 8)
    pygame.draw.rect(surf, WALL_C, (keys_x, 4, 258, 50), border_radius=8, width=1)
    labels = ["1:rett.", "2:curv.", "3:obs.", "4:inc.", "5:folla"]
    for i, lbl in enumerate(labels):
        x  = keys_x + 8 + i * 50
        col = ACCENT_C if (i+1)==scenario_idx else DIM_C
        s  = font_sm.render(lbl, True, col)
        surf.blit(s, (x, 20))

    # pausa
    if paused:
        s = font_lg.render("⏸ PAUSA — SPAZIO per riprendere", True, ACCENT_C)
        surf.blit(s, (WIDTH//2 - s.get_width()//2, HEIGHT//2 - 14))

    # istruzioni
    s = font_sm.render("SPAZIO pausa  |  R reset  |  1-5 scenario  |  ESC esci",
                        True, (50, 70, 100))
    surf.blit(s, (WIDTH//2 - s.get_width()//2, HEIGHT - 22))


def draw_rounded_rect_alpha(surf, color, rect, alpha, radius):
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(s, (*color, alpha), (0, 0, rect[2], rect[3]), border_radius=radius)
    surf.blit(s, (rect[0], rect[1]))


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Simulazione Camminata 2D")
    clock  = pygame.time.Clock()

    try:
        font_lg = pygame.font.SysFont("consolas", 16, bold=True)
        font_md = pygame.font.SysFont("consolas", 13)
        font_sm = pygame.font.SysFont("consolas", 11)
    except:
        font_lg = pygame.font.SysFont(None, 20)
        font_md = pygame.font.SysFont(None, 16)
        font_sm = pygame.font.SysFont(None, 14)

    scenario_idx = 1
    agents, obstacles, title, desc = make_scenario(scenario_idx)
    paused = False

    while True:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_r:
                    agents, obstacles, title, desc = make_scenario(scenario_idx)
                    paused = False
                for k, idx in [(pygame.K_1,1),(pygame.K_2,2),(pygame.K_3,3),
                                (pygame.K_4,4),(pygame.K_5,5)]:
                    if event.key == k:
                        scenario_idx = idx
                        agents, obstacles, title, desc = make_scenario(idx)
                        paused = False

        # ── riavvio automatico scenario finito ────────────────────────────
        if all(not a.active for a in agents) and agents:
            pygame.time.wait(1200)
            agents, obstacles, title, desc = make_scenario(scenario_idx)

        # ── update ────────────────────────────────────────────────────────
        if not paused:
            for agent in agents:
                agent.update(dt, obstacles, agents)

        # ── render ────────────────────────────────────────────────────────
        screen.fill(BG)
        draw_room(screen, obstacles)

        # vettori direzione desiderata (debug leggero)
        for agent in agents:
            if agent.active and agent.waypoints and agent.wp_idx < len(agent.waypoints):
                wx, wy = agent.waypoints[agent.wp_idx]
                dx, dy = wx - agent.x, wy - agent.y
                dist   = math.hypot(dx, dy)
                if dist > 5:
                    nx, ny = dx/dist, dy/dist
                    end_x  = int(agent.x + nx * min(dist, 50))
                    end_y  = int(agent.y + ny * min(dist, 50))
                    s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                    pygame.draw.line(s, (*agent.colors[0], 30),
                        (int(agent.x), int(agent.y)), (end_x, end_y), 1)
                    screen.blit(s, (0,0))

        for agent in agents:
            agent.draw(screen)

        draw_hud(screen, title, desc, scenario_idx, paused, font_lg, font_md, font_sm)

        pygame.display.flip()


if __name__ == "__main__":
    main()