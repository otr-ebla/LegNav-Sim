import random
import math

class Scenarios:
    
    @staticmethod
    def parallel_traffic(w, h, n_humans):
        """
        Robot: Starts Bottom -> Goal Top.
        Humans: Start Top -> Goal Bottom (Continuous Flow).
        """
        # Robot starts at bottom center, facing UP (pi/2)
        rob_start = [w/2, 1.5, 1.57] 
        rob_goal  = [w/2, h-1.5]

        states, goals = [], []
        for _ in range(n_humans):
            # Humans spawn at the TOP area (near robot goal)
            # We add randomness to Y so they don't spawn in a single line
            sx = random.uniform(1.0, w-1.0)
            sy = random.uniform(h-3.0, h-1.0)
            
            # Goal is at the BOTTOM (same X to create lanes, or slightly random)
            gx = sx 
            gy = 1.0
            
            # State: [x, y, vx, vy, theta, w]
            # Theta = -pi/2 (facing DOWN against the robot)
            states.append([sx, sy, 0.0, 0.0, -1.57, 0.0])
            goals.append([gx, gy])
            
        return rob_start, rob_goal, states, goals

    @staticmethod
    def perpendicular_crossing(w, h, n_humans):
        """
        Room: 11x11 usually.
        Robot: Starts Bottom (Random X) -> Goal Top (Random X).
        Humans: Move Left <-> Right (Wall to Wall).
        """
        # --- 1. Robot Setup ---
        # Start: Basso (y=1.0), X Random (Buffer laterale 1.5m)
        rx_start = random.uniform(1.5, w-1.5)
        rob_start = [rx_start, 1.0, 1.57] # 1.57 = 90° (Guarda SU)
        
        # Goal: Alto (y=h-1.0), X Random
        gx_goal = random.uniform(1.5, w-1.5)
        rob_goal = [gx_goal, h-1.0]
        
        # --- 2. Humans Setup ---
        states, goals = [], []
        for i in range(n_humans):
            # Y randomica su quasi tutta l'altezza (da 1.0 a h-1.0)
            # Robot deve schivare flussi a varie altezze
            y_pos = random.uniform(1.0, h-1.0)
            
            # X Start/Goal vicinissimi ai muri (0.6m bordo, considerando raggio 0.3)
            # Alterniamo le direzioni
            if i % 2 == 0:
                # Flusso: Sinistra -> Destra
                sx, gx = 0.6, w-0.6
                theta = 0.0 # 0° (Guarda Destra)
            else:
                # Flusso: Destra -> Sinistra
                sx, gx = w-0.6, 0.6
                theta = 3.14 # 180° (Guarda Sinistra)
                
            states.append([sx, y_pos, 0.0, 0.0, theta, 0.0])
            goals.append([gx, y_pos])
            
        return rob_start, rob_goal, states, goals
    
    @staticmethod
    def circular_crossing(w, h, n_humans):
        """
        Room: 12x12 (Ideal).
        Robot: Starts Bottom -> Goal Top.
        Humans: Spawn on circle edge, goal is opposite side. Patrol (Back & Forth).
        """
        cx, cy = w / 2, h / 2
        radius = min(w, h) / 2 - 0.75 # Lascia 1.5m di margine dai bordi
        
        # --- 1. Robot Setup ---
        # Start in basso, Goal in alto (passando per il centro caotico)
        rob_start = [cx, 1.5, 1.57] 
        rob_goal  = [cx, h-1.5]
        
        # --- 2. Humans Setup ---
        states, goals = [], []
        
        for i in range(n_humans):
            # Angolo distribuito uniformemente + rumore
            angle = (2 * math.pi * i) / n_humans
            angle += random.uniform(-0.1, 0.1) # Piccolo rumore angolare
            
            # Start Point (su bordo cerchio)
            sx = cx + radius * math.cos(angle)
            sy = cy + radius * math.sin(angle)
            
            # Goal Point (diametralmente opposto)
            gx = cx + radius * math.cos(angle + math.pi)
            gy = cy + radius * math.sin(angle + math.pi)
            
            # Aggiungi Rumore Posizionale (X, Y)
            noise_s = 0.5
            sx += random.uniform(-noise_s, noise_s)
            sy += random.uniform(-noise_s, noise_s)
            
            noise_g = 0.5
            gx += random.uniform(-noise_g, noise_g)
            gy += random.uniform(-noise_g, noise_g)
            
            # Orientamento iniziale verso il centro
            theta = angle + math.pi 
            
            states.append([sx, sy, 0.0, 0.0, theta, 0.0])
            goals.append([gx, gy])
            
        return rob_start, rob_goal, states, goals

    @staticmethod
    def random_static(w, h, n_humans):
        """
        Robot: Random valid point.
        Humans: Random valid points.
        """
        # Placeholders, actual validation is done in env._setup_scenario
        rob_start = [2.0, 2.0, 0.0]
        rob_goal  = [w-2.0, h-2.0]
        
        states, goals = [], []
        for _ in range(n_humans):
            states.append([random.uniform(1, w-1), random.uniform(1, h-1), 0,0,0,0])
            goals.append([random.uniform(1, w-1), random.uniform(1, h-1)])
            
        return rob_start, rob_goal, states, goals