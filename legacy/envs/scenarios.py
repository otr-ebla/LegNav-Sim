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
        Obstacles: Rettangoli casuali.
        """
        rob_start = [2.0, 2.0, 1.57]
        rob_goal  = [w-2.0, h-2.0]
        
        states, goals = [], []
        for _ in range(n_humans):
            states.append([random.uniform(1, w-1), random.uniform(1, h-1), 0,0,0,0])
            goals.append([random.uniform(1, w-1), random.uniform(1, h-1)])
        
        # Generazione ostacoli rettangolari invece di circolari
        obs_list = []
        for _ in range(5):
            cx, cy = random.uniform(2, w-2), random.uniform(2, h-2)
            rw, rh = random.uniform(0.4, 1.0), random.uniform(0.4, 1.0)
            obs_list.append({
                "type": "rect", 
                "xmin": cx - rw/2, "xmax": cx + rw/2, 
                "ymin": cy - rh/2, "ymax": cy + rh/2
            })
            
        return rob_start, rob_goal, states, goals, obs_list
        
    @staticmethod
    def bottleneck(w, h, n_humans, gap_center_x=None):
        """
        Room: Divisa da un muro a metà (Y=h/2) con buco in gap_center_x.
        Robot: Start Basso -> Goal Alto.
        Humans: Start Alto -> Waypoint(Gap) -> Goal(Basso).
        """
        if gap_center_x is None: gap_center_x = w / 2
        # Robot parte in basso, goal in alto
        # Nota: Il robot deve capire da solo come passare, gli diamo solo il goal finale
        rob_start = [w/2, 1.5, 1.57]
        rob_goal  = [gap_center_x, h-1.5]
        
        states, goals = [], []
        for i in range(n_humans):
            # Spawn in alto (Start Zone)
            sx = random.uniform(1.0, w-1.0)
            sy = random.uniform(h-2.5, h-1.5)
            
            # --- Waypoint 1: Il Varco (Gap) ---
            # Aggiungiamo un po' di rumore per non farli sovrapporre tutti in un punto
            gap_y = h / 2
            wx = gap_center_x + random.uniform(-0.1, 0.1)
            wy = gap_y - 0.15
            
            # --- Waypoint 2: Fondo Stanza (End) ---
            gx = random.uniform(1.0, w-1.0)
            gy = 1.5
            
            # State: [x, y, vx, vy, theta, w] -> Orientati verso il basso (-pi/2)
            states.append([sx, sy, 0.0, 0.0, -1.57, 0.0])
            
            # Goals è una lista di target: [Gap, End]
            goals.append([[wx, wy], [gx, gy]])
            
        return rob_start, rob_goal, states, goals

    @staticmethod
    def static_groups(w, h, n_humans):
        """
        Robot: Start Basso -> Goal Alto.
        Humans: Fermi in gruppi di 3 o 4 persone, ben distanziati.
        """
        rob_start = [w/2, 1.0, 1.57]
        rob_goal  = [w/2, h-1.0]
        
        states, goals = [], []
        group_centers = []
        humans_left = n_humans
        
        # Loop finché non abbiamo piazzato tutti gli umani
        while humans_left > 0:
            # 1. Determina dimensione gruppo (3 o 4)
            if humans_left <= 4:
                size = humans_left # L'ultimo gruppo prende il resto (potrebbe essere < 3, pazienza)
            else:
                size = random.choice([3, 4])
                # Evitiamo di lasciare 1 o 2 persone sole alla fine se possibile
                if humans_left - size < 3 and humans_left - size > 0:
                    size = 3 # Forziamo 3 per lasciare più gente al prossimo giro
            
            # 2. Trova un centro valido (Ben Distanziato)
            cx, cy = 0, 0
            valid_center = False
            for _ in range(100): # 100 tentativi per trovare spazio
                cx = random.uniform(1.0, w-1.0)
                cy = random.uniform(1.5, h-1.5) # Zona centrale
                
                # A. Distanza di sicurezza da Robot Start/Goal
                if math.hypot(cx - rob_start[0], cy - rob_start[1]) < 3.0: continue
                if math.hypot(cx - rob_goal[0], cy - rob_goal[1]) < 3.0: continue
                
                # B. Distanza dagli altri gruppi (MINIMO 3.5 metri)
                too_close = False
                for ox, oy in group_centers:
                    if math.hypot(cx - ox, cy - oy) < 3.5: 
                        too_close = True
                        break
                
                if not too_close:
                    valid_center = True
                    break

                    

            # Fallback se la stanza è troppo affollata: piazza comunque ma avvisa
            if not valid_center:
                #print("\n\nWarning: Centro valido non trovato, usando fallback.\n\n")
                cx, cy = random.uniform(0.7, w-0.7), random.uniform(0.7, h-0.7)

            group_centers.append((cx, cy))

            # 3. Posiziona gli umani attorno al centro
            for _ in range(size):
                # Disposizione radiale
                angle = random.uniform(0, 6.28)
                dist = random.uniform(0.3, 0.7) # Gruppo compatto (raggio 0.6-1.0m)
                
                px = cx + dist * math.cos(angle)
                py = cy + dist * math.sin(angle)
                
                # Orientamento casuale (conversazione)
                th = random.uniform(0, 6.28)
                
                # Velocità zero, Goal = Start
                states.append([px, py, 0.0, 0.0, th, 0.0])
                goals.append([px, py])
            
            humans_left -= size
            
        return rob_start, rob_goal, states, goals

    @staticmethod
    def intersection(w, h, n_humans):
        """
        Incrocio a 4 vie.
        Robot: Start Random in basso -> Goal Alto.
        Umani: Spawn random nel mezzo, ma Patrol tra i due ESTREMI del corridoio.
        """
        gap = 3.0
        spawn_margin = 0.5 
        
        # Limiti per spawn random (Mezzo)
        min_x_center = (w/2) - (gap/2) + spawn_margin
        max_x_center = (w/2) + (gap/2) - spawn_margin
        min_y_center = (h/2) - (gap/2) + spawn_margin
        max_y_center = (h/2) + (gap/2) - spawn_margin

        # --- 1. ROBOT SETUP ---
        rx = random.uniform(min_x_center, max_x_center)
        ry = random.uniform(1.0, 2.5)
        rt = random.uniform(0, 2 * math.pi)
        rob_start = [rx, ry, rt]
        rob_goal  = [w/2, h-1.5]
        
        # --- 2. HUMANS SETUP ---
        states, goals = [], []
        
        while len(states) < n_humans:
            side = random.choice([1, 2, 3])
            
            # Init variabili
            sx, sy, th = 0, 0, 0
            patrol_point_A, patrol_point_B = [], [] # I due estremi

            if side == 1: # DOWN (Verticale)
                # Spawn: Ovunque nel corridoio verticale
                sx = random.uniform(min_x_center, max_x_center)
                sy = random.uniform(0.5, h - 0.5)
                th = -1.57
                
                # I due estremi del pendolo
                patrol_point_A = [sx, h - 1.0] # Estremo Alto
                patrol_point_B = [sx, 1.0]     # Estremo Basso (Goal attuale)
                
            elif side == 2: # RIGHT (Orizzontale)
                sx = random.uniform(0.5, w - 0.5)
                sy = random.uniform(min_y_center, max_y_center)
                th = 0.0
                
                patrol_point_A = [1.0, sy]     # Estremo Sinistra
                patrol_point_B = [w - 1.0, sy] # Estremo Destra (Goal attuale)
                
            elif side == 3: # LEFT (Orizzontale)
                sx = random.uniform(0.5, w - 0.5)
                sy = random.uniform(min_y_center, max_y_center)
                th = 3.14
                
                patrol_point_A = [w - 1.0, sy] # Estremo Destra
                patrol_point_B = [1.0, sy]     # Estremo Sinistra (Goal attuale)

            # Check Sicurezza Robot
            if math.hypot(sx - rx, sy - ry) < 1.0: continue

            states.append([sx, sy, 0.0, 0.0, th, 0.0])
            
            # Qui passiamo entrambi i punti di patrol, non solo il goal immediato
            # goals diventa: [[P_A, P_B], [P_A, P_B], ...]
            goals.append([patrol_point_A, patrol_point_B])
            
        return rob_start, rob_goal, states, goals