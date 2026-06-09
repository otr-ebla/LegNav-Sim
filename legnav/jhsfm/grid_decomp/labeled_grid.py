from shapely.geometry import Polygon, box
import os
import numpy as np
from collections import defaultdict
import jax.numpy as jnp
    
import matplotlib.pyplot as plt

class GridCell_operations:
    def __init__(self, cell_size=4, world_size=320):
        self.cell_size = cell_size
        self.world_size = world_size
        self.num_cells = int(world_size / cell_size)
        self.half_world = world_size / 2
        self.static_obstacles = []

        self.labeled_grid_dict = None
        self.mesh_index = {}
        self.max_edges_per_obstacle = 4




















    # --- NEW: carica "labeled_grid_cleaned.txt" una volta ---
    def load_labeled_grid_from_file(self, path):
        d = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("Cell"):
                    continue
                head, labels = line.split(":", 1)
                ij = head.replace("Cell", "").strip()
                # tollera "i,j" o "i, j"
                i_str, j_str = [s.strip() for s in ij.split(",")]
                i, j = int(i_str), int(j_str)
                labs = [s.strip() for s in labels.strip().split("|") if s.strip() and s.strip().lower() != "free"]
                d[(i, j)] = labs
        self.labeled_grid_dict = d

    # --- NEW: indicizza "meshes.txt" una volta (name -> 4 vertici -> 4 edge) ---
    def load_meshes_index(self, path):
        self.mesh_index = {}
        with open(path, 'r') as f:
            lines = [ln.strip() for ln in f.readlines()]
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ln.endswith(":"):
                name = ln[:-1]
                verts = []
                for j in range(1, 5):
                    if i + j < len(lines):
                        x_str, y_str = [s.strip() for s in lines[i + j].split(",")]
                        verts.append((float(x_str), float(y_str)))
                if len(verts) == 4:
                    edges = [
                        (verts[0], verts[1]),
                        (verts[1], verts[2]),
                        (verts[2], verts[3]),
                        (verts[3], verts[0]),
                    ]
                    # NIENTE flatten: edge = ((x1,y1),(x2,y2)) -> (2,2)
                    self.mesh_index[name] = np.array(edges, dtype=np.float32)  # (4, 2, 2)
                i += 4
            i += 1

    def obstacles_for_names(self, names):
        """Return stacked obstacles as (n_obs, n_edges, 2, 2)."""
        obs = []
        for n in names:
            e = self.mesh_index.get(n, None)  # e: (4,2,2)
            if e is not None:
                obs.append(e)
        if not obs:
            return np.zeros((0, 4, 2, 2), dtype=np.float32)
        return np.stack(obs, axis=0)  # (n_obs, 4, 2, 2)


    # --- NEW: query 8-neighbors sfruttando la cache ---
    def get_surrounding_obstacle_names_cached(self, x, y, radius=1, grid_size=60):
        if np.isnan(x) or np.isnan(y) or self.labeled_grid_dict is None:
            return set()
        center = self.world_to_grid(x, y, square_size=self.cell_size, grid_size=grid_size)
        if center is None:
            return set()
        i0, j0 = center
        found = set()
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                key = (i0 + di, j0 + dj)
                labs = self.labeled_grid_dict.get(key)
                if labs:
                    found.update(labs)
        return found

    # --- NEW: edges per insieme di nomi (no I/O, no accumulo stato) ---
    # --- dentro edges_for_obstacle_names(...) ---
    def edges_for_obstacle_names(self, names):
        if not names:
            return np.zeros((0, 2, 2), dtype=np.float32)  # (n_edges, 2, 2)
        mats = [self.mesh_index[n] for n in names if n in self.mesh_index]  # ognuno (4,2,2)
        if not mats:
            return np.zeros((0, 2, 2), dtype=np.float32)
        return np.concatenate(mats, axis=0)  # (num_edges, 2, 2)


















    
    def parse_obstacle_file(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        obstacles = {}
        current_name = None
        current_points = []

        for line in lines:
            line = line.strip()
            if not line:
                if current_name and current_points:
                    obstacles[current_name] = Polygon(current_points)
                current_name = None
                current_points = []
            elif ':' in line:
                current_name = line.replace(':', '')
            else:
                try:
                    x, y = map(float, line.split(','))
                    current_points.append((x, y))
                except ValueError:
                    print(f"Warning: Invalid coordinate in line: {line}")
        
        # Catch last block
        if current_name and current_points:
            obstacles[current_name] = Polygon(current_points)

        return obstacles

    def label_grid(self, obstacles, world_size=320, square_size=4):
        num_cells = int(world_size / square_size)
        half_world = world_size / 2
        grid_labels = []

        for i in range(num_cells):
            for j in range(num_cells):
                x0 = -half_world + j * square_size
                y0 = -half_world + i * square_size
                cell = box(x0, y0, x0 + square_size, y0 + square_size)
                

                labels = []
                for name, obstacle in obstacles.items():
                    if obstacle.intersects(cell):
                        labels.append(name)
                if not labels:
                    labels = ["free"]

                grid_labels.append(((i, j), labels))

        return grid_labels

    def world_to_grid(self, x, y, square_size=4, grid_size=60):
        # Add safety check for NaN values
        if np.isnan(x) or np.isnan(y):
            return None  # Return None for NaN coordinates
        
        try:
            i = int((y + 160) // square_size)
            j = int((x + 160) // square_size)
            if 0 <= i < grid_size and 0 <= j < grid_size:
                return i, j
            else:
                return None  # Out of bounds
        except Exception as e:
            print(f"Error in world_to_grid with x={x}, y={y}: {e}")
            return None  # Return None on any error

    def get_surrounding_obstacles_from_world(self, x, y, labeled_grid, radius=1, grid_size=60):
        # Add safety check for NaN values
        if np.isnan(x) or np.isnan(y):
            return set()  # Return empty set for NaN coordinates
        
        center_cell = self.world_to_grid(x, y, grid_size=grid_size)
        if center_cell is None:
            return set()
        
        label_dict = dict(labeled_grid)
        i0, j0 = center_cell
        found_obstacles = set()

        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                ni, nj = i0 + di, j0 + dj
                if 0 <= ni < grid_size and 0 <= nj < grid_size:
                    labels = label_dict.get((ni, nj), ["free"])
                    for label in labels:
                        if label != "free":
                            found_obstacles.add(label)

        return found_obstacles


    def get_obstacle_vertices(self, filename, obstacle_name):
        obstacle_name = str(obstacle_name)
        try:
            with open(filename, 'r') as file:
                lines = file.readlines()
            
            vertices = []
            recording = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue  # skip empty lines

                if line.endswith(':'):  # start of a new object
                    current_name = line[:-1]
                    recording = (current_name == obstacle_name)
                    continue
                
                if recording:
                    try:
                        coords = tuple(map(float, line.split(',')))
                        vertices.append(coords)
                    except ValueError:
                        print(f"Warning: Invalid coordinate for obstacle {obstacle_name}: {line}")
                    if len(vertices) == 4:  # stop after 4 vertices
                        break
            
            if not vertices:
                print(f"Warning: Obstacle '{obstacle_name}' not found in file.")
                return None
            
            return np.array(vertices)
        except Exception as e:
            print(f"Error in get_obstacle_vertices for {obstacle_name}: {e}")
            return None
        
    def write_labeled_grid_to_file(self, labeled_grid, output_path):
        try:
            with open(output_path, 'w') as f:
                for (i, j), labels in labeled_grid:
                    label_str = '|'.join(labels)
                    f.write(f"Cell {i},{j}: {label_str}\n")
            print(f"Labeled grid successfully written to {output_path}")
        except Exception as e:
            print(f"Error writing labeled grid to file: {e}")

    def clean_label_txt_file(self, input_path, output_path):
        try:
            with open(input_path, 'r') as f:
                lines = f.readlines()

            with open(output_path, 'w') as f:
                for line in lines:
                    if "free" not in line:  # Skip lines containing "Free"
                        f.write(line)
            print(f"Cleaned label file successfully written to {output_path}")
        except Exception as e:
            print(f"Error cleaning label file: {e}")

    def get_static_obstacles_formatted(self, obstacle_names):
        static_obstacles = []

        try:
            #with open('/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/meshes.txt', 'r') as f:
            with open('/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/mesh_edges.txt', 'r') as f:
                lines = f.readlines()

            i = 0
            while i < len(lines):
                line = lines[i].strip()

                if line.endswith(":"):
                    obstacle_name = line[:-1]
                    if obstacle_name in obstacle_names:
                        # Parse 4 vertices
                        vertices = []
                        for j in range(1, 5):
                            coord_line = lines[i + j].strip()
                            x_str, y_str = coord_line.split(',')
                            vertices.append([float(x_str.strip()), float(y_str.strip())])

                        # Create 4 edges: [v0,v1], [v1,v2], [v2,v3], [v3,v0] forn1 obstacles
                        edges = [
                            [vertices[0], vertices[1]],
                            [vertices[1], vertices[2]],
                            [vertices[2], vertices[3]],
                            [vertices[3], vertices[0]],
                        ]

                        # Pad with a dummy edge to ensure consistent shape
                        # nan_edge = [[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]]
                        # edges.append(nan_edge)

                        self.static_obstacles.append(edges)
                        i += 4  # Skip the lines we've read

                i += 1

        except FileNotFoundError:
            print("meshes.txt not found")
            return None

        return jnp.array(self.static_obstacles)


    def plot_obstacles_from_static(self, static_obstacles):
        for obstacle in static_obstacles:
            for edge in obstacle:
                vertices = edge
                polygon = plt.Polygon(vertices, edgecolor='black', facecolor='none')
                plt.gca().add_patch(polygon)

        plt.axis('equal')
        plt.show()

    def get_obstacle_names_from_file(self, filepath):
        obstacle_names = []
        
        with open(filepath, 'r') as file:
            for line in file:
                obstacle_name = line.split(':')[0].strip()
                obstacle_names.append(obstacle_name)
        return obstacle_names

# ESECUZIONE DELLO SCRIPT PER RIGENERARE labeled_grid_cleaned.txt
# grid = GridCell_operations(cell_size=4, world_size=320)

# # 2) leggi gli ostacoli dal file aggiornato (stesso formato di mesh_edges.txt)
# #    Se usi proprio mesh_edges.txt, va bene anche parse_obstacle_file su quel file.
# obstacles = grid.parse_obstacle_file("/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/mesh_edges.txt")

# # 3) crea la griglia etichettata
# labeled = grid.label_grid(obstacles, world_size=320, square_size=4)

# # 4) scrivi la griglia completa e poi la versione "pulita" (senza righe 'free')
# grid.write_labeled_grid_to_file(
#     labeled,
#     "/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/labeled_grid.txt"
# )
# grid.clean_label_txt_file(
#     "/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/labeled_grid.txt",
#     "/home/alberto_vaglio/HumanAwareRLNavigation/grid_decomp/labeled_grid_cleaned.txt"
# )
# print("✅ labeled_grid_cleaned.txt rigenerato")