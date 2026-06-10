"""Plot the 2D map from data_sim_lidar_1.npz."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

NPZ_PATH = "legnav/map/data_sim_lidar_1.npz"


def main():
    data = np.load(NPZ_PATH, allow_pickle=True)

    true_map = data["trueMap"]       # (N, 2) wall/boundary points
    obstacles = data["Obstacles"]    # object array of (K, 2) polygon vertices
    pose = data["Pose"]              # (T, 3) robot trajectory [x, y, theta]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Wall/boundary points
    ax.scatter(true_map[:, 0], true_map[:, 1], s=0.3, c="black", label="Map walls")

    # Obstacle polygons
    patches = [Polygon(obs, closed=True) for obs in obstacles]
    pc = PatchCollection(patches, facecolor="steelblue", edgecolor="navy", alpha=0.5)
    ax.add_collection(pc)

    # Robot trajectory
    ax.plot(pose[:, 0], pose[:, 1], color="red", linewidth=1.0, zorder=3, label="Robot trajectory")
    ax.scatter(pose[0, 0], pose[0, 1], c="green", s=60, zorder=4, label="Start")
    ax.scatter(pose[-1, 0], pose[-1, 1], c="orange", s=60, zorder=4, label="End")

    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Indoor Map — data_sim_lidar_1")
    ax.legend(handles=[
        mpatches.Patch(color="black", label="Map walls"),
        mpatches.Patch(facecolor="steelblue", edgecolor="navy", label="Obstacles"),
        mpatches.Patch(color="red", label="Robot trajectory"),
        mpatches.Patch(color="green", label="Start"),
        mpatches.Patch(color="orange", label="End"),
    ])
    plt.tight_layout()
    plt.savefig("legnav/map/map_plot.png", dpi=150)
    plt.show()
    print("Saved to legnav/map/map_plot.png")


if __name__ == "__main__":
    main()
