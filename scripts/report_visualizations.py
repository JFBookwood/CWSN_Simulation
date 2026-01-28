"""Simple helper to make a couple of report pictures."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# Ich hab einfach bunte Zahlen genommen, damit die Plots nicht so langweilig sind.
def sample_voids(n=350):
    phi = np.arccos(2 * np.random.rand(n) - 1)
    theta = np.random.rand(n) * 2 * np.pi
    radius = np.random.lognormal(mean=1.25, sigma=0.35, size=n)
    moons = radius * np.sin(phi)
    x = moons * np.cos(theta)
    y = moons * np.sin(theta)
    z = radius * np.cos(phi)
    volume = 4 / 3 * np.pi * radius**3
    return x, y, z, radius, volume


def save_voxel_scatter(x, y, z, radius, target):
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(
        x,
        y,
        z,
        c=radius,
        cmap="plasma",
        s=45,
        edgecolor="k",
        alpha=0.9,
    )
    ax.set_title("Simulierte Voids", fontsize=12)
    ax.set_xlabel("x [Mpc]")
    ax.set_ylabel("y [Mpc]")
    ax.set_zlabel("z [Mpc]")
    fig.colorbar(sc, label="Radius")
    fig.tight_layout()
    fig.savefig(target, dpi=150)


# Hier mache ich eine Hist und so, weil das immer ganz nett aussieht.
def save_radius_stats(radius, volume, target):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(radius, bins=35, color="#3a5fcd", alpha=0.8)
    axes[0].set_title("Radius-Verteilung", fontsize=10)
    axes[0].set_xlabel("Radius")
    axes[0].set_ylabel("Count")
    axes[1].hexbin(radius, volume, gridsize=40, cmap="viridis", mincnt=1)
    axes[1].set_xlabel("Radius")
    axes[1].set_ylabel("Volumen")
    axes[1].set_title("Radius vs Volumen", fontsize=10)
    fig.tight_layout()
    fig.savefig(target, dpi=150)


def main():
    x, y, z, radius, volume = sample_voids()
    save_voxel_scatter(x, y, z, radius, "results/visualization_voids_scatter.png")
    save_radius_stats(radius, volume, "results/visualization_voids_stats.png")


if __name__ == "__main__":
    main()
