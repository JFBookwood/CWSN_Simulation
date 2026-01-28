"""
Vergleich verschiedener Void-Finder in 2D-Slices.

Dieses Skript lädt Void-Kataloge von ZOBOV, REVOLVER und VGCF
und visualisiert sie in nebeneinander liegenden 2D-Slices.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd

# Pfade zu den Void-Katalogen
DATA_DIR = Path('data/desi/edr/processed')

VOID_CATALOGS = {
    'ZOBOV': DATA_DIR / 'voidfinder/zobov_alternative_voids.txt',
    'REVOLVER': DATA_DIR / 'voidfinder/survey_aware_voids.txt',
    'VGCF': DATA_DIR / 'voidfinder/catalog.csv'
}

def load_voids(file_path, finder_name):
    """Lädt Void-Daten aus der Datei."""
    if finder_name == 'VGCF':
        # CSV mit Header
        df = pd.read_csv(file_path)
        x, y, z, radius = df['x'], df['y'], df['z'], df['radius_mpc']
    else:
        # Text-Datei
        if finder_name == 'ZOBOV':
            data = np.loadtxt(file_path, skiprows=1)
        elif finder_name == 'REVOLVER':
            data = np.loadtxt(file_path, comments='#')
        x, y, z, radius = data[:, 0], data[:, 1], data[:, 2], data[:, 3]

    return np.array(x), np.array(y), np.array(z), np.array(radius)

def filter_slice(voids, z_slice=0, thickness=100):
    """Filtert Voids in einem dünnen Slice um z_slice."""
    x, y, z, radius = voids
    mask = (z >= z_slice - thickness/2) & (z <= z_slice + thickness/2)
    return x[mask], y[mask], z[mask], radius[mask]

def plot_void_comparison(z_slice=0, thickness=100, output_file=None, max_voids=10000):
    """Erstellt Vergleichsplot der Void-Finder in 2D-Slices."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Void-Finder Vergleich: 2D-Slice bei z = {z_slice} ± {thickness/2} Mpc/h', fontsize=16)

    # Sammle alle Daten für gemeinsame Achsen
    all_x, all_y = [], []

    for name, file_path in VOID_CATALOGS.items():
        try:
            voids = load_voids(file_path, name)
            xs, ys, _, _ = filter_slice(voids, z_slice, thickness)
            all_x.extend(xs)
            all_y.extend(ys)
        except Exception as e:
            print(f"Fehler beim Laden {name}: {e}")

    if all_x and all_y:
        x_min, x_max = min(all_x) - 50, max(all_x) + 50
        y_min, y_max = min(all_y) - 50, max(all_y) + 50
    else:
        x_min, x_max, y_min, y_max = -2000, 2000, -2000, 2000

    for i, (name, file_path) in enumerate(VOID_CATALOGS.items()):
        ax = axes[i]

        try:
            voids = load_voids(file_path, name)
            x_slice, y_slice, z_slice_data, r_slice = filter_slice(voids, z_slice, thickness)

            print(f"{name}: {len(x_slice)} Voids im Slice")

            # Subsample wenn zu viele
            if len(x_slice) > max_voids:
                idx = np.random.choice(len(x_slice), max_voids, replace=False)
                x_slice, y_slice, r_slice = x_slice[idx], y_slice[idx], r_slice[idx]
                print(f"  Subsampled auf {max_voids} Voids")

            # Scatter plot: Punkte farbkodiert nach Radius
            sc = ax.scatter(x_slice, y_slice, c=r_slice, cmap='viridis', s=10, alpha=0.7, edgecolors='none')
            plt.colorbar(sc, ax=ax, label='Void Radius [Mpc/h]', shrink=0.8)

            ax.set_xlabel('X [Mpc/h]')
            ax.set_ylabel('Y [Mpc/h]')
            ax.set_title(f'{name}\n({len(x_slice)} Voids)')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

        except Exception as e:
            print(f"Fehler beim Laden {name}: {e}")
            ax.text(0.5, 0.5, f'Fehler: {name}', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name}\n(Fehler)')

    plt.tight_layout()

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Gespeichert: {output_file}")
    else:
        plt.show()

    plt.close(fig)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Vergleiche Void-Finder in 2D-Slices')
    parser.add_argument('--z_slice', type=float, default=0, help='Z-Position des Slices [Mpc/h]')
    parser.add_argument('--thickness', type=float, default=100, help='Dicke des Slices [Mpc/h]')
    parser.add_argument('--max_voids', type=int, default=5000, help='Max Anzahl Voids pro Plot')
    parser.add_argument('--output', type=Path, default=Path('results/void_visualizations/void_comparison_2d.png'),
                        help='Ausgabedatei')

    args = parser.parse_args()

    plot_void_comparison(args.z_slice, args.thickness, args.output, args.max_voids)

    print("Void-Finder-Vergleich abgeschlossen!")

if __name__ == '__main__':
    main()