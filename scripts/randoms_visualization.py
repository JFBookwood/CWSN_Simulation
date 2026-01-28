"""
This script visualizes the survey geometry of DESI data to identify potential issues
with void finding algorithms. DESI has a characteristic "survey geometry" with
radial spokes that can cause false voids between survey arms.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

ROOT=Path(__file__).parent.parent
DATA_DIR=ROOT/'data'/'desi'/'edr'/'processed'/'vide_input'/'mpc'
DEFAULT_FILE=DATA_DIR/'randoms.txt'

def load_data(file_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load XYZ data from file."""
    print(f"Lade Daten: {file_path}")
    data = np.loadtxt(file_path)

    if data.ndim == 1:
        data = data.reshape(-1, 3)

    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    print(f"Anzahl Punkte: {len(x):,}")
    print(f"X-Bereich: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Y-Bereich: [{y.min():.2f}, {y.max():.2f}]")
    print(f"Z-Bereich: [{z.min():.2f}, {z.max():.2f}]")

    return x, y, z

def subsample_data(x: np.ndarray, y: np.ndarray, z: np.ndarray, n_sample: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subsample data for visualization."""
    if len(x) <= n_sample:
        return x, y, z

    print(f"\nSample {n_sample:,} zufällige Punkte für Visualisierung...")
    idx = np.random.choice(len(x), n_sample, replace=False)
    return x[idx], y[idx], z[idx]

def plot_survey_geometry(x: np.ndarray, y: np.ndarray, z: np.ndarray, output_file: Path = None):
    """Create comprehensive survey geometry plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('DESI Survey Geometry Analysis', fontsize=16, fontweight='bold')

    distances = np.sqrt(x**2 + y**2 + z**2)
    phi = np.arctan2(y, x)
    theta = np.arccos(z / distances)

    h1 = axes[0, 0].hist2d(x, y, bins=200, cmap='viridis', norm='log')
    axes[0, 0].set_xlabel('X [Mpc]')
    axes[0, 0].set_ylabel('Y [Mpc]')
    axes[0, 0].set_title(f'XY Projektion (alle {len(x):,} Punkte)')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(h1[3], ax=axes[0, 0], label='Dichte (log)')

    h2 = axes[0, 1].hist2d(x, z, bins=200, cmap='plasma', norm='log')
    axes[0, 1].set_xlabel('X [Mpc]')
    axes[0, 1].set_ylabel('Z [Mpc]')
    axes[0, 1].set_title(f'XZ Projektion (alle {len(x):,} Punkte)')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(h2[3], ax=axes[0, 1], label='Dichte (log)')

    h3 = axes[0, 2].hist2d(y, z, bins=200, cmap='coolwarm', norm='log')
    axes[0, 2].set_xlabel('Y [Mpc]')
    axes[0, 2].set_ylabel('Z [Mpc]')
    axes[0, 2].set_title(f'YZ Projektion (alle {len(x):,} Punkte)')
    axes[0, 2].set_aspect('equal')
    plt.colorbar(h3[3], ax=axes[0, 2], label='Dichte (log)')

    axes[1, 0].hist(distances, bins=100, alpha=0.7, edgecolor='black', density=True)
    axes[1, 0].set_xlabel('Radiale Distanz [Mpc]')
    axes[1, 0].set_ylabel('Normierte Häufigkeit')
    axes[1, 0].set_title('Verteilung der radialen Distanzen')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(distances.mean(), color='red', linestyle='--', label=f'Mittelwert: {distances.mean():.1f}')
    axes[1, 0].legend()

    axes[1, 1].hist(phi, bins=100, alpha=0.7, edgecolor='black', density=True)
    axes[1, 1].set_xlabel('Azimutwinkel φ [rad]')
    axes[1, 1].set_ylabel('Normierte Häufigkeit')
    axes[1, 1].set_title('Azimutale Winkelverteilung')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].hist(theta, bins=100, alpha=0.7, edgecolor='black', density=True)
    axes[1, 2].set_xlabel('Polarwinkel θ [rad]')
    axes[1, 2].set_ylabel('Normierte Häufigkeit')
    axes[1, 2].set_title('Polare Winkelverteilung')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Gespeichert: {output_file}")
    else:
        plt.show()

    plt.close(fig)

def plot_sampled_scatter(x: np.ndarray, y: np.ndarray, z: np.ndarray, output_file: Path = None):
    """Create scatter plots with subsampled data."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('DESI Survey Geometry - Scatter Plots', fontsize=16, fontweight='bold')

    sc1 = axes[0, 0].scatter(x, y, c=z, cmap='viridis', s=1, alpha=0.5)
    axes[0, 0].set_xlabel('X [Mpc]')
    axes[0, 0].set_ylabel('Y [Mpc]')
    axes[0, 0].set_title(f'XY Projektion ({len(x):,} Sample)')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(sc1, ax=axes[0, 0], label='Z [Mpc]')

    sc2 = axes[0, 1].scatter(x, z, c=y, cmap='plasma', s=1, alpha=0.5)
    axes[0, 1].set_xlabel('X [Mpc]')
    axes[0, 1].set_ylabel('Z [Mpc]')
    axes[0, 1].set_title(f'XZ Projektion ({len(x):,} Sample)')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(sc2, ax=axes[0, 1], label='Y [Mpc]')

    sc3 = axes[1, 0].scatter(y, z, c=x, cmap='coolwarm', s=1, alpha=0.5)
    axes[1, 0].set_xlabel('Y [Mpc]')
    axes[1, 0].set_ylabel('Z [Mpc]')
    axes[1, 0].set_title(f'YZ Projektion ({len(x):,} Sample)')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(sc3, ax=axes[1, 0], label='X [Mpc]')

    axes[1, 1].text(0.5, 0.5,
    f'Gesamt: {len(x):,} Punkte\n' +
    'DESI Survey-Geometry:\n' +
    '• Radiale Arme (Spokes)\n' +
    '• Unvollständige Abdeckung\n' +
    '• Redshift-Shells\n' +
    '• Boundary-Effekte',
    ha='center', va='center', fontsize=12,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    axes[1, 1].set_title('Survey-Geometry-Probleme')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if output_file:
        base_name = output_file.stem
        ext = output_file.suffix
        scatter_file = output_file.parent / f"{base_name}_scatter{ext}"
        fig.savefig(scatter_file, dpi=300, bbox_inches='tight')
        print(f"Gespeichert: {scatter_file}")
    else:
        plt.show()

    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description='Visualize DESI survey geometry to identify void finding issues',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DESI Survey Geometry Issues:
- Radial survey arms create artificial voids between spokes
- Incomplete sky coverage leads to boundary effects
- Redshift-distance shells cause density variations
- Traditional void finders may identify false voids

Use VIDE for proper survey geometry handling.
        """
    )
    parser.add_argument('--file', type=Path, default=DEFAULT_FILE,
        help='Path to data file')
    parser.add_argument('--sample', type=int, default=50000,
        help='Number of points to sample for scatter plots')
    parser.add_argument('--output', type=Path, default=Path("results/void_visualizations/survey_geometry"),
        help='Output directory for plots')

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    x, y, z = load_data(args.file)

    output_hist = args.output / "survey_geometry.png"
    plot_survey_geometry(x, y, z, output_hist)

    x_plot, y_plot, z_plot = subsample_data(x, y, z, args.sample)
    output_scatter = args.output / "survey_geometry.png"
    plot_sampled_scatter(x_plot, y_plot, z_plot, output_scatter)

    print("\nSurvey-Geometry-Analyse abgeschlossen!")
    print("Hinweis: Verwenden Sie VIDE für korrekte Void-Erkennung bei Survey-Geometrien.")

if __name__ == '__main__':
    main()
