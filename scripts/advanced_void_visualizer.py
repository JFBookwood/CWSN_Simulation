"""
This module provides advanced visualization capabilities for void analysis,
including 3D rendering, property distributions, and comparative analysis
between different void finders.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.patches import Circle
import pandas as pd
from pathlib import Path
import seaborn as sns
import subprocess
import os
from typing import Optional, Tuple, List

# Set visualization style
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16

# Color palette for consistent visualizations
COLORS = sns.color_palette("husl", 12)

class AdvancedVoidVisualizer:
    def __init__(self, output_dir="results/void_visualizations"):
        self.output_dir = Path(output_dir) / "advanced"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = Path("data/desi/edr/processed")
        self.vide_dir = self.data_dir / "vide_input" / "mpc"
        self.void_dir = self.data_dir / "voidfinder"

    def run_vide_finder(self, data_file: Path, randoms_file: Path, output_prefix: str = "vide_voids") -> Optional[Path]:
        """Run VIDE void finder and return catalog path."""
        try:
            vide_exe = "vide"
            output_dir = self.void_dir / output_prefix
            output_dir.mkdir(exist_ok=True)

            cmd = [
                vide_exe,
                "-d", str(data_file),
                "-r", str(randoms_file),
                "-o", str(output_dir)
            ]

            print(f"Running VIDE: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                catalog_file = output_dir / "catalog.txt"
                if catalog_file.exists():
                    print(f"VIDE completed successfully: {catalog_file}")
                    return catalog_file
                else:
                    print("VIDE completed but catalog not found")
                    return None
            else:
                print(f"VIDE failed with return code {result.returncode}")
                print(f"Error: {result.stderr}")
                return None
        except Exception as e:
            print(f"Error running VIDE: {e}")
            return None

    def load_void_catalog(self, catalog_file: Path) -> pd.DataFrame:
        """Load void catalog from file."""
        try:
            df = pd.read_csv(catalog_file, delim_whitespace=True)
            print(f"Loaded {len(df)} voids from {catalog_file}")
            return df
        except Exception as e:
            print(f"Error loading catalog: {e}")
            return pd.DataFrame()

    def create_3d_void_plot(self, voids: pd.DataFrame, title: str = "3D Void Distribution") -> None:
        """Create 3D visualization of void distribution."""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot void centers
        ax.scatter(voids['x'], voids['y'], voids['z'],
                   c=voids['radius'], cmap='viridis', s=50, alpha=0.7)

        ax.set_xlabel('X [Mpc/h]')
        ax.set_ylabel('Y [Mpc/h]')
        ax.set_zlabel('Z [Mpc/h]')
        ax.set_title(title)

        # Add colorbar
        mappable = plt.cm.ScalarMappable(cmap='viridis')
        mappable.set_array(voids['radius'])
        plt.colorbar(mappable, ax=ax, label='Void Radius [Mpc/h]')

        plt.tight_layout()
        plt.savefig(self.output_dir / "3d_void_distribution.png", dpi=150)
        plt.close()

    def plot_void_properties(self, voids: pd.DataFrame) -> None:
        """Create property distribution plots."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Radius distribution
        axes[0, 0].hist(voids['radius'], bins=30, color='blue', alpha=0.7)
        axes[0, 0].set_xlabel('Radius [Mpc/h]')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Void Radius Distribution')
        axes[0, 0].grid(True, alpha=0.3)

        # Volume distribution
        axes[0, 1].hist(voids['volume'], bins=30, color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Volume [(Mpc/h)^3]')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Void Volume Distribution')
        axes[0, 1].grid(True, alpha=0.3)

        # Density contrast
        if 'density_contrast' in voids.columns:
            axes[1, 0].hist(voids['density_contrast'], bins=30, color='red', alpha=0.7)
            axes[1, 0].set_xlabel('Density Contrast')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Density Contrast Distribution')
            axes[1, 0].grid(True, alpha=0.3)

        # Radius vs Volume
        axes[1, 1].scatter(voids['radius'], voids['volume'], alpha=0.5, color='purple')
        axes[1, 1].set_xlabel('Radius [Mpc/h]')
        axes[1, 1].set_ylabel('Volume [(Mpc/h)^3]')
        axes[1, 1].set_title('Radius vs Volume')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "void_properties.png", dpi=150)
        plt.close()

    def compare_void_finders(self, vide_catalog: Path, revolver_catalog: Path) -> None:
        """Compare void catalogs from different finders."""
        try:
            vide_voids = self.load_void_catalog(vide_catalog)
            revolver_voids = self.load_void_catalog(revolver_catalog)

            if len(vide_voids) == 0 or len(revolver_voids) == 0:
                print("One or both catalogs are empty")
                return

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Radius comparison
            axes[0].hist(vide_voids['radius'], bins=30, alpha=0.5, label='VIDE', color='blue')
            axes[0].hist(revolver_voids['radius'], bins=30, alpha=0.5, label='REVOLVER', color='orange')
            axes[0].set_xlabel('Radius [Mpc/h]')
            axes[0].set_ylabel('Count')
            axes[0].set_title('Void Radius Comparison')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Volume comparison
            axes[1].hist(vide_voids['volume'], bins=30, alpha=0.5, label='VIDE', color='blue')
            axes[1].hist(revolver_voids['volume'], bins=30, alpha=0.5, label='REVOLVER', color='orange')
            axes[1].set_xlabel('Volume [(Mpc/h)^3]')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Void Volume Comparison')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(self.output_dir / "void_finder_comparison.png", dpi=150)
            plt.close()

        except Exception as e:
            print(f"Error in void finder comparison: {e}")

    def run_complete_analysis(self) -> None:
        """Run complete visualization pipeline."""
        print("Starting advanced void visualization analysis...")

        # Run VIDE if needed
        data_file = self.vide_dir / "data.txt"
        randoms_file = self.vide_dir / "randoms.txt"

        if data_file.exists() and randoms_file.exists():
            vide_catalog = self.run_vide_finder(data_file, randoms_file)
            if vide_catalog:
                vide_voids = self.load_void_catalog(vide_catalog)
                if len(vide_voids) > 0:
                    self.create_3d_void_plot(vide_voids, "VIDE Void Distribution")
                    self.plot_void_properties(vide_voids)

        # Compare with REVOLVER if available
        revolver_catalog = self.void_dir / "catalog.csv"
        if revolver_catalog.exists():
            self.compare_void_finders(vide_catalog, revolver_catalog)

        print("Advanced void visualization analysis completed!")
        print(f"Results saved in: {self.output_dir}")

if __name__ == "__main__":
    visualizer = AdvancedVoidVisualizer()
    visualizer.run_complete_analysis()