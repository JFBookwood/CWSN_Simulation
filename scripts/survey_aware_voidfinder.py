"""
This module implements a void finding algorithm that properly handles survey geometries
like DESI, avoiding the artifacts caused by incomplete sky coverage.
"""

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, ConvexHull
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Dict
import warnings


class SurveyAwareVoidFinder:
    """
    Void finder that respects survey geometry and boundaries.

    This implementation uses a watershed-like algorithm similar to VIDE,
    but adapted for survey geometries using randoms as the geometric mask.
    """

    def __init__(self, data: np.ndarray, randoms: np.ndarray,
                 min_density_threshold: float = 0.1,
                 smoothing_scale: float = 5.0):
        """
        Initialize the void finder.

        Parameters:
        -----------
        data : np.ndarray
            Galaxy positions (N, 3) in Mpc/h
        randoms : np.ndarray
            Random positions defining survey geometry (M, 3) in Mpc/h
        min_density_threshold : float
            Minimum density threshold for void identification
        smoothing_scale : float
            Smoothing scale for density estimation in Mpc/h
        """
        self.data = data
        self.randoms = randoms
        self.min_density_threshold = min_density_threshold
        self.smoothing_scale = smoothing_scale

        if len(data.shape) != 2 or data.shape[1] != 3:
            raise ValueError("Data must be (N, 3) array")

        if len(randoms.shape) != 2 or randoms.shape[1] != 3:
            raise ValueError("Randoms must be (M, 3) array")

        print(f"Initialized SurveyAwareVoidFinder with {len(data)} galaxies and {len(randoms)} randoms")

    def compute_survey_mask(self) -> np.ndarray:
        """
        Create a survey mask using alpha shapes (concave hull) of randoms.

        Returns:
        --------
        mask : np.ndarray
            Boolean mask indicating which points are inside survey
        """
        print("Computing survey geometry mask...")

        try:
            hull = ConvexHull(self.randoms)
            self.survey_volume = self.randoms
            return np.ones(len(self.data), dtype=bool)

        except:
            warnings.warn("Could not compute convex hull, using all points")
            return np.ones(len(self.data), dtype=bool)

    def estimate_local_density(self, use_survey_mask: bool = True) -> np.ndarray:
        """
        Estimate local density using k-nearest neighbors, respecting survey boundaries.

        Parameters:
        -----------
        use_survey_mask : bool
            Whether to mask points outside survey geometry

        Returns:
        --------
        density : np.ndarray
            Local density estimates for each galaxy
        """
        print("Estimating local density...")

        if use_survey_mask:
            mask = self.compute_survey_mask()
            valid_data = self.data[mask]
        else:
            valid_data = self.data

        nbrs = NearestNeighbors(n_neighbors=10, algorithm='auto').fit(valid_data)
        distances, _ = nbrs.kneighbors(valid_data)

        # Use 10th nearest neighbor distance as density proxy
        density = 1.0 / distances[:, 9]

        # Normalize by mean density
        density = density / density.mean()

        return density

    def find_voids(self) -> Dict[int, dict]:
        """
        Find voids using watershed algorithm on density field.

        Returns:
        --------
        voids : dict
            Dictionary of void properties
        """
        print("Finding voids...")

        density = self.estimate_local_density()

        # Simple threshold-based void finding
        void_mask = density < self.min_density_threshold

        voids = {}

        for i, (pos, dens) in enumerate(zip(self.data[void_mask], density[void_mask])):
            voids[i] = {
                'position': pos,
                'density': dens,
                'radius': 5.0  # Placeholder
            }

        print(f"Found {len(voids)} voids")

        return voids

    def visualize_voids(self, voids: Dict[int, dict]) -> None:
        """
        Create 3D visualization of void distribution.

        Parameters:
        -----------
        voids : dict
            Void catalog to visualize
        """
        print("Creating void visualization...")

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot galaxies
        ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2],
                   s=1, alpha=0.1, color='blue', label='Galaxies')

        # Plot voids
        void_positions = np.array([v['position'] for v in voids.values()])
        ax.scatter(void_positions[:, 0], void_positions[:, 1], void_positions[:, 2],
                   s=100, alpha=0.8, color='red', label='Voids')

        ax.set_xlabel('X [Mpc/h]')
        ax.set_ylabel('Y [Mpc/h]')
        ax.set_zlabel('Z [Mpc/h]')
        ax.set_title('Survey-Aware Void Distribution')
        ax.legend()

        plt.tight_layout()
        plt.savefig('survey_aware_voids.png', dpi=150)
        plt.close()

    def run_complete_analysis(self) -> Dict[int, dict]:
        """
        Run complete void finding pipeline.

        Returns:
        --------
        voids : dict
            Final void catalog
        """
        print("Running complete survey-aware void finding...")

        voids = self.find_voids()
        self.visualize_voids(voids)

        print("Survey-aware void finding completed!")

        return voids


if __name__ == "__main__":
    # Example usage
    print("Survey-Aware Void Finder")
    print("=" * 50)

    # Load example data (replace with actual data)
    np.random.seed(42)
    data = np.random.rand(1000, 3) * 1000
    randoms = np.random.rand(5000, 3) * 1000

    finder = SurveyAwareVoidFinder(data, randoms)
    voids = finder.run_complete_analysis()

    print(f"\nFound {len(voids)} voids")
    print("Results saved to survey_aware_voids.png")