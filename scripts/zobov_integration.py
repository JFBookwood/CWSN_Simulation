"""
ZOBOV Void Finder Integration
"""

import numpy as np
import pandas as pd
import subprocess
import os
from pathlib import Path
import shutil
from typing import Optional, Tuple, List

class ZOBOVIntegrator:
    def __init__(self, base_dir="data/desi/edr/processed"):
        self.base_dir = Path(base_dir)
        self.zobov_dir = self.base_dir / "zobov_analysis"
        self.zobov_dir.mkdir(exist_ok=True)
        self.data_file = self.base_dir / "vide_input" / "mpc" / "data.txt"
        self.randoms_file = self.base_dir / "vide_input" / "mpc" / "randoms.txt"

    def check_zobov_installation(self) -> bool:
        """Check if ZOBOV is installed and available."""
        try:
            result = subprocess.run(["python", "-c", "import zobov; print('ZOBOV available')"],
                                    capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except:
            return False

    def install_zobov(self):
        """Attempt to install ZOBOV."""
        print("Attempting to install ZOBOV...")
        try:
            subprocess.run(["pip", "install", "git+https://github.com/jjbuchanan/ZOBOV.git"],
                           check=True, timeout=300)
            print("ZOBOV installation completed!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"ZOBOV installation failed: {e}")
            print("Please install ZOBOV manually from: https://github.com/jjbuchanan/ZOBOV")
            return False

    def prepare_zobov_input(self, subsample_factor: int = 1) -> Tuple[Path, Path]:
        """Prepare input files for ZOBOV."""
        print(f"Preparing ZOBOV input with subsample factor: {subsample_factor}")
        data = np.loadtxt(self.data_file)
        if subsample_factor > 1:
            idx = np.random.choice(len(data), len(data) // subsample_factor, replace=False)
            data = data[idx]
        randoms = np.loadtxt(self.randoms_file)
        if subsample_factor > 1:
            idx = np.random.choice(len(randoms), len(randoms) // subsample_factor, replace=False)
            randoms = randoms[idx]
        zobov_data_file = self.zobov_dir / "zobov_data.txt"
        zobov_randoms_file = self.zobov_dir / "zobov_randoms.txt"
        np.savetxt(zobov_data_file, data, fmt='%.6f')
        np.savetxt(zobov_randoms_file, randoms, fmt='%.6f')
        print(f"Prepared {len(data)} galaxies and {len(randoms)} randoms for ZOBOV")
        return zobov_data_file, zobov_randoms_file

    def run_zobov(self, data_file: Path, randoms_file: Path) -> Optional[Path]:
        """Run ZOBOV void finder."""
        print("Running ZOBOV void finder...")
        try:
            import zobov
            os.chdir(self.zobov_dir)
            data = np.loadtxt(data_file)
            randoms = np.loadtxt(randoms_file)
            print(f"Input: {len(data)} galaxies, {len(randoms)} randoms")
            zones, voids, zone_volumes, void_volumes = zobov.run_zobov(
                data, randoms,
                box_size=None,
                output_dir=str(self.zobov_dir)
            )
            catalog_file = self.zobov_dir / "zobov_voids.txt"
            void_centers = []
            void_radii = []
            for void_id, void_data in voids.items():
                center = void_data['center']
                radius = void_data['radius']
                void_centers.append(center)
                void_radii.append(radius)
            if void_centers:
                catalog_data = np.column_stack([void_centers, void_radii])
                np.savetxt(catalog_file, catalog_data, fmt='%.6f',
                           header='x y z radius', comments='')
                print(f"ZOBOV found {len(void_centers)} voids")
                print(f"Results saved to: {catalog_file}")
                return catalog_file
            else:
                print("ZOBOV found no voids")
                return None
        except ImportError:
            print("ZOBOV not available. Please install ZOBOV first.")
            print("Run: pip install git+https://github.com/jjbuchanan/ZOBOV.git")
            return None
        except Exception as e:
            print(f"ZOBOV execution failed: {e}")
            return None

    def convert_zobov_output(self, zobov_output: Path) -> Path:
        """Convert ZOBOV output to standard format."""
        try:
            data = np.loadtxt(zobov_output)
            if data.shape[1] >= 4:
                centers = data[:, :3]
                radii = data[:, 3]
            else:
                centers = data[:, :3]
                radii = np.ones(len(centers)) * 10.0
            catalog_file = self.zobov_dir / "zobov_void_catalog.csv"
            df = pd.DataFrame({
                'x': centers[:, 0],
                'y': centers[:, 1],
                'z': centers[:, 2],
                'radius_mpc': radii
            })
            df.to_csv(catalog_file, index=False)
            print(f"Converted ZOBOV output to standard format: {catalog_file}")
            return catalog_file
        except Exception as e:
            print(f"Failed to convert ZOBOV output: {e}")
            return None

    def compare_with_existing(self, zobov_catalog: Path,
                              existing_catalog: Path = None) -> dict:
        """Compare ZOBOV results with existing void catalog."""
        if existing_catalog is None:
            existing_catalog = self.base_dir / "voidfinder" / "catalog.csv"
        try:
            zobov_df = pd.read_csv(zobov_catalog)
            existing_df = pd.read_csv(existing_catalog)
            comparison = {
                'zobov_voids': len(zobov_df),
                'existing_voids': len(existing_df),
                'zobov_mean_radius': zobov_df['radius_mpc'].mean(),
                'existing_mean_radius': existing_df['radius_mpc'].mean(),
                'zobov_max_radius': zobov_df['radius_mpc'].max(),
                'existing_max_radius': existing_df['radius_mpc'].max()
            }
            print("Void Finder Comparison:")
            print(f"  ZOBOV: {comparison['zobov_voids']} voids")
            print(f"  Existing: {comparison['existing_voids']} voids")
            print(f"  ZOBOV mean radius: {comparison['zobov_mean_radius']:.2f} Mpc")
            print(f"  Existing mean radius: {comparison['existing_mean_radius']:.2f} Mpc")
            return comparison
        except Exception as e:
            print(f"Comparison failed: {e}")
            return {}

    def run_complete_zobov_analysis(self, subsample_factor: int = 1,
                                    compare: bool = True) -> Optional[Path]:
        """Run complete ZOBOV analysis pipeline."""
        print("Starting complete ZOBOV analysis...")
        if not self.check_zobov_installation():
            print("ZOBOV not found. Attempting installation...")
            if not self.install_zobov():
                print("Please install ZOBOV manually and try again.")
                return None
        data_file, randoms_file = self.prepare_zobov_input(subsample_factor)
        zobov_result = self.run_zobov(data_file, randoms_file)
        if zobov_result:
            standard_catalog = self.convert_zobov_output(zobov_result)
            if compare:
                self.compare_with_existing(standard_catalog)
            print("ZOBOV analysis completed successfully!")
            print(f"Void catalog: {standard_catalog}")
            return standard_catalog
        else:
            print("ZOBOV analysis failed.")
            return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ZOBOV Void Finder Integration')
    parser.add_argument('--subsample', type=int, default=1,
                        help='Subsample factor for input data')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with existing void catalog')
    parser.add_argument('--install', action='store_true',
                        help='Attempt to install ZOBOV first')
    args = parser.parse_args()
    integrator = ZOBOVIntegrator()
    if args.install:
        integrator.install_zobov()
    result = integrator.run_complete_zobov_analysis(
        subsample_factor=args.subsample,
        compare=args.compare
    )
    if result:
        print(f"Success! ZOBOV void catalog created: {result}")
    else:
        print("ZOBOV analysis failed. Check installation and try again.")