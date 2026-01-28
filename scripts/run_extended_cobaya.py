"""
Extended Cobaya Sampling Script for Windows
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("c:/Users/Jesse/Desktop/Experimente/CWSN")
COBAYA_PACK = ROOT / "data/desi/edr/processed/voidfinder/vgcf/cobaya_pack"
TEMPLATE_YAML = COBAYA_PACK / "example_cosmo_quick_camb.yaml"


def create_extended_yaml():
    """Create YAML with extended sampling for better convergence."""
    with open(TEMPLATE_YAML, "r") as f:
        content = f.read()

    modifications = {
        "max_samples: 20000": "max_samples: 50000",
        "Rminus1_stop: 0.02": "Rminus1_stop: 0.01",
        "burn_in: 500": "burn_in: 1000",
        "chains_cosmo_quick_camb_floor.covmat": "chains_cosmo_quick_camb_ncdm3.covmat",
        'output: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/chains_cosmo_quick_camb"': 'output: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/chains_cosmo_quick_camb_extended"',
    }

    for old, new in modifications.items():
        content = content.replace(old, new)

    content = content.replace("    nchains: 4\n", "")
    extended_yaml = COBAYA_PACK / "example_cosmo_quick_camb_extended.yaml"

    with open(extended_yaml, "w") as f:
        f.write(content)

    print(f"Created extended YAML: {extended_yaml}")
    return extended_yaml


def run_extended_cobaya():
    """Run Cobaya with extended sampling."""
    print("=" * 70)
    print("RUNNING EXTENDED COBAYA SAMPLING")
    print("=" * 70)

    extended_yaml = create_extended_yaml()
    print(f"Running extended MCMC with 50,000 samples...")
    print(f"Output: chains_cosmo_quick_camb_extended.*")
    print()

    try:
        result = subprocess.run(
            [sys.executable, "-m", "cobaya.run", str(extended_yaml), "--allow-changes"],
            capture_output=False,
            cwd=COBAYA_PACK,
        )

        if result.returncode == 0:
            print("\n" + "=" * 70)
            print("EXTENDED COBAYA SAMPLING COMPLETED SUCCESSFULLY!")
            print("=" * 70)
            print("Output files:")
            print("  chains_cosmo_quick_camb_extended.1.txt (MCMC samples)")
            print("  chains_cosmo_quick_camb_extended.progress (convergence info)")
            print("  chains_cosmo_quick_camb_extended_corner.png (posterior plot)")
            print()
            print("Next steps:")
            print("1. Run diagnostics: python scripts/run_diagnostics_and_propose.py")
            print("2. Compare results: python scripts/compare_cobaya_results.py")
            return True
        else:
            print(f"Cobaya failed with return code: {result.returncode}")
            return False
    except Exception as e:
        print(f"Error running extended Cobaya: {e}")
        return False


if __name__ == "__main__":
    success = run_extended_cobaya()
    if success:
        print("\nExtended sampling completed - check results in cobaya_pack directory")
    else:
        print("\nExtended sampling failed - check error messages above")
