"""
Run Cobaya with FILTERED likelihood data (voids_vgcf_data_filtered.npz)
This uses cleaned data with 78.9% of survey artifacts removed.
Key difference: ξ₀ = +6.03 (sensible) instead of -0.018 (unphysical)
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

COBAYA_PACK = Path(r"c:\Users\Jesse\Desktop\Experimente\CWSN\data\desi\edr\processed\voidfinder\vgcf\cobaya_pack")
TEMPLATE_YAML = COBAYA_PACK / "example_cosmo_quick_camb.yaml"
FILTERED_DATA = COBAYA_PACK / "voids_vgcf_data_filtered.npz"

if not TEMPLATE_YAML.exists():
    print(f"ERROR: Template YAML not found: {TEMPLATE_YAML}")
    sys.exit(1)

if not FILTERED_DATA.exists():
    print(f"ERROR: Filtered data NPZ not found: {FILTERED_DATA}")
    sys.exit(1)

print("Running Cobaya with filtered data...")
print(f"Data file: {FILTERED_DATA}")
print()

with open(TEMPLATE_YAML, 'r') as f:
    yaml_content = f.read()

yaml_filtered = yaml_content.replace(
    'data_file: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/voids_vgcf_data.npz"',
    'data_file: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/voids_vgcf_data_filtered.npz"'
)

yaml_filtered = yaml_filtered.replace(
    'output: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/chains_cosmo_quick_camb"',
    'output: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/chains_cosmo_quick_camb_filtered"'
)

yaml_filtered = yaml_filtered.replace(
    'covmat: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/chains_cosmo_quick_camb_floor.covmat"',
    'covmat: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/chains_cosmo_quick_camb_ncdm3.covmat"'
)

yaml_filtered = yaml_filtered.replace('    nchains: 4\n', '')
filtered_yaml_path = COBAYA_PACK / "example_cosmo_quick_camb_filtered.yaml"

with open(filtered_yaml_path, 'w') as f:
    f.write(yaml_filtered)

print(f"Created modified YAML: {filtered_yaml_path}")
print()
print("Starting Cobaya MCMC...")

os.chdir(str(COBAYA_PACK))

try:
    result = subprocess.run(
        [sys.executable, "-m", "cobaya.run", str(filtered_yaml_path), "--allow-changes"],
        capture_output=False,
        text=True
    )
    if result.returncode == 0:
        print("Cobaya execution successful!")
        print(f"Output files: chains_cosmo_quick_camb_filtered.*")
    else:
        print("ERROR: Cobaya execution failed")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)

