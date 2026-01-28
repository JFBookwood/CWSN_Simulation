"""
REVOLVER Void Finder Integration
"""
import os
import sys
import numpy as np
from pathlib import Path
import subprocess
import shutil
ROOT_DIR = Path(__file__).parent.parent
REVOLVER_DIR = ROOT_DIR / "Revolver"
DATA_DIR = ROOT_DIR / "data" / "desi" / "edr" / "processed"
OUTPUT_DIR = DATA_DIR / "revolver_voids"
GALAXY_FILE = DATA_DIR / "ELG_HIP_positions_xyz_mpc.csv"
RANDOMS_FILE = DATA_DIR / "ELG_HIP_randoms_xyz_mpc.csv"
REVOLVER_PARAMS_TEMPLATE = '''
import numpy as np
input_filename = "{galaxy_file}"
output_folder = "{output_dir}"
random_filename = "{randoms_file}"
file_format = "csv"
skip_header = 1
coords = "cartesian"
box_length = None
Omega_m = 0.3
h = 0.6766
do_reconstruction = True
beta = 0.4
f = 0.8
bias = 1.5
void_method = "zobov"
zobov_buffer = 0.1
zobov_box_div = 2
use_mpi = False
grid_size = 256
min_dens_cut = 0.1
min_void_volume = 1000.0
max_void_radius = 50.0
write_void_catalog = True
write_density_field = False
write_reconstructed_field = False
nthreads = 4
'''
def prepare_revolver_input():
    """Prepare input files for REVOLVER."""
    print("Preparing REVOLVER input files...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not GALAXY_FILE.exists():
        print(f"ERROR: Galaxy file not found: {GALAXY_FILE}")
        return False
    if not RANDOMS_FILE.exists():
        print(f"ERROR: Randoms file not found: {RANDOMS_FILE}")
        return False
    print(f"Galaxy file: {GALAXY_FILE}")
    print(f"Randoms file: {RANDOMS_FILE}")
    print(f"Output directory: {OUTPUT_DIR}")
    return True
def create_parameter_file():
    """Create REVOLVER parameter file."""
    params_content = REVOLVER_PARAMS_TEMPLATE.format(
        galaxy_file=GALAXY_FILE,
        randoms_file=RANDOMS_FILE,
        output_dir=OUTPUT_DIR
    )
    params_file = OUTPUT_DIR / "revolver_params.py"
    with open(params_file, 'w') as f:
        f.write(params_content)
    print(f"Created parameter file: {params_file}")
    return params_file
def run_revolver(params_file):
    """Run REVOLVER void finding."""
    print("Running REVOLVER void finding...")
    original_dir = os.getcwd()
    os.chdir(str(REVOLVER_DIR))
    try:
        cmd = [sys.executable, "revolver.py", "--par", str(params_file)]
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        if result.returncode == 0:
            print("SUCCESS: REVOLVER completed successfully")
            return True
        else:
            print(f"ERROR: REVOLVER failed with return code {result.returncode}")
            return False
    finally:
        os.chdir(original_dir)
def analyze_revolver_output():
    """Analyze REVOLVER output and extract void catalog."""
    print("Analyzing REVOLVER output...")
    void_files = list(OUTPUT_DIR.glob("*void*"))
    print(f"Found void files: {[f.name for f in void_files]}")
    catalog_file = OUTPUT_DIR / "void_catalog.npy"
    if catalog_file.exists():
        print(f"Loading void catalog: {catalog_file}")
        voids = np.load(catalog_file)
        print(f"Void catalog shape: {voids.shape}")
        print(f"Void catalog columns: {voids.dtype.names if hasattr(voids, 'dtype') else 'array'}")
        if hasattr(voids, 'dtype') and voids.dtype.names:
            for name in voids.dtype.names:
                print(f"  {name}: {voids[name][:5]}...")
        else:
            print(f"First few voids: {voids[:5]}")
        return voids
    else:
        for ext in ['.txt', '.dat', '.fits']:
            catalog_files = list(OUTPUT_DIR.glob(f"*void*{ext}"))
            if catalog_files:
                print(f"Found catalog file: {catalog_files[0]}")
                break
        return None
def main():
    """Main REVOLVER integration function."""
    print("=" * 60)
    print("REVOLVER Void Finder Integration")
    print("=" * 60)
    if not prepare_revolver_input():
        return False
    params_file = create_parameter_file()
    if not run_revolver(params_file):
        return False
    voids = analyze_revolver_output()
    if voids is not None:
        print(f"\nSuccessfully extracted {len(voids)} voids from REVOLVER")
        print("REVOLVER integration completed!")
        return True
    else:
        print("Warning: No void catalog found in output")
        return False
if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
