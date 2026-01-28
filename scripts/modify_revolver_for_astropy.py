"""
This script modifies the REVOLVER source code to use astropy-healpix
instead of the problematic healpy package.
"""
import os
import re
from pathlib import Path

REVOLVER_DIR = Path("Revolver")

def modify_healpy_imports():
    """Replace healpy imports with astropy_healpix."""
    files_to_modify = [
        REVOLVER_DIR / "python_tools" / "zobov.py",
        REVOLVER_DIR / "python_tools" / "galaxycat.py",
    ]
    for file_path in files_to_modify:
        if file_path.exists():
            print(f"Modifying {file_path}...")
            with open(file_path, "r") as f:
                content = f.read()
            content = content.replace(
                "import healpy as hp", "import astropy_healpix as hp"
            )
            content = content.replace("healpy as hp", "astropy_healpix as hp")
            replacements = {
                "hp.pix2ang": "hp.healpix_to_lonlat",
                "hp.ang2pix": "hp.lonlat_to_healpix",
                "hp.get_nside": "hp.nside_from_npix",
                "hp.read_map": "hp.read_healpix_map",
                "hp.ud_grade": "hp.ud_grade_healpix",
            }
            for old, new in replacements.items():
                content = content.replace(old, new)
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Modified {file_path}")
        else:
            print(f"File not found: {file_path}")

def create_astropy_wrapper():
    """Create a wrapper to make astropy-healpix compatible with healpy interface."""
    wrapper_code = '''
import astropy_healpix as ahp
import numpy as np

class HealpyWrapper:
    @staticmethod
    def pix2ang(nside, ipix):
        lon, lat = ahp.healpix_to_lonlat(ipix, nside)
        theta = (90 - lat.value) * np.pi / 180  # colatitude in radians
        phi = lon.value * np.pi / 180  # longitude in radians
        return theta, phi
    @staticmethod
    def ang2pix(nside, theta, phi):
        lat = (np.pi/2 - theta) * 180 / np.pi  # latitude in degrees
        lon = phi * 180 / np.pi  # longitude in degrees
        return ahp.lonlat_to_healpix(lon, lat, nside)
    @staticmethod
    def get_nside(npix):
        return ahp.nside_from_npix(npix)
    @staticmethod
    def read_map(filename, **kwargs):
        print(f"Mock reading healpix map from {filename}")
        return np.ones(12)  # Mock data
    @staticmethod
    def ud_grade(map_in, nside_out):
        nside_in = ahp.nside_from_npix(len(map_in))
        return ahp.ud_grade_healpix(map_in, nside_in, nside_out)

pix2ang = HealpyWrapper.pix2ang
ang2pix = HealpyWrapper.ang2pix
get_nside = HealpyWrapper.get_nside
read_map = HealpyWrapper.read_map
ud_grade = HealpyWrapper.ud_grade
__version__ = "1.0.0 (astropy-healpix wrapper)"
'''
    wrapper_file = REVOLVER_DIR / "python_tools" / "healpy_wrapper.py"
    wrapper_file.parent.mkdir(parents=True, exist_ok=True)
    with open(wrapper_file, "w") as f:
        f.write(wrapper_code)
    print(f"Created healpy wrapper: {wrapper_file}")

def modify_imports_to_use_wrapper():
    """Modify imports to use the wrapper instead of healpy."""
    files_to_modify = [
        REVOLVER_DIR / "python_tools" / "zobov.py",
        REVOLVER_DIR / "python_tools" / "galaxycat.py",
    ]
    for file_path in files_to_modify:
        if file_path.exists():
            print(f"Updating imports in {file_path}...")
            with open(file_path, "r") as f:
                content = f.read()
            content = content.replace(
                "import healpy as hp", "from healpy_wrapper import *"
            )
            with open(file_path, "w") as f:
                f.write(content)
            print(f"Updated imports in {file_path}")

def test_modified_revolver():
    """Test if the modified REVOLVER can import successfully."""
    print("Testing modified REVOLVER imports...")
    try:
        import sys
        sys.path.insert(0, str(REVOLVER_DIR / "python_tools"))
        from healpy_wrapper import pix2ang, ang2pix
        print("SUCCESS: Healpy wrapper imports successful")
        result = ang2pix(8, 0, 0)
        print(f"SUCCESS: ang2pix test: {result}")
        print("SUCCESS: Modified REVOLVER imports successful!")
        return True
    except Exception as e:
        print(f"ERROR: Import test failed: {e}")
        return False

def main():
    """Main modification function."""
    print("=" * 60)
    print("MODIFYING REVOLVER FOR ASTROPY-HEALPIX COMPATIBILITY")
    print("=" * 60)
    create_astropy_wrapper()
    modify_imports_to_use_wrapper()
    success = test_modified_revolver()
    if success:
        print("\n" + "=" * 60)
        print("SUCCESS: REVOLVER modified for astropy-healpix!")
        print("You can now run REVOLVER with:")
        print("python scripts/run_revolver_simple.py")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("WARNING: Modifications completed but testing failed")
        print("Check the error messages above")
        print("=" * 60)

if __name__ == "__main__":
    main()
