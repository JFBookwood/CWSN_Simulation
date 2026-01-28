"""
TNG300 Void Analyzation Test Script
"""
from astropy.io import fits
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "Data"
filename = DATA / "TNG300-1-Halos.fits" / "TNG300-1-Halos.fits"
hdul = fits.open(filename)

print(hdul.info())
