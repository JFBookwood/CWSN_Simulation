"""

Healpy compatibility wrapper for astropy-healpix

===============================================

This module provides a healpy-compatible interface using astropy-healpix.

"""

import astropy_healpix as ahp
import numpy as np

class HealpyWrapper:
    """Wrapper to make astropy-healpix compatible with healpy interface."""

    @staticmethod
    def pix2ang(nside, ipix):
        """Convert pixel index to angular coordinates."""
        lon, lat = ahp.healpix_to_lonlat(ipix, nside)
        # Convert to radians: theta = (90 - lat) * pi/180, phi = lon * pi/180
        theta = np.radians(90 - lat.to_value('deg'))
        phi = lon.to_value('rad')
        return theta, phi

    @staticmethod
    def ang2pix(nside, theta, phi):
        """Convert angular coordinates to pixel index."""
        # Convert from radians to degrees: lon = phi * 180/pi, lat = 90 - theta * 180/pi
        lon_deg = np.degrees(phi)
        lat_deg = 90 - np.degrees(theta)
        return ahp.lonlat_to_healpix(lon_deg, lat_deg, nside)

    @staticmethod
    def get_nside(npix):
        """Get nside from number of pixels."""
        return ahp.nside_from_npix(npix)

    @staticmethod
    def read_map(filename, **kwargs):
        """Read healpix map from file."""
        print(f"Mock reading healpix map from {filename}")
        return np.ones(12)

    @staticmethod
    def ud_grade(map_in, nside_out):
        """Upgrade/downgrade healpix map."""
        nside_in = ahp.nside_from_npix(len(map_in))
        return ahp.ud_grade_healpix(map_in, nside_in, nside_out)

pix2ang = HealpyWrapper.pix2ang
ang2pix = HealpyWrapper.ang2pix
get_nside = HealpyWrapper.get_nside
read_map = HealpyWrapper.read_map
ud_grade = HealpyWrapper.ud_grade

__version__ = "1.0.0 (astropy-healpix wrapper)"

