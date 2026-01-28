"""
Utilities for identifying and filtering DESI survey stripe artifacts.
These functions separate real cosmic structure from observational geometry effects.
"""

import numpy as np

# DESI stripe sectors (angular ranges in degrees)
DESI_STRIPE_SECTORS = [
    (-173.5, -162.5),
    (-151.4, -137.4),
    (-126.4, -110.3),
    (-104.3, -85.2),
]

# DESI empty regions (no data)
DESI_EMPTY_REGIONS = [
    (-162.5, -151.4),
    (-137.4, -126.4),
    (-110.3, -104.3),
    (-85.2, 148.4),
]


def angle_from_xy(x, y):
    """
    Calculate angle (azimuth) for point(s) from Earth.

    Parameters:
    -----------
    x, y : float or array
        Cartesian coordinates

    Returns:
    --------
    angle : float or array
        Angle in degrees [-180, 180]
    """
    return np.arctan2(y, x) * 180 / np.pi


def is_in_stripe(x, y, stripe_sectors=None):
    """
    Check if point is in a DESI stripe.

    Parameters:
    -----------
    x, y : float or array
        Cartesian coordinates
    stripe_sectors : list of tuples, optional
        List of (angle_min, angle_max) sectors.
        Default: DESI_STRIPE_SECTORS

    Returns:
    --------
    in_stripe : bool or array
        True if in a stripe
    """
    if stripe_sectors is None:
        stripe_sectors = DESI_STRIPE_SECTORS

    angle = angle_from_xy(x, y)

    in_stripe = np.zeros_like(angle, dtype=bool)

    for angle_min, angle_max in stripe_sectors:
        in_stripe |= (angle >= angle_min) & (angle <= angle_max)

    return in_stripe


def is_in_empty_region(x, y, empty_regions=None):
    """
    Check if point is in empty region (no DESI data).

    Parameters:
    -----------
    x, y : float or array
        Cartesian coordinates
    empty_regions : list of tuples, optional
        List of (angle_min, angle_max) empty regions.
        Default: DESI_EMPTY_REGIONS

    Returns:
    --------
    in_empty : bool or array
        True if in empty region
    """
    if empty_regions is None:
        empty_regions = DESI_EMPTY_REGIONS

    angle = angle_from_xy(x, y)

    in_empty = np.zeros_like(angle, dtype=bool)

    for angle_min, angle_max in empty_regions:
        if angle_min > angle_max:
            in_empty |= (angle >= angle_min) | (angle <= angle_max)
        else:
            in_empty |= (angle >= angle_min) & (angle <= angle_max)

    return in_empty


def remove_stripe_artifacts(galaxies, voids):
    """
    Remove galaxies and voids in stripe artifact regions.

    This function removes data that is likely contaminated by
    DESI survey geometry (radial observation patterns).

    Parameters:
    -----------
    galaxies : array, shape (N, 3)
        Galaxy (x, y, z) coordinates
    voids : array, shape (M, 3)
        Void (x, y, z) centers

    Returns:
    --------
    galaxies_clean : array, shape (N', 3)
        Filtered galaxies (no stripe artifacts)
    voids_clean : array, shape (M', 3)
        Filtered voids (no stripe artifacts)
    mask_gal : array, dtype bool
        Mask indicating which galaxies were kept
    mask_void : array, dtype bool
        Mask indicating which voids were kept
    """
    x_gal = galaxies[:, 0]
    y_gal = galaxies[:, 1]
    x_void = voids[:, 0]
    y_void = voids[:, 1]

    mask_gal = ~is_in_stripe(x_gal, y_gal)
    mask_void = ~is_in_stripe(x_void, y_void)

    galaxies_clean = galaxies[mask_gal]
    voids_clean = voids[mask_void]

    return galaxies_clean, voids_clean, mask_gal, mask_void


def analyze_stripe_contamination(galaxies, voids=None):
    """
    Analyze how much data is contaminated by stripe artifacts.

    Parameters:
    -----------
    galaxies : array, shape (N, 3)
        Galaxy coordinates
    voids : array, optional
        Void coordinates

    Returns:
    --------
    stats : dict
        Statistics about stripe contamination
    """
    x = galaxies[:, 0]
    y = galaxies[:, 1]

    in_stripe = is_in_stripe(x, y)
    in_empty = is_in_empty_region(x, y)

    n_stripe = np.sum(in_stripe)
    n_empty = np.sum(in_empty)
    n_total = len(galaxies)

    stats = {
        'n_total': n_total,
        'n_in_stripe': n_stripe,
        'n_in_empty': n_empty,
        'n_valid': n_total - n_stripe - n_empty,
        'fraction_stripe': n_stripe / n_total,
        'fraction_empty': n_empty / n_total,
        'fraction_valid': (n_total - n_stripe - n_empty) / n_total,
    }

    if voids is not None:
        x_void = voids[:, 0]
        y_void = voids[:, 1]

        in_stripe_void = is_in_stripe(x_void, y_void)

        stats['n_voids_in_stripe'] = np.sum(in_stripe_void)
        stats['n_voids_total'] = len(voids)
        stats['fraction_voids_stripe'] = np.sum(in_stripe_void) / len(voids)

    return stats


def print_contamination_report(galaxies, voids=None):
    """
    Print detailed report about stripe contamination.

    Parameters:
    -----------
    galaxies : array, shape (N, 3)
        Galaxy coordinates
    voids : array, optional
        Void coordinates
    """
    stats = analyze_stripe_contamination(galaxies, voids)

    print("\n" + "=" * 70)
    print("DESI STRIPE CONTAMINATION ANALYSIS")
    print("=" * 70)

    print(f"\nGALAXIES:")
    print(f"  Total galaxies:        {stats['n_total']:,}")
    print(f"  In stripe regions:     {stats['n_in_stripe']:,} ({stats['fraction_stripe'] * 100:.1f}%)")
    print(f"  In empty regions:      {stats['n_in_empty']:,} ({stats['fraction_empty'] * 100:.1f}%)")
    print(f"  Potentially valid:     {stats['n_valid']:,} ({stats['fraction_valid'] * 100:.1f}%)")

    if voids is not None:
        print(f"\nVOIDS:")
        print(f"  Total voids:           {stats['n_voids_total']}")
        print(f"  In stripe regions:     {stats['n_voids_in_stripe']} ({stats['fraction_voids_stripe'] * 100:.1f}%)")

    print("\nRECOMMENDATION:")
    if stats['fraction_stripe'] > 0.2:
        print("  WARNING: More than 20% of data affected by survey geometry!")
        print("     Should use FILTERED analysis to avoid systematic bias.")
    else:
        print("  ✓ Stripe contamination is acceptable (<20%)")

    print("=" * 70 + "\n")

    return stats
