"""
This script analyzes void catalogs, either mock observations or the TNG-based void map
in `data/desi/edr/TNG`, to simulate void expansion and derive neutrino mass constraints.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import camb
    HAVE_CAMB = True
except ImportError:
    camb = None
    HAVE_CAMB = False

TNG_ANALYSIS_DIR = Path(__file__).resolve().parent / "TNG_Analyzation"
if TNG_ANALYSIS_DIR.exists() and str(TNG_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(TNG_ANALYSIS_DIR))

try:
    from read_ivolume import read_ivolume
except ImportError:
    read_ivolume = None

OUTPUT_DIR = Path("data/desi/edr/processed/neutrino_analysis")
DEFAULT_TNG_DIR = Path("data/desi/edr/TNG")
M_NU_SCENARIOS = [0.0, 0.06, 0.1, 0.2, 0.3, 0.4]
CAMB_COSMOLOGY = {
    'H0': 67.66,
    'ombh2': 0.02237,
    'omch2': 0.120,
    'mnu': 0.0,
    'omk': 0.0
}
CAMB_POWER = {
    'As': 2.1e-9,
    'ns': 0.965,
    'r': 0.0
}
_CAMB_SIGMA8_CACHE: dict[float, float] = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run void expansion neutrino analysis")


parser.add_argument("--use-tng-data", action="store_true",
                    help="substitute the mock observed void catalog with the TNG-derived void map")
parser.add_argument("--tng-dir", type=Path, default=DEFAULT_TNG_DIR,
                    help="location of the TNG data folder")
parser.add_argument("--tng-segmentation", type=str,
                    default="snap_009.FIX.Y-0-1-2-3-4.iwat",
                    help="name of the segmentation file that contains the void labels")
parser.add_argument("--grid-size", type=int, default=512,
                    help="number of cells per side in the segmentation grid")
parser.add_argument("--box-size", type=float, default=205000.0,
                    help="physical box size in kpc/h for the segmentation grid")
returnparser.parse_args()


def load_observed_voids():
    print("Loading default observed void catalog...")


mock_voids = np.array([
    [0, 0, 0, 186.0, 2.7e7, 10000],
    [100, 100, 100, 158.3, 1.66e7, 8000],
    [200, -50, 75, 137.1, 1.08e7, 6000],
    [50, 150, -25, 124.5, 8.08e6, 5000],
    [-75, 25, 125, 122.9, 7.77e6, 4800],
    [175, -100, 50, 122.8, 7.76e6, 4800],
    [25, -75, -50, 122.4, 7.68e6, 4700],
    [-25, 175, 100, 120.8, 7.38e6, 4500],
    [125, 75, -75, 118.4, 6.95e6, 4200]
])
print(f"Using {len(mock_voids)} observed voids")
df = pd.DataFrame(mock_voids, columns=[
                  'x', 'y', 'z', 'radius', 'volume', 'voxels'])
df['source'] = 'mock_observed'
returndf


def load_tng_void_catalog(tng_dir, segmentation_file, grid_size, box_size):
    if read_ivolume is None:
        raise RuntimeError(
            "TNG helpers are not available; ensure scripts/TNG_Analyzation/read_ivolume.py exists")
    segmentation_path = Path(tng_dir) / segmentation_file
    if not segmentation_path.exists():
        raise FileNotFoundError(segmentation_path)


print(f"Loading TNG void catalog from {segmentation_path}")
reg = read_ivolume(segmentation_path)
voxel_size = box_size/grid_size
voxel_volume = voxel_size**3
ids, counts = np.unique(reg, return_counts=True)
mask = ids > 0
ids = ids[mask]
counts = counts[mask]
radii = ((3.0/(4.0*np.pi))*(counts*voxel_volume))**(1.0/3.0)
volumes = counts*voxel_volume
df = pd.DataFrame({
    'void_id': ids,
    'radius': radii,
    'volume': volumes,
    'voxels': counts
})
df['x'] = 0.0
df['y'] = 0.0
df['z'] = 0.0
df['source'] = 'tng_void_map'
returndf


def prepare_observed_void_catalog(args):
    if args.use_tng_data:
        return load_tng_void_catalog(
            args.tng_dir, args.tng_segmentation, args.grid_size, args.box_size)
    return load_observed_voids()


def compute_sigma8_with_camb(m_nu: float) -> float | None:
    if not HAVE_CAMB:
        return None

    key = round(m_nu, 6)
    if key in _CAMB_SIGMA8_CACHE:
        return _CAMB_SIGMA8_CACHE[key]
    pars = camb.CAMBparams()
    cosmo = CAMB_COSMOLOGY.copy()
    cosmo['mnu'] = m_nu
    pars.set_cosmology(**cosmo)
    pars.InitPower.set_params(**CAMB_POWER)
    pars.set_matter_power(redshifts=[0.0], kmax=2.0)
    results = camb.get_results(pars)
    sigma8 = results.get_sigma8()
    _CAMB_SIGMA8_CACHE[key] = sigma8
    return sigma8

def camb_sigma8_ratio(m_nu: float) -> float:
    if not HAVE_CAMB:
        return 1.0
    sigma_ref = compute_sigma8_with_camb(0.0)
    if sigma_ref is None or sigma_ref == 0:
        return 1.0
    sigma_mnu = compute_sigma8_with_camb(m_nu)
    if sigma_mnu is None:
        return 1.0
    return sigma_mnu / sigma_ref


def simulate_void_expansion(m_nu, n_realizations=10):
    print(f"Simulating void expansion for m_nu = {m_nu} eV")

    base_radii = np.array([10, 15, 20, 25, 30, 35, 40, 50, 60, 70])
    base_scale = 1.0 - 0.02 * m_nu
    nu_effect = base_scale * camb_sigma8_ratio(m_nu)
    simulated_voids = []
    for i in range(n_realizations):
        scatter = np.random.normal(1.0, 0.1, len(base_radii))
        radii = base_radii * nu_effect * scatter
        for j, r in enumerate(radii):
            volume = (4 / 3) * np.pi * r ** 3
            simulated_voids.append({
                'realization': i,
                'void_id': j,
                'radius': r,
                'volume': volume,
                'm_nu': m_nu,
                'source': f'sim_mnu_{m_nu}'
            })
    return pd.DataFrame(simulated_voids)


def generate_simulation_data():
    print("Generating simulation data...")

    all_simulations = []
    for m_nu in M_NU_SCENARIOS:
        sim_voids = simulate_void_expansion(m_nu)
        all_simulations.append(sim_voids)
    simulation_data = pd.concat(all_simulations, ignore_index=True)
    print(f"Generated {len(simulation_data)} simulated void measurements")
    return simulation_data


def compare_void_distributions(observed_voids, simulation_data):
    print("Comparing void size distributions...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Void Size Distribution: Neutrino Mass Analysis')
    ax = axes[0, 0]
    for m_nu in M_NU_SCENARIOS:
        sim_data = simulation_data[simulation_data['m_nu'] == m_nu]
        ax.hist(sim_data['radius'], bins=20, alpha=0.7,
                label=f'Sim m_nu={m_nu} eV', density=True, histtype='step', linewidth=2)
    ax.hist(observed_voids['radius'], bins=20, alpha=0.8,
            label='Observed', density=True, color='black', histtype='stepfilled')
    ax.set_xlabel('Void Radius [Mpc/h]')
    ax.set_ylabel('Normalized Count')
    ax.legend()
    ax.set_title('Void Radius Distribution')
    ax = axes[0, 1]
    for m_nu in M_NU_SCENARIOS:
        sim_data = simulation_data[simulation_data['m_nu'] == m_nu]
        ax.hist(sim_data['volume'], bins=20, alpha=0.7,
                label=f'Sim m_nu={m_nu} eV', density=True, histtype='step', linewidth=2)
    ax.hist(observed_voids['volume'], bins=20, alpha=0.8,
            label='Observed', density=True, color='black', histtype='stepfilled')
    ax.set_xlabel('Void Volume [(Mpc/h)^3]')
    ax.set_ylabel('Normalized Count')
    ax.set_xscale('log')
    ax.legend()
    ax.set_title('Void Volume Distribution')
    ax = axes[1, 0]
    mean_radii = []
    std_radii = []
    for m_nu in M_NU_SCENARIOS:
        sim_data = simulation_data[simulation_data['m_nu'] == m_nu]
        mean_radii.append(sim_data['radius'].mean())
        std_radii.append(sim_data['radius'].std())
    mean_radii_arr = np.array(mean_radii)
    std_radii_arr = np.array(std_radii)
    obs_radius_mean = observed_voids['radius'].mean()
    ax.errorbar(M_NU_SCENARIOS, mean_radii_arr, yerr=std_radii_arr,
                marker='o', linestyle='-', color='blue', label='Simulations')
    ax.axhline(y=obs_radius_mean, color='red', linestyle='--',
               label=f'Observed: {obs_radius_mean:.1f} Mpc/h')
    ax.fill_between(M_NU_SCENARIOS,
                    mean_radii_arr - std_radii_arr,
                    mean_radii_arr + std_radii_arr,
                    alpha=0.3, color='blue')
    ax.set_xlabel('Neutrino Mass [eV]')
    ax.set_ylabel('Mean Void Radius [Mpc/h]')
    ax.legend()
    ax.set_title('Mean Void Radius vs Neutrino Mass')
    ax = axes[1, 1]
    chi_squared_values = []
    obs_hist, bin_edges = np.histogram(
        observed_voids['radius'], bins=15, density=True)
    for m_nu in M_NU_SCENARIOS:
        sim_data = simulation_data[simulation_data['m_nu'] == m_nu]
        sim_hist, _ = np.histogram(sim_data['radius'], bins=bin_edges, density=True)
        denominator = np.maximum(obs_hist, 1e-10)
        chi2_val = np.sum((obs_hist - sim_hist) ** 2 / denominator)
        if not np.isfinite(chi2_val):
            chi2_val = 1000.0
        chi_squared_values.append(chi2_val)
    ax.plot(M_NU_SCENARIOS, chi_squared_values, 'o-', color='red', linewidth=2)
    ax.set_xlabel('Neutrino Mass [eV]')
    ax.set_ylabel('chi^2')
    ax.set_title('Goodness of Fit (chi^2)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_DIR / "neutrino_void_analysis.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    return chi_squared_values

def calculate_neutrino_constraints(chi_squared_values):
    print("Calculating neutrino mass constraints...")
    best_fit_idx = np.argmin(chi_squared_values)
    best_fit_m_nu = M_NU_SCENARIOS[best_fit_idx]
    min_chi2 = chi_squared_values[best_fit_idx]
    print(f"Best-fit neutrino mass: {best_fit_m_nu} eV (chi2 = {min_chi2:.2f})")
    m_nu_68 = [m for m, chi2 in zip(M_NU_SCENARIOS, chi_squared_values) if chi2 <= min_chi2 + 1.0]
    m_nu_95 = [m for m, chi2 in zip(M_NU_SCENARIOS, chi_squared_values) if chi2 <= min_chi2 + 4.0]
    print(f"68% CL upper limit: < {max(m_nu_68):.2f} eV")
    print(f"95% CL upper limit: < {max(m_nu_95):.2f} eV")
    return {
        'best_fit': best_fit_m_nu,
        'min_chi2': min_chi2,
        'cl_68': max(m_nu_68),
        'cl_95': max(m_nu_95)
    }


def create_summary_report(constraints, chi_squared_values, observed_voids):
    print("\n" + "=" * 60)
    print("NEUTRINO MASS CONSTRAINTS FROM VOID EXPANSION ANALYSIS")
    print("=" * 60)
    theory_note = "CAMB-driven growth" if HAVE_CAMB else "empirical scaling (CAMB unavailable)"
    print(f"Theory: {theory_note}")
    sources = observed_voids['source'].unique()
    source_label = ", ".join(sources)
    print(f"Observed catalog(s): {source_label}")
    print("RESULTS:")
    print(f"Best-fit neutrino mass: {constraints['best_fit']:.3f} eV")
    print(f"Minimum chi^2: {constraints['min_chi2']:.2f}")
    print(f"68% CL upper limit: < {constraints['cl_68']:.3f} eV")
    print(f"95% CL upper limit: < {constraints['cl_95']:.3f} eV")
    print("COMPARISON WITH LITERATURE:")
    print("- Planck (CMB): sum(m_nu) < 0.12 eV (95% CL)")
    print("- DESI BAO + Planck: sum(m_nu) < 0.072 eV (95% CL)")
    print(
        f"- THIS ANALYSIS (Void Expansion): sum(m_nu) < {constraints['cl_95']:.3f} eV (95% CL)")
    results = {
        'neutrino_masses': M_NU_SCENARIOS,
        'chi_squared': chi_squared_values,
        'constraints': constraints,
        'observed_sources': sources.tolist()
    }
    np.savez(OUTPUT_DIR / "neutrino_analysis_results.npz", **results)
    print(f"Results saved to: {OUTPUT_DIR / 'neutrino_analysis_results.npz'}")

def main():
    args = parse_args()
    print("=" * 70)
    print("VOID EXPANSION ANALYSIS FOR NEUTRINO MASS ESTIMATION")
    print("=" * 70)
    if HAVE_CAMB:
        print("CAMB detected; scaling void expansion using sigma8 ratios.")
    else:
        print("CAMB not detected; falling back to empirical scaling.")
    observed_voids = prepare_observed_void_catalog(args)
    simulation_data = generate_simulation_data()
    chi_squared_values = compare_void_distributions(
        observed_voids, simulation_data)
    constraints = calculate_neutrino_constraints(chi_squared_values)
    create_summary_report(constraints, chi_squared_values, observed_voids)
    print("\nAnalysis completed successfully!")
    print(f"Results saved in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
