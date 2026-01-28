"""
Create a filtered likelihood .npz file for void-galaxy correlation function
"""

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

ROOT = r'c:\Users\Jesse\Desktop\Experimente\CWSN'
VGCF_DIR = os.path.join(ROOT, 'data', 'desi', 'edr', 'processed', 'voidfinder', 'vgcf')
COBAYA_PACK = os.path.join(VGCF_DIR, 'cobaya_pack')

multipoles_file = os.path.join(VGCF_DIR, 'multipoles_survey_aware.csv')
if not os.path.exists(multipoles_file):
    exit(1)

multipoles = pd.read_csv(multipoles_file)
s = multipoles['s_center_mpc'].values
xi0 = multipoles['xi0'].values
xi2 = multipoles['xi2'].values
pairs_gal = multipoles['pairs_gal'].values
pairs_ran = multipoles['pairs_ran'].values
n_voids = int(multipoles['num_voids'].iloc[0])

original_npz_file = os.path.join(COBAYA_PACK, 'voids_vgcf_data.npz')
if not os.path.exists(original_npz_file):
    z_eff = 0.8
    H_fid = 100.0
    DM_fid = 1000.0
    s_mask = ~np.isnan(xi0) & ~np.isnan(xi2)
    mu_centers = np.linspace(0.05, 0.95, 10)
else:
    data = np.load(original_npz_file, allow_pickle=True)
    z_eff = float(data['z_eff'])
    H_fid = float(data['H_fid'])
    DM_fid = float(data['DM_fid'])
    s_mask = data['s_mask'].astype(bool)
    mu_centers = data['mu_centers']

valid = ~np.isnan(xi0)
if np.sum(valid) < 3:
    exit(1)

s_valid = s[valid]
xi0_valid = xi0[valid]

from scipy.interpolate import CubicSpline

try:
    cs = CubicSpline(s_valid, xi0_valid, bc_type='natural')
    xi0_sm = cs(s)
except Exception as e:
    xi0_sm = np.nan_to_num(xi0, nan=0.0)

n_bins = len(s[~np.isnan(xi0)])
errors_xi0 = np.sqrt(1.0 / (pairs_gal[~np.isnan(xi0)] + 1e-10))
errors_xi0 = np.maximum(errors_xi0, 0.01)
errors_xi2 = np.sqrt(1.0 / (pairs_gal[~np.isnan(xi2)] + 1e-10))
errors_xi2 = np.maximum(errors_xi2, 0.02)
cov_full = np.diag(np.concatenate([errors_xi0**2, errors_xi2**2]))
cov_reg = cov_full + 1e-3 * np.eye(len(cov_full))

output_file = os.path.join(COBAYA_PACK, 'voids_vgcf_data_survey_aware.npz')
dvec = np.concatenate([xi0[~np.isnan(xi0)], xi2[~np.isnan(xi2)]])

save_dict = {
    's': s,
    's_mask': s_mask if 's_mask' in locals() else ~np.isnan(xi0),
    'xi0': xi0,
    'xi0_sm': xi0_sm,
    'xi2': xi2,
    'mu_centers': mu_centers,
    'dvec': dvec,
    'cov': cov_full,
    'cov_reg': cov_reg,
    'z_eff': z_eff,
    'H_fid': H_fid,
    'DM_fid': DM_fid,
    'n_voids': n_voids,
    'n_voids_original': 500,
}

np.savez(output_file, **save_dict)



