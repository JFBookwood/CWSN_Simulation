# kapitel: start
import os
import json
import argparse
import numpy as np
import pandas as pd
ROOT = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN'
VGCF_DIR = os.path.join(ROOT, 'data', 'desi', 'edr', 'processed', 'voidfinder', 'vgcf')
GRID_NPZ = os.path.join(VGCF_DIR, 'xi_grid.npz')
MULTIPOLES_CSV = os.path.join(VGCF_DIR, 'multipoles.csv')
OUT_JSON = os.path.join(VGCF_DIR, 'fit_params.json')
OUT_PNG = os.path.join(VGCF_DIR, 'fit_multipoles.png')

def interp_template(s_centers, xi0_values):

    def f(s):
        s = np.asarray(s)
        return np.interp(s, s_centers, xi0_values, left=xi0_values[0], right=xi0_values[-1])
    return f

def model_xi(s_centers, mu_centers, params, T):
    A, beta, a_par, a_perp, C = params
    S, MU = np.meshgrid(s_centers, mu_centers, indexing='ij')
    s_par = a_par * S * MU
    s_perp = a_perp * S * np.sqrt(np.maximum(0.0, 1.0 - MU ** 2))
    Sprime = np.sqrt(s_par ** 2 + s_perp ** 2)
    MUprime = np.divide(s_par, Sprime, out=np.zeros_like(s_par), where=Sprime > 0)
    Xi_iso = T(Sprime)
    Xi = A * Xi_iso * (1.0 + beta * MUprime ** 2) ** 2 + C
    return Xi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', choices=['mpc', 'mpch'], default='mpc')
    ap.add_argument('--smin_fit', type=float, default=10.0)
    ap.add_argument('--smax_fit', type=float, default=90.0)
    args = ap.parse_args()
    z = np.load(GRID_NPZ)
    s_edges = z['s_edges']
    mu_edges = z['mu_edges']
    XI = z['xi']
    NG = z['NG']
    NR = z['NR']
    unit = str(z.get('unit', args.unit))
    mp = pd.read_csv(MULTIPOLES_CSV)
    s_centers = mp[f's_center_{unit}'].to_numpy(dtype=np.float64)
    xi0 = mp['xi0'].to_numpy(dtype=np.float64)
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    smin, smax = (float(args.smin_fit), float(args.smax_fit))
    s_mask = (s_centers >= smin) & (s_centers <= smax)
    XI_obs = XI[s_mask, :].copy()
    NGw = NG[s_mask, :].copy()
    NRw = NR[s_mask, :].copy()
    W = np.sqrt(np.maximum(NRw, 1.0))
    T = interp_template(s_centers[s_mask], xi0[s_mask])
    p0 = np.array([1.0, 0.5, 1.0, 1.0, 0.0], dtype=np.float64)
    lb = np.array([0.1, -0.2, 0.8, 0.8, -0.1])
    ub = np.array([5.0, 1.5, 1.2, 1.2, 0.1])
    try:
        from scipy.optimize import least_squares
    except Exception as e:
        print(f'[Error] SciPy not available ({e}). Please install scipy to run the fit.')
        return

    def residuals(p):
        XIm = model_xi(s_centers[s_mask], mu_centers, p, T)
        mask_valid = np.isfinite(XI_obs)
        r = (XIm - XI_obs)[mask_valid] * W[mask_valid]
        return r.ravel()
    res = least_squares(residuals, p0, bounds=(lb, ub), xtol=1e-06, ftol=1e-06, gtol=1e-06, max_nfev=2000)
    A, beta, a_par, a_perp, C = res.x.tolist()
    print(f'Fit result: A={A:.3f}, beta={beta:.3f}, alpha_par={a_par:.4f}, alpha_perp={a_perp:.4f}, C={C:.4f}')
    XIm_full = model_xi(s_centers, mu_centers, res.x, interp_template(s_centers, xi0))
    dmu = 1.0 / len(mu_centers)
    mu_c = mu_centers
    p2 = 0.5 * (3.0 * mu_c ** 2 - 1.0)
    xi0_mod = np.sum(XIm_full, axis=1) * (2.0 * dmu)
    xi2_mod = 5.0 * np.sum(XIm_full * p2[None, :], axis=1) * (2.0 * dmu)
    out = {'A': A, 'beta': beta, 'alpha_par': a_par, 'alpha_perp': a_perp, 'C': C, 'unit': unit, 'smin_fit': smin, 'smax_fit': smax, 'success': bool(res.success), 'cost': float(res.cost), 'nfev': int(res.nfev), 'message': str(res.message)}
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote: {OUT_JSON}')
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(s_centers, xi0, 'o', ms=3, label='data $\\\\xi_0$')
        plt.plot(s_centers, xi0_mod, '-', label='model $\\\\xi_0$')
        plt.plot(s_centers, mp['xi2'].to_numpy(dtype=float), 'o', ms=3, label='data $\\\\xi_2$')
        plt.plot(s_centers, xi2_mod, '-', label='model $\\\\xi_2$')
        plt.axhline(0, color='k', lw=1)
        plt.xlim(s_centers.min(), s_centers.max())
        plt.xlabel(f's [{unit}]')
        plt.ylabel('$\\\\xi_\\\\ell$')
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_PNG, dpi=150)
        plt.close()
        print(f'Wrote: {OUT_PNG}')
    except Exception as e:
        print(f'[Info] Matplotlib not available or plotting failed ({e}); continuing.')
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
