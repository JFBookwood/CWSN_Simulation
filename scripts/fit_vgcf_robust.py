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
JK_NPZ = os.path.join(VGCF_DIR, 'multipoles_jk.npz')
OUT_JSON = os.path.join(VGCF_DIR, 'fit_params_robust.json')
OUT_PNG = os.path.join(VGCF_DIR, 'fit_multipoles_robust.png')

def smooth_template(s, xi0):
    try:
        from scipy.signal import savgol_filter
        win = max(5, len(xi0) // 5 * 2 + 1)
        win = min(win, len(xi0) if len(xi0) % 2 == 1 else len(xi0) - 1)
        poly = 3 if win >= 7 else 2
        return savgol_filter(xi0, window_length=max(5, win), polyorder=poly, mode='interp')
    except Exception:
        k = max(3, len(xi0) // 10)
        k = k + 1 - k % 2
        pad = k // 2
        x = np.pad(xi0, (pad, pad), mode='edge')
        kernel = np.ones(k) / k
        return np.convolve(x, kernel, mode='valid')

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
    ap.add_argument('--smin_fit', type=float, default=20.0)
    ap.add_argument('--smax_fit', type=float, default=80.0)
    args = ap.parse_args()
    z = np.load(GRID_NPZ)
    s_edges = z['s_edges']
    mu_edges = z['mu_edges']
    unit = str(z.get('unit', args.unit))
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    mp = pd.read_csv(MULTIPOLES_CSV)
    s = mp[f's_center_{unit}'].to_numpy(dtype=float)
    xi0_data = mp['xi0'].to_numpy(dtype=float)
    xi2_data = mp['xi2'].to_numpy(dtype=float)
    j = np.load(JK_NPZ)
    xi0_jk = j['xi0_jk']
    xi2_jk = j['xi2_jk']
    s_jk = j['s_centers']
    if not np.allclose(s, s_jk):
        xi0_jk = np.array([np.interp(s, s_jk, row) for row in xi0_jk])
        xi2_jk = np.array([np.interp(s, s_jk, row) for row in xi2_jk])
    mask = (s >= args.smin_fit) & (s <= args.smax_fit)
    s_fit = s[mask]
    dvec = np.concatenate([xi0_data[mask], xi2_data[mask]], axis=0)
    K = xi0_jk.shape[0]
    mean0 = xi0_jk[:, mask].mean(axis=0)
    mean2 = xi2_jk[:, mask].mean(axis=0)
    X = np.concatenate([xi0_jk[:, mask] - mean0, xi2_jk[:, mask] - mean2], axis=1)
    cov = (K - 1) / K * (X.T @ X)
    tpl = smooth_template(s, xi0_data)
    T = interp_template(s, tpl)
    try:
        from scipy.optimize import least_squares
    except Exception as e:
        print(f'[Error] SciPy not available ({e}). Please install scipy to run the fit.')
        return

    def model_vector(p):
        Xi = model_xi(s, mu_centers, p, T)
        dmu = 1.0 / len(mu_centers)
        p2 = 0.5 * (3.0 * mu_centers ** 2 - 1.0)
        xi0m = np.sum(Xi, axis=1) * (2.0 * dmu)
        xi2m = 5.0 * np.sum(Xi * p2[None, :], axis=1) * (2.0 * dmu)
        return np.concatenate([xi0m[mask], xi2m[mask]], axis=0)
    U, Svals, Vt = np.linalg.svd(cov, full_matrices=False)
    inv_sqrt = U * (1.0 / np.sqrt(np.maximum(Svals, 1e-12))) @ Vt
    p0 = np.array([1.0, 0.3, 1.0, 1.0, 0.0])
    lb = np.array([0.1, -0.2, 0.8, 0.8, -0.1])
    ub = np.array([5.0, 1.5, 1.2, 1.2, 0.1])

    def residuals(p):
        r = model_vector(p) - dvec
        return inv_sqrt @ r
    res = least_squares(residuals, p0, bounds=(lb, ub), xtol=1e-06, ftol=1e-06, gtol=1e-06, max_nfev=3000)
    A, beta, a_par, a_perp, C = res.x.tolist()
    print(f'Robust fit: A={A:.3f}, beta={beta:.3f}, alpha_par={a_par:.4f}, alpha_perp={a_perp:.4f}, C={C:.4f}')
    J = res.jac
    try:
        cov_p = np.linalg.inv(J.T @ J)
        perr = np.sqrt(np.diag(cov_p))
    except np.linalg.LinAlgError:
        cov_p = np.full((len(res.x), len(res.x)), np.nan)
        perr = np.full(len(res.x), np.nan)
    out = {'A': A, 'beta': beta, 'alpha_par': a_par, 'alpha_perp': a_perp, 'C': C, 'errors': {'A': float(perr[0]), 'beta': float(perr[1]), 'alpha_par': float(perr[2]), 'alpha_perp': float(perr[3]), 'C': float(perr[4])}, 'unit': unit, 'smin_fit': float(args.smin_fit), 'smax_fit': float(args.smax_fit), 'success': bool(res.success), 'cost': float(res.cost), 'nfev': int(res.nfev), 'message': str(res.message)}
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f'Wrote: {OUT_JSON}')
    try:
        import matplotlib.pyplot as plt
        Xi = model_xi(s, mu_centers, res.x, T)
        dmu = 1.0 / len(mu_centers)
        p2 = 0.5 * (3.0 * mu_centers ** 2 - 1.0)
        xi0m = np.sum(Xi, axis=1) * (2.0 * dmu)
        xi2m = 5.0 * np.sum(Xi * p2[None, :], axis=1) * (2.0 * dmu)
        plt.figure(figsize=(6, 4))
        plt.plot(s, xi0_data, 'o', ms=3, label='data $\\\\xi_0$')
        plt.plot(s, xi0m, '-', label='model $\\\\xi_0$')
        plt.plot(s, xi2_data, 'o', ms=3, label='data $\\\\xi_2$')
        plt.plot(s, xi2m, '-', label='model $\\\\xi_2$')
        plt.axhline(0, color='k', lw=1)
        plt.xlim(s.min(), s.max())
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
