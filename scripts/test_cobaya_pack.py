# kapitel: start
import os
import numpy as np
ROOT = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN'
VGCF_DIR = os.path.join(ROOT, 'data', 'desi', 'edr', 'processed', 'voidfinder', 'vgcf')
PACK_DIR = os.path.join(VGCF_DIR, 'cobaya_pack')
DATA_NPZ = os.path.join(PACK_DIR, 'voids_vgcf_data.npz')

def interp_template(s_grid, xi0_values):

    def f(s):
        return np.interp(s, s_grid, xi0_values, left=xi0_values[0], right=xi0_values[-1])
    return f

def model_grid(s, mu_centers, params, tpl):
    A, beta, a_par, a_perp, C = params
    S, MU = np.meshgrid(s, mu_centers, indexing='ij')
    s_par = a_par * S * MU
    s_perp = a_perp * S * np.sqrt(np.maximum(0.0, 1.0 - MU ** 2))
    Sprime = np.sqrt(s_par ** 2 + s_perp ** 2)
    MUprime = np.divide(s_par, Sprime, out=np.zeros_like(s_par), where=Sprime > 0)
    Xi_iso = tpl(Sprime)
    Xi = A * Xi_iso * (1.0 + beta * MUprime ** 2) ** 2 + C
    return Xi

def main():
    z = np.load(DATA_NPZ)
    s = z['s']
    s_fit = z['s_fit']
    s_mask = z['s_mask']
    mu_centers = z['mu_centers']
    dvec = z['dvec']
    cov = z['cov_reg'] if 'cov_reg' in z.files else z['cov']
    xi0_sm = z['xi0_sm']
    tpl = interp_template(s, xi0_sm)
    params = [1.0, 0.3, 1.0, 1.0, 0.0]
    Xi = model_grid(s, mu_centers, params, tpl)
    dmu = 1.0 / len(mu_centers)
    p2 = 0.5 * (3.0 * mu_centers ** 2 - 1.0)
    xi0m = np.sum(Xi, axis=1) * (2.0 * dmu)
    xi2m = 5.0 * np.sum(Xi * p2[None, :], axis=1) * (2.0 * dmu)
    mvec = np.concatenate([xi0m[s_mask], xi2m[s_mask]], axis=0)
    U, Svals, Vt = np.linalg.svd(cov, full_matrices=False)
    inv_cov = U * (1.0 / np.maximum(Svals, 1e-12)) @ Vt
    r = mvec - dvec
    chi2 = float(r @ (inv_cov @ r))
    logp = -0.5 * chi2
    print(f"Smoke test OK. chi2={chi2:.3f}, logp={logp:.3f}, ndim={len(dvec)}, shrinkage={z.get('shrinkage_lambda', None)}")
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
