# kapitel: start
import os
import argparse
import numpy as np
import pandas as pd
ROOT = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN'
VGCF_DIR = os.path.join(ROOT, 'data', 'desi', 'edr', 'processed', 'voidfinder', 'vgcf')
OUT_DIR = os.path.join(VGCF_DIR, 'cobaya_pack')
MULTIPOLES_CSV = os.path.join(VGCF_DIR, 'multipoles.csv')
JK_NPZ = os.path.join(VGCF_DIR, 'multipoles_jk.npz')
GRID_NPZ = os.path.join(VGCF_DIR, 'xi_grid.npz')
C_KM_S = 299792.458

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

def lcdm_E(z, om):
    return np.sqrt(om * (1 + z) ** 3 + (1 - om))

def lcdm_DM_Mpc(z, H0, om, nstep: int=2048):
    z_arr = np.linspace(0.0, z, max(2, nstep))
    Ez = lcdm_E(z_arr, om)
    integ = np.trapz(1.0 / Ez, z_arr)
    return C_KM_S / H0 * integ

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', choices=['mpc', 'mpch'], default='mpc')
    ap.add_argument('--smin', type=float, default=20.0)
    ap.add_argument('--smax', type=float, default=120.0)
    ap.add_argument('--zeff', type=float, default=0.8)
    ap.add_argument('--fid_H0', type=float, default=67.66)
    ap.add_argument('--fid_om', type=float, default=0.3111)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    zgrid = np.load(GRID_NPZ)
    mu_edges = zgrid['mu_edges']
    unit = str(zgrid.get('unit', args.unit))
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    mp = pd.read_csv(MULTIPOLES_CSV)
    s = mp[f's_center_{unit}'].to_numpy(dtype=float)
    xi0 = mp['xi0'].to_numpy(dtype=float)
    xi2 = mp['xi2'].to_numpy(dtype=float)
    j = np.load(JK_NPZ)
    xi0_jk = j['xi0_jk']
    xi2_jk = j['xi2_jk']
    s_jk = j['s_centers']
    if not np.allclose(s, s_jk):
        xi0_jk = np.array([np.interp(s, s_jk, row) for row in xi0_jk])
        xi2_jk = np.array([np.interp(s, s_jk, row) for row in xi2_jk])
    mask = (s >= args.smin) & (s <= args.smax)
    s_fit = s[mask]
    dvec = np.concatenate([xi0[mask], xi2[mask]], axis=0)
    K = xi0_jk.shape[0]
    mean0 = xi0_jk[:, mask].mean(axis=0)
    mean2 = xi2_jk[:, mask].mean(axis=0)
    X = np.concatenate([xi0_jk[:, mask] - mean0, xi2_jk[:, mask] - mean2], axis=1)
    cov = (K - 1) / K * (X.T @ X)
    p_dim = dvec.size
    lam = min(0.9, max(0.1, 1.0 - (K - 1) / (p_dim + 1.0)))
    D = np.diag(np.diag(cov))
    cov_reg = (1.0 - lam) * cov + lam * D
    try:
        w, V = np.linalg.eigh(cov_reg)
        eps = 1e-10 * (np.trace(cov_reg) / max(1, p_dim))
        w = np.maximum(w, eps)
        cov_reg = V * w @ V.T
    except Exception:
        pass
    xi0_sm = smooth_template(s, xi0)
    z_eff = float(args.zeff)
    H_fid = args.fid_H0 * lcdm_E(z_eff, args.fid_om)
    DM_fid = lcdm_DM_Mpc(z_eff, args.fid_H0, args.fid_om)
    data_npz = os.path.join(OUT_DIR, 'voids_vgcf_data.npz')
    np.savez(data_npz, s=s, s_fit=s_fit, s_mask=mask, mu_edges=mu_edges, mu_centers=mu_centers, dvec=dvec, cov=cov, cov_reg=cov_reg, shrinkage_lambda=lam, xi0_sm=xi0_sm, unit=unit, z_eff=z_eff, H_fid=H_fid, DM_fid=DM_fid, fid_H0=args.fid_H0, fid_om=args.fid_om)
    like_py = os.path.join(OUT_DIR, 'likelihood_voids_vgcf.py')
    like_code = '# Cobaya likelihood for VGCF (template-based RSD/AP model)\nimport numpy as np\nfrom cobaya.likelihood import Likelihood\n\ndef _interp_template(s_grid, xi0_values):\n    def f(s):\n        return np.interp(s, s_grid, xi0_values, left=xi0_values[0], right=xi0_values[-1])\n    return f\n\nclass VoidsVGCF(Likelihood):\n    data_file: str\n\n    def initialize(self):\n        z = np.load(self.data_file)\n        self.s = z["s"]; self.s_fit = z["s_fit"]; self.s_mask = z["s_mask"]\n        self.mu_edges = z["mu_edges"]; self.mu_centers = z["mu_centers"]\n        self.dvec = z["dvec"]\n        self.cov = z["cov_reg"] if "cov_reg" in z.files else z["cov"]\n        try:\n            self.unit = z["unit"].item()\n        except Exception:\n            self.unit = str(z["unit"])\n        self.tpl = _interp_template(self.s, z["xi0_sm"])\n        U, S, Vt = np.linalg.svd(self.cov, full_matrices=False)\n        self.inv_cov = (U * (1.0 / np.maximum(S, 1e-12))) @ Vt\n\n    def get_requirements(self):\n        return {}\n\n    def _model_grid(self, A, beta, alpha_par, alpha_perp, C):\n        S, MU = np.meshgrid(self.s, self.mu_centers, indexing=\\"ij\\")\n        s_par = alpha_par * S * MU\n        s_perp = alpha_perp * S * np.sqrt(np.maximum(0.0, 1.0 - MU**2))\n        Sprime = np.sqrt(s_par**2 + s_perp**2)\n        MUprime = np.divide(s_par, Sprime, out=np.zeros_like(s_par), where=Sprime>0)\n        Xi_iso = self.tpl(Sprime)\n        Xi = A * Xi_iso * (1.0 + beta * MUprime**2) ** 2 + C\n        return Xi\n\n    def _model_vector(self, A, beta, alpha_par, alpha_perp, C):\n        Xi = self._model_grid(A, beta, alpha_par, alpha_perp, C)\n        dmu = 1.0 / len(self.mu_centers)\n        p2 = 0.5 * (3.0 * self.mu_centers**2 - 1.0)\n        xi0m = np.sum(Xi, axis=1) * (2.0 * dmu)\n        xi2m = 5.0 * np.sum(Xi * p2[None, :], axis=1) * (2.0 * dmu)\n        xi0m = xi0m[self.s_mask]; xi2m = xi2m[self.s_mask]\n        return np.concatenate([xi0m, xi2m], axis=0)\n\n    def logp(self, **params):\n        A = params.get(\\"A\\", 1.0); beta = params.get(\\"beta\\", 0.3)\n        alpha_par = params.get(\\"alpha_par\\", 1.0); alpha_perp = params.get(\\"alpha_perp\\", 1.0)\n        C = params.get(\\"C\\", 0.0)\n        m = self._model_vector(A, beta, alpha_par, alpha_perp, C)\n        r = m - self.dvec\n        chi2 = float(r @ (self.inv_cov @ r))\n        return -0.5 * chi2\n'
    with open(like_py, 'w', encoding='utf-8') as f:
        f.write(like_code)
    like_cosmo_py = os.path.join(OUT_DIR, 'likelihood_voids_vgcf_cosmo.py')
    like_cosmo_code = '# Cobaya likelihood for VGCF with AP derived from cosmology at z_eff\nimport numpy as np\nfrom cobaya.likelihood import Likelihood\n\ndef _interp_template(s_grid, xi0_values):\n    def f(s):\n        return np.interp(s, s_grid, xi0_values, left=xi0_values[0], right=xi0_values[-1])\n    return f\n\nclass VoidsVGCF_APCosmo(Likelihood):\n    data_file: str\n    use_ap_from_cosmo: bool = True\n\n    def initialize(self):\n        z = np.load(self.data_file)\n        self.s = z["s"]; self.s_mask = z["s_mask"]\n        self.mu_centers = z["mu_centers"]\n        self.dvec = z["dvec"]\n        self.cov = z["cov_reg"] if "cov_reg" in z.files else z["cov"]\n        self.z_eff = float(z["z_eff"])\n        self.H_fid = float(z["H_fid"])\n        self.DM_fid = float(z["DM_fid"])\n        self.tpl = _interp_template(self.s, z["xi0_sm"])\n        U, S, Vt = np.linalg.svd(self.cov, full_matrices=False)\n        self.inv_cov = (U * (1.0 / np.maximum(S, 1e-12))) @ Vt\n\n    def get_requirements(self):\n        # Try to request distances; provider may expose these helpers\n        req = {}\n        try:\n            # Some providers expose methods implicitly; no explicit req needed\n            return req\n        except Exception:\n            return req\n\n    def _get_H_DM(self):\n        H = None; DM = None\n        # Try a few common provider methods\n        try:\n            H = float(self.provider.get_Hubble(self.z_eff))\n        except Exception:\n            try:\n                H = float(self.provider.Hubble(self.z_eff))\n            except Exception:\n                pass\n        try:\n            DA = float(self.provider.get_angular_diameter_distance(self.z_eff))\n            DM = DA * (1.0 + self.z_eff)\n        except Exception:\n            try:\n                DM = float(self.provider.get_comoving_angular_distance(self.z_eff))\n            except Exception:\n                pass\n        return H, DM\n\n    def _alphas(self):\n        if not getattr(self, \'use_ap_from_cosmo\', True):\n            return None, None\n        H, DM = self._get_H_DM()\n        if (H is None) or (DM is None):\n            return None, None\n        alpha_par = self.H_fid / H\n        alpha_perp = DM / self.DM_fid\n        return float(alpha_par), float(alpha_perp)\n\n    def _model_grid(self, A, beta, alpha_par, alpha_perp, C):\n        S, MU = np.meshgrid(self.s, self.mu_centers, indexing=\\"ij\\")\n        s_par = alpha_par * S * MU\n        s_perp = alpha_perp * S * np.sqrt(np.maximum(0.0, 1.0 - MU**2))\n        Sprime = np.sqrt(s_par**2 + s_perp**2)\n        MUprime = np.divide(s_par, Sprime, out=np.zeros_like(s_par), where=Sprime>0)\n        Xi_iso = self.tpl(Sprime)\n        Xi = A * Xi_iso * (1.0 + beta * MUprime**2) ** 2 + C\n        return Xi\n\n    def _model_vector(self, A, beta, alpha_par, alpha_perp, C):\n        Xi = self._model_grid(A, beta, alpha_par, alpha_perp, C)\n        dmu = 1.0 / len(self.mu_centers)\n        p2 = 0.5 * (3.0 * self.mu_centers**2 - 1.0)\n        xi0m = np.sum(Xi, axis=1) * (2.0 * dmu)\n        xi2m = 5.0 * np.sum(Xi * p2[None, :], axis=1) * (2.0 * dmu)\n        xi0m = xi0m[self.s_mask]; xi2m = xi2m[self.s_mask]\n        return np.concatenate([xi0m, xi2m], axis=0)\n\n    def logp(self, **params):\n        A = params.get(\\"A\\", 1.0); beta = params.get(\\"beta\\", 0.3)\n        C = params.get(\\"C\\", 0.0)\n        ap = self._alphas()\n        if (ap is not None) and (ap[0] is not None):\n            alpha_par, alpha_perp = ap\n        else:\n            alpha_par = params.get(\\"alpha_par\\", 1.0)\n            alpha_perp = params.get(\\"alpha_perp\\", 1.0)\n        m = self._model_vector(A, beta, alpha_par, alpha_perp, C)\n        r = m - self.dvec\n        chi2 = float(r @ (self.inv_cov @ r))\n        return -0.5 * chi2\n'
    with open(like_cosmo_py, 'w', encoding='utf-8') as f:
        f.write(like_cosmo_code)
    example_yaml_path = os.path.join(OUT_DIR, 'example.yaml')
    yaml_template = '# Minimal Cobaya config for the VGCF likelihood\n# pip install cobaya\n# Run: cobaya-run "{example_yaml}"\n\nlikelihood:\n  voids_vgcf:\n    class: "likelihood_voids_vgcf.VoidsVGCF"\n    data_file: "{data_file}"\n\nparams:\n  A:\n    prior: {{min: 0.05, max: 5.0}}\n    ref: 1.0\n  beta:\n    prior: {{min: -0.2, max: 1.5}}\n    ref: 0.3\n  alpha_par:\n    prior: {{min: 0.9, max: 1.1}}\n    ref: 1.0\n  alpha_perp:\n    prior: {{min: 0.9, max: 1.1}}\n    ref: 1.0\n  C:\n    prior: {{min: -0.05, max: 0.05}}\n    ref: 0.0\n\nsampler:\n  mcmc:\n    burn_in: 200\n    max_tries: 5\n    Rminus1_stop: 0.02\n    covmat: auto\n\noutput: "{output_dir}"\n'
    final_yaml = yaml_template.format(example_yaml=example_yaml_path.replace('\\', '/'), data_file=data_npz.replace('\\', '/'), output_dir=os.path.join(OUT_DIR, 'chains').replace('\\', '/'))
    with open(example_yaml_path, 'w', encoding='utf-8') as f:
        f.write(final_yaml)
    example_cosmo_yaml = os.path.join(OUT_DIR, 'example_cosmo.yaml')
    yaml_cosmo = '# Cobaya config: VGCF with AP derived from CLASS at z_eff\n# pip install cobaya classy\nlikelihood:\n  voids_vgcf:\n    class: "likelihood_voids_vgcf_cosmo.VoidsVGCF_APCosmo"\n    data_file: "{data_file}"\n    use_ap_from_cosmo: true\n\ntheory:\n  classy:\n    extra_args:\n      N_ncdm: 1\n\nparams:\n  # Cosmology (broad priors for demo; refine as needed)\n  omega_b:\n    prior: {{min: 0.018, max: 0.026}}\n    ref: 0.022\n  omega_cdm:\n    prior: {{min: 0.09, max: 0.15}}\n    ref: 0.12\n  h:\n    prior: {{min: 0.6, max: 0.8}}\n    ref: 0.67\n  m_ncdm:\n    prior: {{min: 0.0, max: 0.6}}\n    ref: 0.06\n  A_s:\n    prior: {{min: 1.5e-9, max: 3.5e-9}}\n    ref: 2.1e-9\n  n_s:\n    prior: {{min: 0.9, max: 1.05}}\n    ref: 0.965\n\n  # VGCF nuisance\n  A:\n    prior: {{min: 0.05, max: 5.0}}\n    ref: 1.0\n  beta:\n    prior: {{min: -0.2, max: 1.5}}\n    ref: 0.3\n  C:\n    prior: {{min: -0.05, max: 0.05}}\n    ref: 0.0\n\nsampler:\n  mcmc:\n    burn_in: 200\n    max_tries: 5\n    Rminus1_stop: 0.03\n    covmat: auto\n\noutput: "{output_dir}"\n'
    final_yaml_cosmo = yaml_cosmo.format(data_file=data_npz.replace('\\', '/'), output_dir=os.path.join(OUT_DIR, 'chains_cosmo').replace('\\', '/'))
    with open(example_cosmo_yaml, 'w', encoding='utf-8') as f:
        f.write(final_yaml_cosmo)
    print(f'Wrote data: {data_npz}')
    print(f'Wrote likelihood: {like_py}')
    print(f'Wrote cosmo likelihood: {like_cosmo_py}')
    print(f'Wrote example YAML: {example_yaml_path}')
    print(f'Wrote example cosmo YAML: {example_cosmo_yaml}')
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
