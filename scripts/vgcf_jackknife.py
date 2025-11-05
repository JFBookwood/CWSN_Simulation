# kapitel: start
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
ROOT = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN'
VIDE_DIR = os.path.join(ROOT, 'data', 'desi', 'edr', 'processed', 'vide_input', 'mpc')
VOID_DIR = os.path.join(ROOT, 'data', 'desi', 'edr', 'processed', 'voidfinder')
DATA_FILE = os.path.join(VIDE_DIR, 'data.txt')
RANDS_FILE = os.path.join(VIDE_DIR, 'randoms.txt')
VOIDS_CSV = os.path.join(VOID_DIR, 'catalog.csv')
OUT_DIR = os.path.join(VOID_DIR, 'vgcf')

def load_xyz_txt(path: str) -> np.ndarray:
    try:
        arr = np.loadtxt(path, dtype=np.float64)
    except Exception:
        rows = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                s = line.strip().split()
                if len(s) != 3:
                    continue
                try:
                    rows.append([float(s[0]), float(s[1]), float(s[2])])
                except ValueError:
                    continue
        arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
    return arr

def unit_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n <= 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n

def compute_multipoles(gal, ran, centers, s_edges, mu_edges):
    nn_g = NearestNeighbors(algorithm='ball_tree')
    nn_r = NearestNeighbors(algorithm='ball_tree')
    nn_g.fit(gal)
    nn_r.fit(ran)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    NG = np.zeros((len(s_centers), len(mu_centers)), dtype=np.float64)
    NR = np.zeros_like(NG)
    alpha = len(gal) / float(len(ran))
    for i, c in enumerate(centers):
        if i % max(1, len(centers) // 10) == 0:
            pass
        los = unit_vector(c)
        _, idx_g = nn_g.radius_neighbors(c.reshape(1, -1), radius=s_edges[-1], return_distance=True)
        if idx_g[0].size:
            vec = gal[idx_g[0]] - c
            s = np.linalg.norm(vec, axis=1)
            mask = (s >= s_edges[0]) & (s <= s_edges[-1])
            if np.any(mask):
                s = s[mask]
                vec = vec[mask]
                mu = np.abs(np.sum(vec * los, axis=1) / np.maximum(s, 1e-12))
                H, _, _ = np.histogram2d(s, mu, bins=[s_edges, mu_edges])
                NG += H
        _, idx_r = nn_r.radius_neighbors(c.reshape(1, -1), radius=s_edges[-1], return_distance=True)
        if idx_r[0].size:
            vec = ran[idx_r[0]] - c
            s = np.linalg.norm(vec, axis=1)
            mask = (s >= s_edges[0]) & (s <= s_edges[-1])
            if np.any(mask):
                s = s[mask]
                vec = vec[mask]
                mu = np.abs(np.sum(vec * los, axis=1) / np.maximum(s, 1e-12))
                H, _, _ = np.histogram2d(s, mu, bins=[s_edges, mu_edges])
                NR += H
    mask = NR > 0
    XI = np.full_like(NG, np.nan)
    XI[mask] = NG[mask] / (alpha * NR[mask]) - 1.0
    dmu = 1.0 / len(mu_centers)
    p2 = 0.5 * (3.0 * mu_centers ** 2 - 1.0)
    xi0 = np.full(len(s_centers), np.nan)
    xi2 = np.full(len(s_centers), np.nan)
    for i_s in range(len(s_centers)):
        valid = np.isfinite(XI[i_s])
        if not np.any(valid):
            continue
        xi_mu = XI[i_s, valid]
        mu_c = mu_centers[valid]
        p2v = 0.5 * (3.0 * mu_c ** 2 - 1.0)
        xi0[i_s] = np.sum(xi_mu) * (2.0 * dmu)
        xi2[i_s] = 5.0 * np.sum(xi_mu * p2v) * (2.0 * dmu)
    return (xi0, xi2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA_FILE)
    ap.add_argument('--randoms', default=RANDS_FILE)
    ap.add_argument('--voids', default=VOIDS_CSV)
    ap.add_argument('--unit', choices=['mpc', 'mpch'], default='mpc')
    ap.add_argument('--smax', type=float, default=120.0)
    ap.add_argument('--smin', type=float, default=0.0)
    ap.add_argument('--ns', type=int, default=24)
    ap.add_argument('--nmu', type=int, default=16)
    ap.add_argument('--rand_subsample', type=int, default=500000)
    ap.add_argument('--jackknife', type=int, default=20)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    gal = load_xyz_txt(args.data)
    ran = load_xyz_txt(args.randoms)
    vdf = pd.read_csv(args.voids)
    centers = vdf[['x', 'y', 'z']].to_numpy(dtype=np.float64)
    if args.rand_subsample and len(ran) > args.rand_subsample:
        idx = np.random.choice(len(ran), size=args.rand_subsample, replace=False)
        ran = ran[idx]
        print(f'Subsampled randoms to {len(ran):,}')
    print(f'Jackknife K={args.jackknife}; N_voids={len(centers):,}')
    s_edges = np.linspace(args.smin, args.smax, int(args.ns) + 1)
    mu_edges = np.linspace(0.0, 1.0, int(args.nmu) + 1)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    K = int(args.jackknife)
    idx_all = np.arange(len(centers))
    rng = np.random.default_rng(12345)
    perm = rng.permutation(idx_all)
    xi0_jk = np.zeros((K, len(s_centers)), dtype=np.float64)
    xi2_jk = np.zeros_like(xi0_jk)
    for k in range(K):
        mask_keep = np.ones(len(centers), dtype=bool)
        mask_keep[perm[k::K]] = False
        c_keep = centers[mask_keep]
        xi0, xi2 = compute_multipoles(gal, ran, c_keep, s_edges, mu_edges)
        xi0_jk[k] = xi0
        xi2_jk[k] = xi2
        print(f'  JK {k + 1}/{K} done; kept {len(c_keep)} voids')
    out_npz = os.path.join(OUT_DIR, 'multipoles_jk.npz')
    np.savez(out_npz, xi0_jk=xi0_jk, xi2_jk=xi2_jk, s_centers=s_centers, unit=args.unit)
    print(f'Wrote: {out_npz}')
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
