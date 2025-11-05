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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA_FILE)
    ap.add_argument('--randoms', default=RANDS_FILE)
    ap.add_argument('--voids', default=VOIDS_CSV)
    ap.add_argument('--unit', choices=['mpc', 'mpch'], default='mpc')
    ap.add_argument('--smax', type=float, default=120.0, help='Max separation for pairs')
    ap.add_argument('--smin', type=float, default=0.0, help='Min separation for pairs')
    ap.add_argument('--ns', type=int, default=24, help='# radial bins')
    ap.add_argument('--nmu', type=int, default=12, help='# mu bins in [0,1]')
    ap.add_argument('--rand_subsample', type=int, default=500000)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Loading galaxies: {args.data}')
    gal = load_xyz_txt(args.data)
    print(f'Loading randoms: {args.randoms}')
    ran = load_xyz_txt(args.randoms)
    print(f'Loading voids: {args.voids}')
    vdf = pd.read_csv(args.voids)
    centers = vdf[['x', 'y', 'z']].to_numpy(dtype=np.float64)
    if args.rand_subsample and len(ran) > args.rand_subsample:
        idx = np.random.choice(len(ran), size=args.rand_subsample, replace=False)
        ran = ran[idx]
        print(f'Subsampled randoms to {len(ran):,}')
    print(f'N_gal={len(gal):,}; N_ran={len(ran):,}; N_voids={len(centers):,}')
    nn_g = NearestNeighbors(algorithm='ball_tree')
    nn_r = NearestNeighbors(algorithm='ball_tree')
    nn_g.fit(gal)
    nn_r.fit(ran)
    s_edges = np.linspace(args.smin, args.smax, int(args.ns) + 1)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    mu_edges = np.linspace(0.0, 1.0, int(args.nmu) + 1)
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    NG = np.zeros((len(s_centers), len(mu_centers)), dtype=np.float64)
    NR = np.zeros_like(NG)
    alpha = len(gal) / float(len(ran))
    print('Accumulating (s,mu) pair counts...')
    for i, c in enumerate(centers):
        if i % max(1, len(centers) // 20) == 0:
            print(f'  progress: {i}/{len(centers)} ({100.0 * i / len(centers):.1f}%)')
        los = unit_vector(c)
        dg, idx_g = nn_g.radius_neighbors(c.reshape(1, -1), radius=args.smax, return_distance=True)
        if idx_g[0].size:
            vec = gal[idx_g[0]] - c
            s = np.linalg.norm(vec, axis=1)
            mask = (s >= args.smin) & (s <= args.smax)
            if np.any(mask):
                s = s[mask]
                vec = vec[mask]
                mu = np.abs(np.sum(vec * los, axis=1) / np.maximum(s, 1e-12))
                H, _, _ = np.histogram2d(s, mu, bins=[s_edges, mu_edges])
                NG += H
        dr, idx_r = nn_r.radius_neighbors(c.reshape(1, -1), radius=args.smax, return_distance=True)
        if idx_r[0].size:
            vec = ran[idx_r[0]] - c
            s = np.linalg.norm(vec, axis=1)
            mask = (s >= args.smin) & (s <= args.smax)
            if np.any(mask):
                s = s[mask]
                vec = vec[mask]
                mu = np.abs(np.sum(vec * los, axis=1) / np.maximum(s, 1e-12))
                H, _, _ = np.histogram2d(s, mu, bins=[s_edges, mu_edges])
                NR += H
    mask = NR > 0
    XI = np.full_like(NG, np.nan)
    XI[mask] = NG[mask] / (alpha * NR[mask]) - 1.0
    grid_npz = os.path.join(OUT_DIR, 'xi_grid.npz')
    np.savez(grid_npz, s_edges=s_edges, mu_edges=mu_edges, xi=XI, NG=NG, NR=NR, alpha=alpha, unit=args.unit)
    print(f'Wrote: {grid_npz}')
    dmu = 1.0 / len(mu_centers)
    P2 = 0.5 * (3.0 * mu_centers ** 2 - 1.0)
    xi0 = np.full(len(s_centers), np.nan)
    xi2 = np.full(len(s_centers), np.nan)
    for i_s in range(len(s_centers)):
        valid = ~np.isnan(XI[i_s])
        if not np.any(valid):
            continue
        xi_mu = XI[i_s, valid]
        mu_c = mu_centers[valid]
        p2 = 0.5 * (3.0 * mu_c ** 2 - 1.0)
        xi0[i_s] = np.sum(xi_mu) * (2.0 * dmu)
        xi2[i_s] = 5.0 * np.sum(xi_mu * p2) * (2.0 * dmu)
    out = pd.DataFrame({'s_center_' + args.unit: s_centers, 'xi0': xi0, 'xi2': xi2, 'N_pairs_g': NG.sum(axis=1), 'N_pairs_r': NR.sum(axis=1), 'N_voids': len(centers)})
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, 'multipoles.csv')
    out.to_csv(out_csv, index=False)
    print(f'Wrote: {out_csv}')
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(s_centers, xi0, label='$\\xi_0$')
        plt.plot(s_centers, xi2, label='$\\xi_2$')
        plt.axhline(0, color='k', lw=1)
        plt.xlabel(f's [{args.unit}]')
        plt.ylabel('$\\xi_\\ell$')
        plt.legend()
        plt.tight_layout()
        out_png = os.path.join(OUT_DIR, 'multipoles.png')
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f'Wrote: {out_png}')
    except Exception as e:
        print(f'[Info] Matplotlib not available or plotting failed ({e}); CSV written.)')
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
