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
OUT_DIR = os.path.join(VOID_DIR, 'qc')

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA_FILE)
    ap.add_argument('--randoms', default=RANDS_FILE)
    ap.add_argument('--voids', default=VOIDS_CSV)
    ap.add_argument('--unit', choices=['mpc', 'mpch'], default='mpc')
    ap.add_argument('--rand_subsample', type=int, default=500000, help='Subsample randoms for speed (0=disable)')
    ap.add_argument('--rmax_over_rv', type=float, default=3.0)
    ap.add_argument('--nbins', type=int, default=30)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Loading galaxies: {args.data}')
    gal = load_xyz_txt(args.data)
    print(f'Loading randoms: {args.randoms}')
    ran = load_xyz_txt(args.randoms)
    print(f'Loading voids: {args.voids}')
    vdf = pd.read_csv(args.voids)
    centers = vdf[['x', 'y', 'z']].to_numpy(dtype=np.float64)
    radii = vdf[[c for c in vdf.columns if c.startswith('radius_')][0]].to_numpy(dtype=np.float64)
    if args.rand_subsample and len(ran) > args.rand_subsample:
        idx = np.random.choice(len(ran), size=args.rand_subsample, replace=False)
        ran = ran[idx]
        print(f'Subsampled randoms to {len(ran):,}')
    print(f'N_gal={len(gal):,}; N_ran={len(ran):,}; N_voids={len(centers):,}')
    nn_g = NearestNeighbors(algorithm='ball_tree')
    nn_r = NearestNeighbors(algorithm='ball_tree')
    nn_g.fit(gal)
    nn_r.fit(ran)
    nb = int(args.nbins)
    x_max = float(args.rmax_over_rv)
    edges = np.linspace(0.0, x_max, nb + 1)
    num_g = np.zeros(nb, dtype=np.int64)
    num_r = np.zeros(nb, dtype=np.int64)
    alpha = len(gal) / float(len(ran))
    print('Accumulating counts in scaled shells...')
    for i, (c, rv) in enumerate(zip(centers, radii)):
        if i % max(1, len(centers) // 20) == 0:
            print(f'  progress: {i}/{len(centers)} ({100.0 * i / len(centers):.1f}%)')
        rmax = x_max * rv
        d_g, _ = nn_g.radius_neighbors(c.reshape(1, -1), radius=rmax, return_distance=True)
        if d_g[0].size > 0:
            xg = d_g[0] / rv
            hg, _ = np.histogram(xg, bins=edges)
            num_g += hg.astype(np.int64)
        d_r, _ = nn_r.radius_neighbors(c.reshape(1, -1), radius=rmax, return_distance=True)
        if d_r[0].size > 0:
            xr = d_r[0] / rv
            hr, _ = np.histogram(xr, bins=edges)
            num_r += hr.astype(np.int64)
    mask = num_r > 0
    x_centers = 0.5 * (edges[:-1] + edges[1:])
    xi = np.full(nb, np.nan)
    err = np.full(nb, np.nan)
    xi[mask] = num_g[mask] / (alpha * num_r[mask]) - 1.0
    err[mask] = np.sqrt(np.maximum(num_g[mask], 1)) / (alpha * num_r[mask])
    out_csv = os.path.join(OUT_DIR, 'profile_delta.csv')
    out = pd.DataFrame({'x_r_over_Rv': x_centers, 'xi': xi, 'xi_err_Poisson': err, 'N_gal_counts': num_g, 'N_ran_counts': num_r, 'N_voids': len(centers), 'unit': args.unit})
    out.to_csv(out_csv, index=False)
    print(f'Wrote: {out_csv}')
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 4))
        plt.hist(radii, bins=30, histtype='step', color='k')
        plt.xlabel(f'Void radius [{args.unit}]')
        plt.ylabel('Count')
        plt.tight_layout()
        out_png1 = os.path.join(OUT_DIR, 'radius_hist.png')
        plt.savefig(out_png1, dpi=150)
        plt.close()
        plt.figure(figsize=(5, 4))
        plt.errorbar(x_centers[mask], xi[mask], yerr=err[mask], fmt='o', ms=3)
        plt.axhline(0, color='k', lw=1)
        plt.xlabel('r/R_v')
        plt.ylabel('delta = n_g/(alpha n_r) - 1')
        plt.tight_layout()
        out_png2 = os.path.join(OUT_DIR, 'profile_delta.png')
        plt.savefig(out_png2, dpi=150)
        plt.close()
        print(f'Wrote: {out_png1}')
        print(f'Wrote: {out_png2}')
    except Exception as e:
        print(f'[Info] Matplotlib not available or plotting failed ({e}); CSV written.')
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
