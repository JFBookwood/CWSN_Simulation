#start
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
ROOT = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN'
VIDE_DIR = os.path.join(ROOT, 'data', 'desi', 'edr', 'processed', 'vide_input', 'mpc')
DATA_FILE = os.path.join(VIDE_DIR, 'data.txt')
RANDS_FILE = os.path.join(VIDE_DIR, 'randoms.txt')
OUT_DIR = os.path.join(ROOT, 'data', 'desi', 'edr', 'processed', 'voidfinder')
np.random.seed(42)

def load_xyz_txt(path: str) -> np.ndarray:
    try:
        arr = np.loadtxt(path, dtype=np.float64)
    except Exception:
        rows = []
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                try:
                    rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
        arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, 3)
    return arr

def estimate_density_kNN(data: np.ndarray, randoms: np.ndarray, k: int=32) -> np.ndarray:
    nbr = NearestNeighbors(n_neighbors=k, algorithm='auto', n_jobs=-1)
    nbr.fit(randoms)
    dists, _ = nbr.kneighbors(data, n_neighbors=k, return_distance=True)
    rk = dists[:, -1]
    vol = 4.0 / 3.0 * np.pi * np.maximum(rk, 1e-06) ** 3
    dens = k / vol
    med = np.median(dens)
    dens_norm = dens / med if med > 0 else dens
    return np.log10(np.maximum(dens_norm, 1e-12))

def find_minima_candidates(coords: np.ndarray, logdens: np.ndarray, frac: float=0.02, min_separation: float=5.0) -> np.ndarray:
    n = len(coords)
    m = max(10, int(frac * n))
    idx_sorted = np.argsort(logdens)[:m]
    seeds = coords[idx_sorted]
    keep = []
    for s in seeds:
        if not keep:
            keep.append(s)
            continue
        if np.min(np.linalg.norm(np.asarray(keep) - s, axis=1)) >= min_separation:
            keep.append(s)
    return np.asarray(keep)

#sphere
def grow_empty_sphere(center: np.ndarray, data: np.ndarray, randoms: np.ndarray, target_quantile: float=0.99) -> float:
    nbr_data = NearestNeighbors(n_neighbors=1, algorithm='auto')
    nbr_rnd = NearestNeighbors(n_neighbors=128, algorithm='auto')
    nbr_data.fit(data)
    nbr_rnd.fit(randoms)
    r_data, _ = nbr_data.kneighbors(center.reshape(1, -1), n_neighbors=1, return_distance=True)
    r_rnd, _ = nbr_rnd.kneighbors(center.reshape(1, -1), n_neighbors=128, return_distance=True)
    r_data = float(r_data[0, 0])
    r_rnd_q = float(np.quantile(r_rnd[0], target_quantile))
    return max(0.0, min(r_data, r_rnd_q))

def suppress_overlaps(centers: np.ndarray, radii: np.ndarray, overlap_frac: float=0.3) -> np.ndarray:
    order = np.argsort(-radii)
    kept = []
    for i in order:
        c_i, r_i = (centers[i], radii[i])
        if r_i <= 0:
            continue
        accept = True
        for j in kept:
            c_j, r_j = (centers[j], radii[j])
            d = np.linalg.norm(c_i - c_j)
            if d >= r_i + r_j:
                continue
            if d < min(r_i, r_j) * overlap_frac:
                accept = False
                break
        if accept:
            kept.append(i)
    return np.asarray(kept, dtype=int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=DATA_FILE)
    ap.add_argument('--randoms', default=RANDS_FILE)
    ap.add_argument('--unit', choices=['mpc', 'mpch'], default='mpc')
    ap.add_argument('--k', type=int, default=16, help='k for kNN density (in randoms)')
    ap.add_argument('--seed_frac', type=float, default=0.003, help='Fraction of lowest-density points as seeds')
    ap.add_argument('--min_sep', type=float, default=8.0, help='Min center separation for seeds (same unit as coords)')
    ap.add_argument('--target_q', type=float, default=0.99, help='Quantile for random-distance cap when growing sphere')
    ap.add_argument('--max_voids', type=int, default=500, help='Maximum number of voids to output')
    ap.add_argument('--rand_subsample', type=int, default=400000, help='If >0 and randoms are larger, subsample to this size')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Loading data: {args.data}')
    data = load_xyz_txt(args.data)
    print(f'Loading randoms: {args.randoms}')
    randoms = load_xyz_txt(args.randoms)
    if args.rand_subsample and len(randoms) > args.rand_subsample:
        idx = np.random.choice(len(randoms), size=args.rand_subsample, replace=False)
        randoms = randoms[idx]
        print(f'Subsampled randoms to {len(randoms):,}')
    if data.shape[1] != 3 or randoms.shape[1] != 3:
        raise SystemExit('Input files must be whitespace x y z with 3 columns.')
    print(f'Data: N={len(data):,}; Randoms: N={len(randoms):,}')
    if len(randoms) < 10 * len(data):
        print('[Warn] Randoms are fewer than 10x data; densities may be noisy.')
    print('Estimating kNN density...')
    logdens = estimate_density_kNN(data, randoms, k=args.k)
    print('Selecting seed minima...')
    seeds = find_minima_candidates(data, logdens, frac=args.seed_frac, min_separation=args.min_sep)
    print(f'Seeds: {len(seeds):,}')
    from sklearn.neighbors import NearestNeighbors as _NN
    nn_data = _NN(n_neighbors=1, algorithm='auto')
    nn_rnd = _NN(n_neighbors=128, algorithm='auto')
    nn_data.fit(data)
    nn_rnd.fit(randoms)

    def _grow_with_prefit(center):
        d_data, _ = nn_data.kneighbors(center.reshape(1, -1), n_neighbors=2, return_distance=True)
        r0 = float(d_data[0, 0])
        r1 = float(d_data[0, 1]) if d_data.shape[1] > 1 else float(d_data[0, 0])
        r_data = r1 if r0 < 1e-09 else r0
        r_rnd, _ = nn_rnd.kneighbors(center.reshape(1, -1), n_neighbors=128, return_distance=True)
        r_rnd_q = float(np.quantile(r_rnd[0], args.target_q))
        return max(0.0, min(r_data, r_rnd_q))
    print('Growing empty spheres (with progress)...')
    radii = np.empty(len(seeds), dtype=float)
    for i, c in enumerate(seeds):
        if i % max(1, len(seeds) // 20) == 0:
            print(f'  progress: {i}/{len(seeds)} ({100.0 * i / len(seeds):.1f}%)')
        radii[i] = _grow_with_prefit(c)
    print('Suppressing overlaps...')
    idx_keep = suppress_overlaps(seeds, radii, overlap_frac=0.3)
    centers_keep = seeds[idx_keep]
    radii_keep = radii[idx_keep]
    order = np.argsort(-radii_keep)
    order = order[:args.max_voids]
    centers_keep = centers_keep[order]
    radii_keep = radii_keep[order]
    cat_csv = os.path.join(OUT_DIR, 'catalog.csv')
    xyz_txt = os.path.join(OUT_DIR, 'void_centers.xyz')
    df = pd.DataFrame({'x': centers_keep[:, 0], 'y': centers_keep[:, 1], 'z': centers_keep[:, 2], 'radius_' + args.unit: radii_keep})
    df.to_csv(cat_csv, index=False)
    np.savetxt(xyz_txt, centers_keep, fmt='%.6f')
    print(f'Wrote: {cat_csv}')
    print(f'Wrote: {xyz_txt}')
# main
if __name__ == '__main__':
    main()