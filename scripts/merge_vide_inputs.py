# kapitel: start
import os
import json
import argparse
import shutil
import numpy as np
from typing import Tuple
ROOT = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN'
PROC = os.path.join(ROOT, 'data', 'desi', 'edr', 'processed')
DEFAULT_DATA = os.path.join(PROC, 'ELG_HIP_positions_xyz.txt')
DEFAULT_RANDOMS_MPC = os.path.join(PROC, 'ELG_HIP_randoms_xyz.txt')
DEFAULT_RANDOMS_MPCH = os.path.join(PROC, 'ELG_HIP_randoms_xyz_mpch.txt')

def load_stats(path: str) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 3:
        raise ValueError(f'Expected 3 columns in {path}, found {arr.shape[1]}')
    n = arr.shape[0]
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    return (n, mins, maxs, means, stds)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--unit', choices=['mpc', 'mpch'], default='mpc', help='Units of inputs and outputs')
    ap.add_argument('--data', type=str, default=None, help='Path to data xyz txt (defaults per unit)')
    ap.add_argument('--randoms', type=str, default=None, help='Path to randoms xyz txt (defaults per unit)')
    args = ap.parse_args()
    unit = args.unit
    data_src = args.data or DEFAULT_DATA
    if unit == 'mpc':
        rand_src = args.randoms or DEFAULT_RANDOMS_MPC
    else:
        rand_src = args.randoms or DEFAULT_RANDOMS_MPCH
    if not os.path.exists(data_src):
        raise FileNotFoundError(f'Data file not found: {data_src}')
    if not os.path.exists(rand_src):
        raise FileNotFoundError(f'Randoms file not found: {rand_src}')
    out_dir = os.path.join(PROC, 'vide_input', unit)
    os.makedirs(out_dir, exist_ok=True)
    data_dst = os.path.join(out_dir, 'data.txt')
    rand_dst = os.path.join(out_dir, 'randoms.txt')
    shutil.copyfile(data_src, data_dst)
    shutil.copyfile(rand_src, rand_dst)
    print(f'Computing stats (this may take a moment for randoms)...')
    n_data, d_min, d_max, d_mean, d_std = load_stats(data_dst)
    n_rand, r_min, r_max, r_mean, r_std = load_stats(rand_dst)

    def vec(v):
        return f'[{v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}]'
    print('\nSummary:')
    print(f'  Unit: {unit}')
    print(f'  Data:    N={n_data:,}  min={vec(d_min)}  max={vec(d_max)}  mean={vec(d_mean)}  std={vec(d_std)}')
    print(f'  Randoms: N={n_rand:,}  min={vec(r_min)}  max={vec(r_max)}  mean={vec(r_mean)}  std={vec(r_std)}')
    d_rmean = float(np.linalg.norm(d_mean))
    r_rmean = float(np.linalg.norm(r_mean))
    if d_rmean > 0 and r_rmean > 0:
        ratio = max(d_rmean, r_rmean) / min(d_rmean, r_rmean)
        if ratio > 1.3 and ratio < 1.7:
            print('\n[Warning] The mean radius scales differ by ~h (~0.67). Check that both inputs use the same unit (mpc vs mpch).')
    manifest = {'unit': unit, 'sources': {'data': data_src, 'randoms': rand_src}, 'outputs': {'data': data_dst, 'randoms': rand_dst}, 'stats': {'data': {'N': int(n_data), 'min': d_min.tolist(), 'max': d_max.tolist(), 'mean': d_mean.tolist(), 'std': d_std.tolist()}, 'randoms': {'N': int(n_rand), 'min': r_min.tolist(), 'max': r_max.tolist(), 'mean': r_mean.tolist(), 'std': r_std.tolist()}}}
    with open(os.path.join(out_dir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f'\nWrote structured inputs to: {out_dir}')
    print('Files:')
    print(f'  - {data_dst}')
    print(f'  - {rand_dst}')
    print(f"  - {os.path.join(out_dir, 'manifest.json')}")
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
