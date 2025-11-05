# kapitel: start
import os
import sys
import argparse
import numpy as np
IN_CSV = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN\\\\data\\\\desi\\\\edr\\\\processed\\\\ELG_HIP_positions_xyz_mpc.csv'
OUT_TXT = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN\\\\data\\\\desi\\\\edr\\\\processed\\\\ELG_HIP_positions_xyz.txt'
H0 = 67.66
h = H0 / 100.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mpch', action='store_true', help='Convert Mpc -> Mpc/h for output')
    ap.add_argument('--subsample', type=int, default=1, help='Keep 1/N of rows (>=1)')
    args = ap.parse_args()
    if not os.path.exists(IN_CSV):
        print(f'Input CSV not found: {IN_CSV}')
        sys.exit(1)
    print(f'Reading: {IN_CSV}')
    data = np.loadtxt(IN_CSV, delimiter=',', skiprows=1)
    xyz = data[:, :3].copy()
    if args.subsample and args.subsample > 1:
        rng = np.random.default_rng(42)
        mask = rng.random(len(xyz)) < 1.0 / args.subsample
        xyz = xyz[mask]
        print(f'Subsampled by factor {args.subsample}: kept {len(xyz)} rows')
    if args.mpch:
        xyz *= h
        print(f'Converted to Mpc/h with h={h:.4f}')
    np.savetxt(OUT_TXT, xyz, fmt='%.8f')
    print(f'Wrote: {OUT_TXT} (no header, whitespace, columns: x y z)')
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
