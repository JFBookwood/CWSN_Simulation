#start
import sys
import argparse
from math import sqrt
from pathlib import Path

def weighted_stats(values, weights):
    W = sum(weights)
    if W == 0:
        return (float('nan'), float('nan'))
    mu = sum((v * w for v, w in zip(values, weights))) / W
    var = sum((w * (v - mu) ** 2 for v, w in zip(values, weights))) / W
    return (mu, sqrt(var))

def weighted_percentiles(values, weights, ps=(0.16, 0.5, 0.84)):
    order = sorted(zip(values, weights), key=lambda t: t[0])
    vals = [v for v, _ in order]
    wts = [w for _, w in order]
    W = sum(wts)
    if W == 0:
        return [float('nan') for _ in ps]
    cum = 0.0
    out = []
    targets = list(ps)
    idx = 0
    for v, w in order:
        prev = cum
        cum += w
        while idx < len(targets) and cum >= targets[idx] * W:
            out.append(v)
            idx += 1
        if idx >= len(targets):
            break
    while len(out) < len(ps):
        out.append(vals[-1])
    return out

def read_chain_txt(path, mnu_name='mnu'):
    mnu = []
    wt = []
    weight_idx = None
    mnu_idx = None
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip():
                continue
            if line.startswith('#'):
                cols = line[1:].strip().split()
                try:
                    weight_idx = cols.index('weight')
                except ValueError:
                    weight_idx = 0
                if mnu_name in cols:
                    mnu_idx = cols.index(mnu_name)
                elif mnu_name == 'mnu' and 'm_ncdm' in cols:
                    mnu_idx = cols.index('m_ncdm')
                continue
            parts = line.split()
            if weight_idx is None or mnu_idx is None:
                continue
            try:
                w = float(parts[weight_idx])
                v = float(parts[mnu_idx])
            except (IndexError, ValueError):
                continue
            wt.append(w)
            mnu.append(v)
    return (mnu, wt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('chain_txt', help='Cobaya chain text file (*.txt)')
    ap.add_argument('--mnu_name', default='mnu', help='Column name: mnu or m_ncdm')
    ap.add_argument('--min', dest='mnu_min', type=float, default=None, help='Apply hard lower cutoff')
    ap.add_argument('--max', dest='mnu_max', type=float, default=None, help='Apply hard upper cutoff')
    args = ap.parse_args()
    chain_path = Path(args.chain_txt)
    if not chain_path.exists():
        print(f'File not found: {chain_path}')
        sys.exit(2)
    mnu, wt = read_chain_txt(chain_path, mnu_name=args.mnu_name)
    if not mnu:
        print('No mnu-like samples found in chain.')
        sys.exit(1)
    if args.mnu_min is not None or args.mnu_max is not None:
        mnu2, wt2 = ([], [])
        lo = args.mnu_min if args.mnu_min is not None else -float('inf')
        hi = args.mnu_max if args.mnu_max is not None else float('inf')
        for v, w in zip(mnu, wt):
            if lo <= v <= hi:
                mnu2.append(v)
                wt2.append(w)
        mnu, wt = (mnu2, wt2)
    mu, sd = weighted_stats(mnu, wt)
    p16, p50, p84 = weighted_percentiles(mnu, wt, ps=(0.16, 0.5, 0.84))
    print(f'Samples used: {len(mnu)}')
    print(f'mnu (mean ± 1σ): {mu:.6g} ± {sd:.3g}')
    print(f'mnu (median [16%,84%]): {p50:.6g} [{p16:.6g}, {p84:.6g}]')
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
