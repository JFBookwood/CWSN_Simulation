#start
import sys
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
            tW = targets[idx] * W
            if w == 0:
                out.append(v)
            else:
                frac = (tW - prev) / w
                frac = 0.0 if frac < 0 else 1.0 if frac > 1 else frac
                out.append(v)
            idx += 1
        if idx >= len(targets):
            break
    while len(out) < len(ps):
        out.append(vals[-1])
    return out

def read_chain_txt(path):
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
                if 'mnu' in cols:
                    mnu_idx = cols.index('mnu')
                elif 'm_ncdm' in cols:
                    mnu_idx = cols.index('m_ncdm')
                else:
                    pass
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
    if len(sys.argv) < 2:
        print('Usage: compute_mnu_stats.py <chain_txt_file>')
        sys.exit(2)
    chain_path = Path(sys.argv[1])
    if not chain_path.exists():
        print(f'File not found: {chain_path}')
        sys.exit(2)
    mnu, wt = read_chain_txt(chain_path)
    if not mnu:
        print('No mnu samples found in chain.')
        sys.exit(1)
    mu, sd = weighted_stats(mnu, wt)
    p16, p50, p84 = weighted_percentiles(mnu, wt, ps=(0.16, 0.5, 0.84))
    print(f'mnu (mean ± 1σ): {mu:.6g} ± {sd:.3g}')
    print(f'mnu (median [16%,84%]): {p50:.6g} [{p16:.6g}, {p84:.6g}]')
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
