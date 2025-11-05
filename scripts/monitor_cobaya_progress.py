# kapitel: start
from __future__ import annotations
import argparse
import os
import time
import re
from typing import Dict, List, Optional, Tuple
Number = float

def normalize_key(s: str) -> str:
    return re.sub('[^a-z0-9]+', '', s.lower())

def parse_progress(path: str) -> Tuple[List[str], List[List[str]]]:
    header: Optional[List[str]] = None
    rows: List[List[str]] = []
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if header is None:
                    header = parts
                elif len(parts) >= len(header):
                    rows.append(parts[:len(header)])
        if header is None:
            return ([], [])
        return (header, rows)
    except FileNotFoundError:
        return ([], [])

def find_columns(header: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    norm = [normalize_key(h) for h in header]
    cand_r1 = {'rminus1', 'r1', 'r_1', 'gelmanr1', 'r1mean', 'r1means'}
    cand_r1cl = {'rminus1cl', 'r1cl', 'r_1cl', 'gelmanr1cl'}
    cand_acc = {'acceptance', 'acceptancerate', 'accrate', 'acc', 'accept'}
    cand_n = {'n', 'naccepted', 'accepted', 'nsamples', 'samples', 'npoints'}
    cand_time = {'time', 'elapsed', 'seconds', 'secs', 't'}

    def first_match(cands: set[str]) -> Optional[int]:
        for i, k in enumerate(norm):
            if k in cands:
                return i
        return None
    for key, cands in (('r1', cand_r1), ('r1cl', cand_r1cl), ('acc', cand_acc), ('n', cand_n), ('time', cand_time)):
        idx = first_match(cands)
        if idx is not None:
            mapping[key] = idx
    return mapping

def safe_float(s: str) -> Optional[Number]:
    try:
        return float(s)
    except Exception:
        m = re.search('[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?', s)
        if m:
            try:
                return float(m.group(0))
            except Exception:
                return None
        return None

def fmt_eta(seconds: float) -> str:
    if seconds is None or seconds != seconds or seconds <= 0:
        return '--'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f'{h}h {m}m'
    if m > 0:
        return f'{m}m {s}s'
    return f'{s}s'

def estimate_eta(rows: List[List[str]], col: Dict[str, int], target_r1: Optional[float], max_samples: Optional[int], file_ctime: Optional[float]) -> Tuple[Optional[float], Dict[str, Optional[Number]]]:
    now = time.time()

    def series(key: str) -> List[Number]:
        idx = col.get(key)
        if idx is None:
            return []
        vals: List[Number] = []
        for r in rows:
            v = safe_float(r[idx])
            if v is not None:
                vals.append(v)
        return vals
    r1_vals = series('r1')
    n_vals = series('n')
    t_vals_raw = series('time')
    t_vals: List[Number] = []
    if t_vals_raw:
        t_vals = t_vals_raw
    elif file_ctime is not None and rows:
        t_vals = [i * 60.0 for i in range(len(rows))]
    diag: Dict[str, Optional[Number]] = {'r1': r1_vals[-1] if r1_vals else None, 'n': n_vals[-1] if n_vals else None}
    eta_r1: Optional[float] = None
    eta_n: Optional[float] = None
    if target_r1 is not None and r1_vals and (len(r1_vals) >= 3):
        k = min(8, len(r1_vals))
        r = r1_vals[-k:]
        if t_vals and len(t_vals) == len(r1_vals):
            t = t_vals[-k:]
        else:
            t = list(range(len(r)))[-k:]
        t_mean = sum(t) / len(t)
        r_mean = sum(r) / len(r)
        var_t = sum(((ti - t_mean) ** 2 for ti in t))
        if var_t > 0:
            cov_tr = sum(((ti - t_mean) * (ri - r_mean) for ti, ri in zip(t, r)))
            slope = cov_tr / var_t
            if slope < 0 and r[-1] > target_r1:
                eta_r1 = (r[-1] - target_r1) / -slope
                if not t_vals or len(t_vals) != len(r1_vals):
                    eta_r1 *= 60.0
        diag['r1_slope'] = slope if var_t > 0 else None
    else:
        diag['r1_slope'] = None
    if max_samples is not None and n_vals and (len(n_vals) >= 2):
        n1, n2 = (n_vals[-2], n_vals[-1])
        if t_vals and len(t_vals) == len(n_vals):
            dt = t_vals[-1] - t_vals[-2]
        else:
            dt = 60.0
        dn = n2 - n1
        if dt > 0 and dn > 0:
            rate = dn / dt
            remaining = max(0.0, max_samples - n2)
            eta_n = remaining / rate if rate > 0 else None
            diag['sample_rate_sps'] = rate
        else:
            diag['sample_rate_sps'] = None
    else:
        diag['sample_rate_sps'] = None
    etas = [x for x in (eta_r1, eta_n) if x is not None and x == x and (x > 0)]
    eta = max(etas) if etas else eta_r1 or eta_n
    return (eta, diag)

def main():
    ap = argparse.ArgumentParser(description='Monitor Cobaya .progress and estimate ETA')
    ap.add_argument('--prefix', required=True, help='Output prefix (without .progress)')
    ap.add_argument('--target-rminus1', type=float, default=None, help='Target R-1 threshold (e.g. 0.02)')
    ap.add_argument('--max-samples', type=int, default=None, help='Target max_samples (accepted steps)')
    ap.add_argument('--interval', type=int, default=30, help='Polling interval in seconds')
    args = ap.parse_args()
    progress_path = args.prefix + '.progress'
    print(f'Monitoring: {progress_path}')
    if args.target_rminus1 is not None:
        print(f'Target R-1: {args.target_rminus1}')
    if args.max_samples is not None:
        print(f'Target max_samples: {args.max_samples}')
    last_mtime = 0.0
    while True:
        try:
            st = os.stat(progress_path)
            mtime = st.st_mtime
            ctime = st.st_ctime
        except FileNotFoundError:
            print('Waiting for progress file...')
            time.sleep(args.interval)
            continue
        if mtime != last_mtime:
            header, rows = parse_progress(progress_path)
            if not header or not rows:
                print('Reading progress... (no data yet)')
            else:
                cols = find_columns(header)
                eta, diag = estimate_eta(rows, cols, args.target_rminus1, args.max_samples, ctime)
                r1 = diag.get('r1')
                n = diag.get('n')
                rate = diag.get('sample_rate_sps')
                r1_slope = diag.get('r1_slope')
                msg = []
                if r1 is not None:
                    msg.append(f'R-1 {r1:.3f}')
                if n is not None and args.max_samples:
                    msg.append(f'samples {int(n):,}/{args.max_samples:,}')
                if rate is not None:
                    msg.append(f'rate {rate * 60:.1f}/min')
                if r1_slope is not None:
                    msg.append(f'd(R-1)/hr {r1_slope * 3600:.3f}')
                msg.append(f'ETA {fmt_eta(eta)}')
                print(' | '.join(msg))
            last_mtime = mtime
        time.sleep(args.interval)
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
