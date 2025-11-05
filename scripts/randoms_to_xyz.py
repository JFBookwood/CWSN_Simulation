# kapitel: start
import os
import argparse
import numpy as np
from astropy.table import Table, vstack
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
from astropy.io import fits
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
IN_DIR = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN\\\\data\\\\desi\\\\edr\\\\lss\\\\v2.0\\\\LSScats\\\\clustering'
OUT_DIR = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN\\\\data\\\\desi\\\\edr\\\\processed'
CSV_OUT = os.path.join(OUT_DIR, 'ELG_HIP_randoms_xyz_mpc.csv')
TXT_OUT = os.path.join(OUT_DIR, 'ELG_HIP_randoms_xyz.txt')
DATA_TXT = os.path.join(OUT_DIR, 'ELG_HIP_positions_xyz.txt')
DATA_CSV = os.path.join(OUT_DIR, 'ELG_HIP_positions_xyz_mpc.csv')
Z_MIN, Z_MAX = (0.6, 1.6)
H0 = 67.66
h = H0 / 100.0
SKIPPED_FILES: list[str] = []

def list_random_files():
    files = []
    for hemi in ('N', 'S'):
        for i in range(0, 18):
            fname = f'ELG_HIP_{hemi}_{i}_clustering.ran.fits'
            path = os.path.join(IN_DIR, fname)
            if os.path.exists(path):
                files.append(path)
            else:
                print(f'Warning: missing random file: {path}')
    if not files:
        raise FileNotFoundError('No random FITS files found. Make sure to run the download script for randoms.')
    return files

def _read_one_random(p: str):
    try:
        return Table.read(p)
    except Exception as e:
        print(f'[Warn] Astropy Table.read failed on {p} with: {e}')
        print('       Attempting lenient read via fits(memmap=False)...')
        try:
            with fits.open(p, memmap=False) as hdul:
                hdu = None
                for h in hdul:
                    if isinstance(h, (fits.BinTableHDU, fits.TableHDU)):
                        hdu = h
                        break
                if hdu is None:
                    raise RuntimeError(f'No table HDU found in {p}')
                data = hdu.data
                return Table(data)
        except Exception as e2:
            print(f'[Error] Could not read {p} even with memmap=False: {e2}')
            print('        This file is likely truncated/corrupted. Skipping.')
            SKIPPED_FILES.append(p)
            return None

def load_randoms(files, workers: int=1):
    tables = []
    if workers and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(_read_one_random, p): p for p in files}
            for fut in as_completed(futures):
                t = fut.result()
                if t is not None:
                    tables.append(t)
    else:
        for p in files:
            t = _read_one_random(p)
            if t is not None:
                tables.append(t)
    if not tables:
        raise RuntimeError('All random files failed to read; nothing to process.')
    return vstack(tables, join_type='exact')

def radec_to_cartesian(ra_deg, dec_deg, z):
    chi = cosmo.comoving_distance(z).to(u.Mpc).value
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = chi * np.cos(dec) * np.cos(ra)
    y = chi * np.cos(dec) * np.sin(ra)
    zc = chi * np.sin(dec)
    return (x, y, zc)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mpch', action='store_true', help='Write outputs in Mpc/h instead of Mpc')
    ap.add_argument('--workers', type=int, default=8, help='Parallel readers for FITS (>=1)')
    ap.add_argument('--min_ratio', type=float, default=5.0, help='Hard-fail if Nrand/Ndata < this (when data available)')
    ap.add_argument('--max_ratio', type=float, default=60.0, help='Hard-fail if Nrand/Ndata > this (when data available)')
    ap.add_argument('--redownload_script', type=str, default=None, help='Path to script that can re-download missing/corrupt random tiles')
    ap.add_argument('--redownload_args', type=str, default='', help='Extra args passed to re-download script')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    files = list_random_files()
    print(f'Found {len(files)} random files.')
    t = load_randoms(files, workers=max(1, args.workers))
    if 'Z' not in t.colnames:
        raise RuntimeError("Column 'Z' not found in randoms catalog")
    sel = (t['Z'] > Z_MIN) & (t['Z'] < Z_MAX)
    n_before = len(t)
    t = t[sel]
    n_after = len(t)
    kept_frac = n_after / n_before if n_before else 0.0
    print(f'Randoms rows: before z-cut={n_before:,}, after z-cut={n_after:,} (kept {kept_frac:.2%})')
    data_rows = None
    if os.path.exists(DATA_TXT):
        data_rows = sum((1 for _ in open(DATA_TXT, 'r', encoding='utf-8', errors='ignore')))
    elif os.path.exists(DATA_CSV):
        data_rows = max(0, sum((1 for _ in open(DATA_CSV, 'r', encoding='utf-8', errors='ignore'))) - 1)
    if data_rows and data_rows > 0:
        dr_ratio = n_after / data_rows
        print(f'Data→Randoms ratio (Nrand/Ndata) ≈ {dr_ratio:.2f} (data={data_rows:,}, rand={n_after:,})')
        if dr_ratio < args.min_ratio or dr_ratio > args.max_ratio:
            print(f'[Error] Nrand/Ndata={dr_ratio:.2f} is outside allowed bounds [{args.min_ratio:.2f}, {args.max_ratio:.2f}].')
            if SKIPPED_FILES and args.redownload_script:
                print('Attempting to re-download skipped/corrupt files...')
                cmd = [args.redownload_script] + [p for p in args.redownload_args.split() if p]
                try:
                    subprocess.run(cmd, check=False)
                    print('Re-download script finished. You may re-run this script to include recovered files.')
                except Exception as e:
                    print(f'[Warn] Failed to run re-download script: {e}')
            raise SystemExit(2)
    ra = np.asarray(t['RA'], dtype=float)
    dec = np.asarray(t['DEC'], dtype=float)
    zz = np.asarray(t['Z'], dtype=float)
    x, y, zc = radec_to_cartesian(ra, dec, zz)
    xyz = np.column_stack([x, y, zc])
    if args.mpch:
        xyz *= h
        suffix = 'mpch'
    else:
        suffix = 'mpc'
    csv_path = CSV_OUT.replace('mpc', suffix)
    header = 'x_{},y_{},z_{}'.format(suffix, suffix, suffix)
    np.savetxt(csv_path, xyz, delimiter=',', header=header, comments='')
    print(f'Wrote CSV: {csv_path}')
    txt_path = TXT_OUT if suffix == 'mpc' else TXT_OUT.replace('.txt', '_mpch.txt')
    np.savetxt(txt_path, xyz, fmt='%.8f')
    print(f'Wrote TXT: {txt_path}')
    if SKIPPED_FILES:
        log_path = os.path.join(OUT_DIR, f'ELG_HIP_randoms_skipped_{suffix}.txt')
        with open(log_path, 'w', encoding='utf-8') as f:
            for p in SKIPPED_FILES:
                f.write(p + '\n')
        skip_pct = 100.0 * len(SKIPPED_FILES) / max(1, len(files))
        print(f'[Info] Skipped {len(SKIPPED_FILES)} file(s) ({skip_pct:.1f}%). List written to: {log_path}')
        if skip_pct > 5.0:
            print('[Warning] More than 5% of random files were skipped. Consider re-downloading the missing/corrupted files.')
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
