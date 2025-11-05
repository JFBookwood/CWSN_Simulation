# kapitel: start
import os
import numpy as np
from astropy.table import Table, vstack
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
IN_DIR = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN\\\\data\\\\desi\\\\edr\\\\lss\\\\v2.0\\\\LSScats\\\\clustering'
OUT_DIR = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN\\\\data\\\\desi\\\\edr\\\\processed'
OUT_FILE = os.path.join(OUT_DIR, 'ELG_HIP_positions_xyz_mpc.csv')
INPUT_FILES = [os.path.join(IN_DIR, 'ELG_HIP_N_clustering.dat.fits'), os.path.join(IN_DIR, 'ELG_HIP_S_clustering.dat.fits')]
Z_MIN, Z_MAX = (0.6, 1.6)

def load_tables(files):
    tables = []
    for p in files:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Missing file: {p}')
        t = Table.read(p)
        tables.append(t)
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
    os.makedirs(OUT_DIR, exist_ok=True)
    t = load_tables(INPUT_FILES)
    if 'Z' not in t.colnames:
        raise RuntimeError("Column 'Z' not found in clustering catalog")
    sel = (t['Z'] > Z_MIN) & (t['Z'] < Z_MAX)
    t = t[sel]
    ra = np.asarray(t['RA'], dtype=float)
    dec = np.asarray(t['DEC'], dtype=float)
    zz = np.asarray(t['Z'], dtype=float)
    x, y, zc = radec_to_cartesian(ra, dec, zz)
    out = np.column_stack([x, y, zc, zz, ra, dec])
    header = 'x_mpc,y_mpc,z_mpc,z,ra_deg,dec_deg'
    np.savetxt(OUT_FILE, out, delimiter=',', header=header, comments='')
    print(f'Wrote: {OUT_FILE}')
    print(f'Rows: {out.shape[0]}')
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
