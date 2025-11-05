# kapitel: start
import os
import numpy as np
from astropy.table import Table, vstack
BASE_DIR = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN\\\\data\\\\desi\\\\edr\\\\lss\\\\v2.0\\\\LSScats\\\\clustering'
FILES = [os.path.join(BASE_DIR, 'ELG_HIP_N_clustering.dat.fits'), os.path.join(BASE_DIR, 'ELG_HIP_S_clustering.dat.fits')]

def load_tables(files):
    tables = []
    for p in files:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Missing file: {p}')
        t = Table.read(p)
        tables.append(t)
    return vstack(tables, join_type='exact')

def main():
    t = load_tables(FILES)
    print(t)
    print('Columns:', t.colnames)
    for col in ['Z', 'ZWARN']:
        if col not in t.colnames:
            print(f'Column {col} not found; check catalog schema.')
    if 'Z' in t.colnames:
        if 'ZWARN' in t.colnames:
            sel = (t['ZWARN'] == 0) & (t['Z'] > 0.6) & (t['Z'] < 1.6)
        else:
            sel = (t['Z'] > 0.6) & (t['Z'] < 1.6)
        t_sel = t[sel]
        print(f'Selected {len(t_sel)} of {len(t)} rows after basic cuts.')
    else:
        t_sel = t
    scalar_cols = [c for c in t.colnames if getattr(t[c], 'ndim', 1) == 1]
    preview = t[scalar_cols][:1000]
    preview_path = os.path.join(BASE_DIR, 'ELG_HIP_preview.csv')
    preview.write(preview_path, format='csv', overwrite=True)
    print(f'Wrote preview: {preview_path}')
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
