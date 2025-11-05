# kapitel: start
import os
import sys
import hashlib
from urllib.request import urlopen
from urllib.error import URLError, HTTPError
BASE_URL = 'https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/clustering/'
OUT_DIR = 'c:\\\\Users\\\\Jesse\\\\Desktop\\\\Experimente\\\\CWSN\\\\data\\\\desi\\\\edr\\\\lss\\\\v2.0\\\\LSScats\\\\clustering'
DATA_FILES = ['ELG_HIP_N_clustering.dat.fits', 'ELG_HIP_S_clustering.dat.fits']
RANDOM_FILES_N = [f'ELG_HIP_N_{i}_clustering.ran.fits' for i in range(0, 18)]
RANDOM_FILES_S = [f'ELG_HIP_S_{i}_clustering.ran.fits' for i in range(0, 18)]
ALL_FILES = DATA_FILES + RANDOM_FILES_N + RANDOM_FILES_S
CHUNK = 1024 * 512

def download_file(url: str, out_path: str):
    try:
        with urlopen(url) as resp:
            total = resp.length
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            downloaded = 0
            with open(out_path, 'wb') as f:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        print(f'  {os.path.basename(out_path)}: {downloaded / 1000000.0:.1f}MB / {total / 1000000.0:.1f}MB ({pct:.1f}%)', end='\r')
            print(f"  {os.path.basename(out_path)}: DONE.{' ' * 40}")
    except HTTPError as e:
        print(f'HTTP error for {url}: {e.code} {e.reason}')
        return False
    except URLError as e:
        print(f'URL error for {url}: {e.reason}')
        return False
    return True

def main():
    print('Starting DESI EDR ELG_HIP downloads...\n')
    print(f'Output dir: {OUT_DIR}\n')
    failed = []
    for fname in ALL_FILES:
        url = BASE_URL + fname
        out_path = os.path.join(OUT_DIR, fname)
        if os.path.exists(out_path):
            print(f'Skipping existing file: {fname}')
            continue
        print(f'Downloading: {fname}')
        ok = download_file(url, out_path)
        if not ok:
            failed.append(fname)
    if failed:
        print('\nSome files failed:')
        for f in failed:
            print(' -', f)
        sys.exit(1)
    print('\nAll requested files downloaded successfully.')
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
