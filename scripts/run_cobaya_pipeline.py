# kapitel: start
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path
ROOT = Path('c:/Users/Jesse/Desktop/Experimente/CWSN')
COBAYA_PACK = ROOT / 'data/desi/edr/processed/voidfinder/vgcf/cobaya_pack'
YAML_QUICK = COBAYA_PACK / 'example_cosmo_quick.yaml'
YAML_FULL = COBAYA_PACK / 'example_cosmo_full.yaml'
YAML_FULL_PAR = COBAYA_PACK / 'example_cosmo_full_parallel.yaml'
YAML_FULL_PAR_SEEDED = COBAYA_PACK / 'example_cosmo_full_parallel_seeded.yaml'
COVMAT_QUICK = COBAYA_PACK / 'chains_cosmo_quick.covmat'
MODES = {'quick': YAML_QUICK, 'full': YAML_FULL, 'full_parallel': YAML_FULL_PAR, 'full_parallel_seeded': YAML_FULL_PAR_SEEDED}

def run_cmd(cmd: list[str], env: dict | None=None) -> int:
    print(f"\n>>> Running: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env)
    proc.wait()
    return proc.returncode

def main():
    ap = argparse.ArgumentParser(description='Launch Cobaya VGCF runs with optional MPI and seeding')
    ap.add_argument('--mode', choices=list(MODES), default='full_parallel_seeded')
    ap.add_argument('--mpi', type=int, default=1, help='MPI ranks (mpiexec -n N); use 1 for no MPI')
    ap.add_argument('--omp', type=int, default=None, help='OMP threads per process (sets OMP_NUM_THREADS)')
    ap.add_argument('--resume', action='store_true', help='Pass resume behavior (YAMLs already have resume: true where relevant)')
    ap.add_argument('--dry', action='store_true', help='Print commands without executing')
    args = ap.parse_args()
    yaml_path = MODES[args.mode]
    if not yaml_path.exists():
        print(f'ERROR: YAML not found: {yaml_path}', file=sys.stderr)
        sys.exit(1)
    env = os.environ.copy()
    if args.omp is not None:
        env['OMP_NUM_THREADS'] = str(args.omp)
        print(f'Setting OMP_NUM_THREADS={args.omp}')

    def cobaya_cmd(yaml: Path) -> list[str]:
        cmd = ['cobaya-run', str(yaml)]
        if args.resume:
            cmd.append('--resume')
        return cmd
    if args.mode == 'full_parallel_seeded' and (not COVMAT_QUICK.exists()):
        print('Seeded mode requested but quick covmat missing. Running quick run first...')
        quick_cmd = cobaya_cmd(YAML_QUICK)
        if args.mpi > 1:
            quick_cmd = ['mpiexec', '-n', str(args.mpi)] + quick_cmd
        if args.dry:
            print('DRY-RUN:', ' '.join(quick_cmd))
        else:
            code = run_cmd(quick_cmd, env=env)
            if code != 0:
                print('ERROR: Quick run failed; cannot proceed to seeded full run.', file=sys.stderr)
                sys.exit(code)
        if not COVMAT_QUICK.exists():
            print("WARNING: Quick covmat still not found; proceeding anyway (MCMC will use 'auto' or proposals).")
    cmd = cobaya_cmd(yaml_path)
    if args.mpi > 1:
        cmd = ['mpiexec', '-n', str(args.mpi)] + cmd
    if args.dry:
        print('DRY-RUN:', ' '.join(cmd))
        sys.exit(0)
    code = run_cmd(cmd, env=env)
    sys.exit(code)
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
