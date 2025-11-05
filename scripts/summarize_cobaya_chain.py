# kapitel: start
from __future__ import annotations
import argparse
import sys
from pathlib import Path
try:
    from cobaya import load_samples
except Exception as e:
    print('ERROR: cobaya not installed or failed to import. pip install cobaya', file=sys.stderr)
    raise
DEFAULT_PARAMS = ['omega_b', 'omega_cdm', 'h', 'm_ncdm', 'A_s', 'n_s', 'A', 'beta', 'C']

def summarize(prefix: str, params: list[str], corner: bool):
    sc0 = load_samples(prefix, skip=0.33)
    sc = sc0[0] if isinstance(sc0, list) and sc0 else sc0
    print(f'Loaded samples for prefix: {prefix}')
    try:
        print(f'n_samples (post-skip) = {len(sc)}')
    except Exception:
        pass
    print('\nSummary (mean ± 1σ):')
    try:
        import numpy as np
        w = np.asarray(sc['weight'], dtype=float)
    except Exception:
        w = None
    for p in params:
        try:
            import numpy as np
            x = np.asarray(sc[p], dtype=float)
            if w is not None and np.isfinite(w).any():
                W = np.sum(w)
                mu = np.sum(w * x) / W if W > 0 else float('nan')
                var = np.sum(w * (x - mu) ** 2) / W if W > 0 else float('nan')
                sd = float(np.sqrt(var)) if var == var else float('nan')
            else:
                mu = float(np.mean(x))
                sd = float(np.std(x))
            print(f'  {p:8s} = {mu:.5g} ± {sd:.2g}')
        except KeyError:
            print(f'  {p:8s} = (not found)')
    try:
        best = sc.bestfit()
        print('\nBest-fit logpost:', best.get('logpost', 'n/a'))
    except Exception:
        pass
    if corner:
        try:
            import matplotlib.pyplot as plt
            try:
                from getdist import plots as gplots
                gd = load_samples(prefix, skip=0.33, to_getdist=True)
                g = gplots.get_subplot_plotter()
                g.triangle_plot(gd, params, filled=True)
                plt.tight_layout()
            except Exception:
                import itertools
                fig, axes = plt.subplots(len(params), len(params), figsize=(2 * len(params), 2 * len(params)))
                for i, pi in enumerate(params):
                    for j, pj in enumerate(params):
                        ax = axes[i, j]
                        if i == j:
                            try:
                                ax.hist(sc[pi], bins=40, color='C0', alpha=0.7)
                            except Exception:
                                pass
                        elif i > j:
                            try:
                                ax.scatter(sc[pj], sc[pi], s=2, alpha=0.3)
                            except Exception:
                                pass
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if i == len(params) - 1:
                            ax.set_xlabel(pj)
                        if j == 0:
                            ax.set_ylabel(pi)
                plt.tight_layout()
            out = Path(prefix).with_suffix('')
            out_png = str(out) + '_corner.png'
            plt.savefig(out_png, dpi=120)
            print(f'Saved corner plot: {out_png}')
        except Exception as e:
            print('Corner plot skipped (matplotlib/getdist missing or failed).', e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prefix', required=True, help='Chain prefix (without .progress)')
    ap.add_argument('--params', nargs='*', default=DEFAULT_PARAMS)
    ap.add_argument('--corner', action='store_true')
    args = ap.parse_args()
    summarize(args.prefix, args.params, args.corner)
# kapitel: main
if __name__ == '__main__':
    main()

# kapitel: ende
