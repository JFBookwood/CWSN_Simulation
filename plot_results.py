"""
Plots MCMC results: posterior distributions and likelihood contours.

This script loads MCMC chains and creates corner plots for the parameter posteriors.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import corner
import getdist
from getdist import plots, MCSamples

# Pfad zu den Ketten
CHAINS_DIR = Path('data/desi/edr/processed/voidfinder/vgcf/cobaya_pack')

def load_chain(chain_file):
    """Lädt MCMC-Kette aus Datei."""
    print(f"Lade Kette: {chain_file}")
    # Cobaya-Ketten haben Header mit Parameternamen
    data = np.loadtxt(chain_file, comments='#')
    print(f"Kettenlänge: {len(data)} Samples")
    print(f"Shape: {data.shape}")
    # Spalten: weight, minuslogpost, ombh2, omch2, H0, mnu, As, ns, A, beta, C, minuslogprior, minuslogprior__0, chi2, chi2__voids_vgcf
    # Wir brauchen die Parameter-Spalten (2 bis 10)
    params = data[:, 2:11]  # 9 Parameter
    print(f"Parameter-Shape: {params.shape}")
    return params

def plot_posterior_corner(chain_data, param_names, output_file=None):
    """Erstellt Corner-Plot der Posterior-Verteilungen."""
    # Entferne burn-in (erste 10%)
    burn_in = int(0.1 * len(chain_data))
    samples = chain_data[burn_in:]

    print(f"Verwende {len(samples)} Samples nach burn-in")

    # Erstelle MCSamples für getdist
    try:
        mc_samples = MCSamples(samples=samples, names=param_names, labels=param_names)

        # Corner-Plot
        g = plots.get_subplot_plotter()
        g.triangle_plot(mc_samples, filled=True, title_limit=1)

        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            g.export(output_file)
            print(f"Gespeichert: {output_file}")
        else:
            plt.show()

    except Exception as e:
        print(f"Fehler bei getdist: {e}")
        # Fallback: corner.py
        # Filter Parameter mit Variation
        stds = np.std(samples, axis=0)
        valid_idx = stds > 1e-6
        samples_valid = samples[:, valid_idx]
        labels_valid = [param_names[i] for i in range(len(param_names)) if valid_idx[i]]
        print(f"Plotting {len(labels_valid)} varying parameters")
        fig = corner.corner(samples_valid, labels=labels_valid, show_titles=True, title_kwargs={"fontsize": 12})
        if output_file:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Gespeichert (corner): {output_file}")
        else:
            plt.show()
        plt.close(fig)

def plot_likelihood_contours(chain_data, param_names, param1='m_nu', param2='H0', output_file=None):
    """Plottet Likelihood-Konturen für zwei Parameter."""
    burn_in = int(0.1 * len(chain_data))
    samples = chain_data[burn_in:]

    # Finde Indizes der Parameter
    try:
        idx1 = param_names.index(param1)
        idx2 = param_names.index(param2)
    except ValueError:
        print(f"Parameter {param1} oder {param2} nicht gefunden in {param_names}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # 2D Histogramm
    h = ax.hist2d(samples[:, idx1], samples[:, idx2], bins=50, cmap='Blues', density=True)
    plt.colorbar(h[3], ax=ax, label='Density')

    ax.set_xlabel(param1)
    ax.set_ylabel(param2)
    ax.set_title(f'Likelihood Konturen: {param1} vs {param2}')
    ax.grid(True, alpha=0.3)

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Gespeichert: {output_file}")
    else:
        plt.show()

    plt.close(fig)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Plot MCMC Posterior und Likelihood')
    parser.add_argument('--chain', type=Path, default=CHAINS_DIR / 'chains_final_run.1.txt',
                        help='Pfad zur MCMC-Kette')
    parser.add_argument('--params', nargs='+', default=['ombh2', 'omch2', 'H0', 'mnu', 'As', 'ns', 'A', 'beta', 'C'],
                        help='Parameter-Namen')
    parser.add_argument('--corner', type=Path, default=Path('results/posterior_corner.png'),
                        help='Ausgabe für Corner-Plot')
    parser.add_argument('--likelihood', type=Path, default=Path('results/likelihood_contours.png'),
                        help='Ausgabe für Likelihood-Konturen')
    parser.add_argument('--param1', type=str, default='m_nu', help='Erster Parameter für Konturen')
    parser.add_argument('--param2', type=str, default='H0', help='Zweiter Parameter für Konturen')

    args = parser.parse_args()

    # Lade Kette
    chain_data = load_chain(args.chain)

    # Corner-Plot
    plot_posterior_corner(chain_data, args.params, args.corner)

    # Likelihood-Konturen
    plot_likelihood_contours(chain_data, args.params, args.param1, args.param2, args.likelihood)

    print("Plotting abgeschlossen!")

if __name__ == '__main__':
    main()
