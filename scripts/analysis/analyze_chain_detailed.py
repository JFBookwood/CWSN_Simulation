import numpy as np
import pandas as pd
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plotting style
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

chain_file = 'chains_filtered_final.1.txt'
data = []
header = None

with open(chain_file, 'r') as f:
    for line in f:
        if line.startswith('#'):
            header = line[1:].strip().split()
            continue
        if header:
            parts = line.split()
            if len(parts) >= len(header):
                try:
                    data.append([float(p) for p in parts[:len(header)]])
                except:
                    continue

df = pd.DataFrame(data, columns=header)
burn_in = 1000
mnu_samples = df['mnu'].values[burn_in:]

print('=== CHAIN ANALYSIS: chains_filtered_final.1.txt ===')
print(f'Total samples: {len(df)}')
print(f'Burn-in samples: {burn_in}')
print(f'Post burn-in samples: {len(mnu_samples)}')
print()
print('=== mnu PARAMETER STATISTICS ===')
print(f'Mean: {np.mean(mnu_samples):.6f} eV')
print(f'Median: {np.median(mnu_samples):.6f} eV')
print(f'Std Dev: {np.std(mnu_samples):.6f} eV')
print(f'Min: {np.min(mnu_samples):.6f} eV')
print(f'Max: {np.max(mnu_samples):.6f} eV')
print()
print('=== PERCENTILES ===')
for p in [2.5, 5, 16, 50, 84, 95, 97.5]:
    val = np.percentile(mnu_samples, p)
    print(f'{p:5.1f}%: {val:.6f} eV')

print()
print('=== HPD INTERVALS ===')
sorted_samples = np.sort(mnu_samples)
n = len(sorted_samples)

hpd68_size = int(0.68 * n)
hpd68_ranges = sorted_samples[hpd68_size:] - sorted_samples[:n-hpd68_size]
hpd68_idx = np.argmin(hpd68_ranges)
hpd68_low = sorted_samples[hpd68_idx]
hpd68_high = sorted_samples[hpd68_idx + hpd68_size]

hpd95_size = int(0.95 * n)
hpd95_ranges = sorted_samples[hpd95_size:] - sorted_samples[:n-hpd95_size]
hpd95_idx = np.argmin(hpd95_ranges)
hpd95_low = sorted_samples[hpd95_idx]
hpd95_high = sorted_samples[hpd95_idx + hpd95_size]

print(f'HPD68: [{hpd68_low:.6f}, {hpd68_high:.6f}]')
print(f'HPD95: [{hpd95_low:.6f}, {hpd95_high:.6f}]')

print()
print('=== PROBABILITIES ===')
print(f'P(mnu > 0.06): {np.sum(mnu_samples > 0.06) / len(mnu_samples):.4f}')
print(f'P(mnu > 0.10): {np.sum(mnu_samples > 0.10) / len(mnu_samples):.4f}')
print(f'P(mnu > 0.15): {np.sum(mnu_samples > 0.15) / len(mnu_samples):.4f}')
print(f'P(mnu < 0.05): {np.sum(mnu_samples < 0.05) / len(mnu_samples):.4f}')

print()
print('=== OTHER PARAMETERS (first 100 samples post burn-in) ===')
all_params = header
for param in ['ombh2', 'omch2', 'H0', 'As', 'ns', 'A', 'beta', 'C']:
    if param in all_params:
        idx = all_params.index(param)
        param_samples = df[param].values[burn_in:]
        print(f'{param:8}: {np.mean(param_samples):12.6f} ± {np.std(param_samples):.6f}')

print()
print('=== CORRELATION MATRIX (first 100 params) ===')
select_params = ['mnu', 'ombh2', 'omch2', 'H0', 'As', 'ns', 'A', 'beta', 'C']
select_params = [p for p in select_params if p in all_params]
corr_matrix = df[select_params].iloc[burn_in:].corr()
print(corr_matrix)

print()
print('=== LOG POSTERIOR & CHI2 ===')
minuslogpost = df['minuslogpost'].values[burn_in:]
chi2 = df['chi2'].values[burn_in:]
print(f'Min log(posterior): {np.min(minuslogpost):.3f}')
print(f'Max log(posterior): {np.max(minuslogpost):.3f}')
print(f'Mean log(posterior): {np.mean(minuslogpost):.3f}')
print(f'Chi2 mean: {np.mean(chi2):.3f}')
print(f'Chi2 min: {np.min(chi2):.3f}')

print()
print('=== SAMPLE WEIGHTS ===')
weights = df['weight'].values[burn_in:]
print(f'Min weight: {np.min(weights)}')
print(f'Max weight: {np.max(weights)}')
print(f'Mean weight: {np.mean(weights):.2f}')
print(f'Unique weights: {len(np.unique(weights))}')
