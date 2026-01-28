"""
Compare OLD Cobaya results (0.060 ± 0.002 eV, ξ₀ = -0.018)
vs NEW results (ξ₀ = +6.03, filtered data).
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
COBAYA_PACK = Path(r"c:\Users\Jesse\Desktop\Experimente\CWSN\data\desi\edr\processed\voidfinder\vgcf\cobaya_pack")
print("="*70 + "\nCOMPARING OLD vs NEW COBAYA RESULTS\n" + "="*70)
chains_old = f"{COBAYA_PACK}/chains_cosmo_quick_camb_ncdm3.1.txt"
chains_new = f"{COBAYA_PACK}/chains_cosmo_quick_camb_filtered.1.txt"
if not Path(chains_old).exists():
    print(f"ERROR: Old chains not found: {chains_old}")
    exit(1)
if not Path(chains_new).exists():
    print(f"ERROR: New chains not found. Ensure Cobaya has finished running.")
    exit(1)
print("Loading OLD chains (unfiltered, 500 voids, ξ₀ = -0.018)...")
data_old = np.loadtxt(chains_old)
print(f"  Shape: {data_old.shape}")
print("Loading NEW chains (filtered, 112 voids, ξ₀ = +6.03)...")
data_new = np.loadtxt(chains_new)
print(f"  Shape: {data_new.shape}")
for col_idx in range(data_old.shape[1]):
    col_min = data_old[:, col_idx].min()
    col_max = data_old[:, col_idx].max()
    if 0.05 < col_min < 0.1 and 0.2 < col_max < 0.4:
        mnu_col = col_idx
        print(f"Found mnu column (OLD): column {col_idx} [range: {col_min:.4f} - {col_max:.4f}]")
        break
for col_idx in range(data_new.shape[1]):
    col_min = data_new[:, col_idx].min()
    col_max = data_new[:, col_idx].max()
    if 0.05 < col_min < 0.1 and 0.2 < col_max < 0.4:
        mnu_col_new = col_idx
        print(f"Found mnu column (NEW): column {col_idx} [range: {col_min:.4f} - {col_max:.4f}]")
        break
mnu_old = data_old[500:, mnu_col]
mnu_new = data_new[500:, mnu_col_new]
mnu_old_mean = np.mean(mnu_old)
mnu_old_std = np.std(mnu_old)
mnu_new_mean = np.mean(mnu_new)
mnu_new_std = np.std(mnu_new)
print("="*70)
print("NEUTRINO MASS CONSTRAINTS (Σmν in eV)")
print("="*70)
print(f"OLD (unfiltered, 500 voids, ξ₀ = -0.018): Σmν = {mnu_old_mean:.4f} ± {mnu_old_std:.4f} eV [Range: {mnu_old.min():.4f} - {mnu_old.max():.4f}]")
print(f"NEW (filtered, 112 voids, ξ₀ = +6.03): Σmν = {mnu_new_mean:.4f} ± {mnu_new_std:.4f} eV [Range: {mnu_new.min():.4f} - {mnu_new.max():.4f}]")
delta_mnu = abs(mnu_new_mean - mnu_old_mean)
rel_diff = (delta_mnu / mnu_old_mean) * 100 if mnu_old_mean > 0 else 0
print(f"CHANGE: Δ(Σmν) = {delta_mnu:.4f} eV ({rel_diff:.1f}% relative), Direction: {'LOWER' if mnu_new_mean < mnu_old_mean else 'HIGHER'}")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0,0].hist(mnu_old, bins=50, alpha=0.6, label="OLD (unfiltered)", color='red')
axes[0,0].hist(mnu_new, bins=50, alpha=0.6, label="NEW (filtered)", color='blue')
axes[0,0].axvline(mnu_old_mean, color='red', linestyle='--', linewidth=2, label=f"OLD mean: {mnu_old_mean:.4f}")
axes[0,0].axvline(mnu_new_mean, color='blue', linestyle='--', linewidth=2, label=f"NEW mean: {mnu_new_mean:.4f}")
axes[0,0].set_xlabel("Σmν (eV)")
axes[0,0].set_ylabel("Count")
axes[0,0].legend()
axes[0,0].set_title("Neutrino Mass Posterior Comparison")
axes[0,0].grid(alpha=0.3)
axes[0,1].plot(mnu_old, alpha=0.7, color='red', linewidth=0.5)
axes[0,1].axhline(mnu_old_mean, color='red', linestyle='--', linewidth=1, label='Mean')
axes[0,1].set_ylabel("Σmν (eV)")
axes[0,1].set_title("OLD Chain Trace (unfiltered)")
axes[0,1].grid(alpha=0.3)
axes[0,1].legend()
axes[1,0].plot(mnu_new, alpha=0.7, color='blue', linewidth=0.5)
axes[1,0].axhline(mnu_new_mean, color='blue', linestyle='--', linewidth=1, label='Mean')
axes[1,0].set_ylabel("Σmν (eV)")
axes[1,0].set_xlabel("Step")
axes[1,0].set_title("NEW Chain Trace (filtered)")
axes[1,0].grid(alpha=0.3)
axes[1,0].legend()
axes[1,1].axis('off')
summary_text = f"""
SUMMARY: Impact of Removing Survey Artifacts
Data Quality Change:
  • Original: 209,833 galaxies, 500 voids
  • Filtered: 44,240 galaxies, 112 voids
  • Removed: 165,593 galaxies (78.9%), 388 voids (77.6%)
Correlation Function Change:
  • ξ₀ Old: -0.018 (unphysical!)
  • ξ₀ New: +6.03 (sensible)
  • Change: 100% difference (sign flip!)
Neutrino Mass Constraint:
  OLD: Σmν = {mnu_old_mean:.4f} ± {mnu_old_std:.4f} eV
  NEW: Σmν = {mnu_new_mean:.4f} ± {mnu_new_std:.4f} eV
  Difference: {delta_mnu:.4f} eV ({rel_diff:.1f}%)
  Interpretation:
  The old measurement was likely BIASED
  by survey geometry artifacts. The new
  measurement with clean data should be
  more RELIABLE.
"""
axes[1,1].text(0.1, 0.5, summary_text, fontfamily='monospace', fontsize=9,
               verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.savefig(str(COBAYA_PACK / "comparison_mnu_old_vs_new.png"), dpi=150, bbox_inches='tight')
print(f"Saved: comparison_mnu_old_vs_new.png")
plt.show()
comparison_data = pd.DataFrame({
    'Metric': [
        'N_galaxies',
        'N_voids',
        'xi0_mean',
        'mnu_mean',
        'mnu_std',
        'mnu_min',
        'mnu_max'
    ],
    'OLD_unfiltered': [
        209833,
        500,
        -0.018,
        mnu_old_mean,
        mnu_old_std,
        mnu_old.min(),
        mnu_old.max()
    ],
    'NEW_filtered': [
        44240,
        112,
        6.0254,
        mnu_new_mean,
        mnu_new_std,
        mnu_new.min(),
        mnu_new.max()
    ]
})
comparison_data.to_csv(str(COBAYA_PACK / "cobaya_comparison_summary.csv"), index=False)
print(f"Saved: cobaya_comparison_summary.csv")
print("="*70)
print("✓ COMPARISON COMPLETE")
print("="*70)

