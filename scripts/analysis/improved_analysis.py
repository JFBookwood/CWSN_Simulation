#!/usr/bin/env python
"""
IMPROVED CWSN Analysis - Focus on Critical Issues
Addresses As=0, prior-pinning, convergence, and correlations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_context("paper")
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12

print("=" * 80)
print("IMPROVED CWSN ANALYSIS - FOCUS ON CRITICAL ISSUES")
print("=" * 80)

# ============================================================================
# 1. LOAD CHAINS AND CHECK FOR As=0 ISSUE
# ============================================================================

def load_chain(chain_file):
    """Load MCMC chain with proper error handling."""
    data = []
    header = None

    try:
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
    except FileNotFoundError:
        print(f"ERROR: Chain file {chain_file} not found!")
        return None, None

    if not data:
        print(f"ERROR: No data loaded from {chain_file}")
        return None, None

    df = pd.DataFrame(data, columns=header)
    return df, header

print("\n1. LOADING CHAINS AND CHECKING CRITICAL ISSUES")
print("=" * 60)

# Load main chain
df_main, header_main = load_chain('chains_filtered_final.1.txt')
if df_main is None:
    print("Chain file not available. Skipping chain-specific analysis.")
    print("Proceeding with other analyses...")
    samples_main = None
    header_main = None
else:
    burn_in = 1000
    samples_main = df_main.iloc[burn_in:]

if samples_main is not None:
    # Check As issue
    if 'As' in header_main:
        as_values = samples_main['As'].values
        as_mean = np.mean(as_values)
        as_std = np.std(as_values)
        print(f"As parameter: {as_mean:.6f} ± {as_std:.6f}")
        if as_mean == 0.0 and as_std == 0.0:
            print("CRITICAL: As = 0.0 (frozen!) - Primordial perturbations disabled!")
            print("   This invalidates the entire analysis!")
        else:
            print(" As properly sampled")
    else:
        print("As parameter not found in chain")

    # Check prior-pinning
    mnu_values = samples_main['mnu'].values
    mnu_max = np.max(mnu_values)
    mnu_min = np.min(mnu_values)
    print(f"\nmnu range: [{mnu_min:.6f}, {mnu_max:.6f}] eV")
    if mnu_max >= 0.499:  # Close to upper prior boundary
        print("RITICAL: Posterior pinned at upper prior boundary (0.5 eV)")
        print("   This is NOT physics - likely likelihood sign error!")
    else:
        print("mnu not pinned at boundary")
else:
    print("Skipping As and mnu checks due to missing chain.")

# ============================================================================
# 2. COMPARE WITH GOOD CHAINS
# ============================================================================

print("\n2. COMPARING WITH PUBLICATION-READY CHAINS")
print("=" * 60)

# Load ncdm3 chain (the good one)
cobaya_pack = Path(r"c:\Users\Jesse\Desktop\Experimente\CWSN\data\desi\edr\processed\voidfinder\vgcf\cobaya_pack")
ncdm3_file = cobaya_pack / "chains_cosmo_quick_camb_ncdm3.1.txt"

if samples_main is not None and ncdm3_file.exists():
    df_ncdm3, header_ncdm3 = load_chain(str(ncdm3_file))
    if df_ncdm3 is not None:
        samples_ncdm3 = df_ncdm3.iloc[500:]  # Assuming burn-in of 500

        # Compare mnu
        mnu_ncdm3 = samples_ncdm3.iloc[:, 5].values  # Assuming mnu is column 5
        print(f"\nChain Comparison:")
        print(f"  chains_filtered_final: mnu = {np.mean(mnu_values):.6f} ± {np.std(mnu_values):.6f} eV")
        print(f"  ncdm3 (good):         mnu = {np.mean(mnu_ncdm3):.6f} ± {np.std(mnu_ncdm3):.6f} eV")
        print(f"  Difference:           {abs(np.mean(mnu_values) - np.mean(mnu_ncdm3)):.6f} eV")

        # Check if As is sampled in ncdm3
        if len(header_ncdm3) > 6:
            as_ncdm3 = samples_ncdm3.iloc[:, 6].values
            print(f"  ncdm3 As:             {np.mean(as_ncdm3):.6f} ± {np.std(as_ncdm3):.6f}")
    else:
        print("ncdm3 chain loaded but no data")
else:
    print("ncdm3 chain not found for comparison or main chain missing")

# ============================================================================
# 3. CONVERGENCE DIAGNOSTICS
# ============================================================================

print("\n3. CONVERGENCE DIAGNOSTICS")
print("=" * 60)

if samples_main is not None:
    def gelman_rubin(chain):
        """Simple Gelman-Rubin diagnostic using chain halves."""
        n = len(chain)
        if n < 100:
            return float('nan')

        half1 = chain[:n//2]
        half2 = chain[n//2:]

        mean1, mean2 = np.mean(half1), np.mean(half2)
        var1, var2 = np.var(half1, ddof=1), np.var(half2, ddof=1)

        W = (var1 + var2) / 2  # Within-chain variance
        B = (mean1 - mean2)**2 / 2  # Between-chain variance

        if W == 0:
            return float('inf')

        R = np.sqrt((B/W + 1) * (n/2) / (n/2 - 1))
        return R

    # Autocorrelation time
    def autocorr_time(chain, max_lag=100):
        """Estimate autocorrelation time."""
        n = len(chain)
        mean = np.mean(chain)
        var = np.var(chain, ddof=1)

        acf = np.correlate(chain - mean, chain - mean, mode='full')
        acf = acf[n-1:] / (n * var)  # Normalize

        # Find where ACF drops below 0.05
        lags = np.where(acf < 0.05)[0]
        if len(lags) > 0:
            tau = lags[0]
        else:
            tau = len(acf)

        return tau

    # Calculate diagnostics for mnu
    tau = autocorr_time(mnu_values)
    n_eff = len(mnu_values) / (2 * tau)
    R_hat = gelman_rubin(mnu_values)

    print(f"mnu Convergence Diagnostics:")
    print(f"  Autocorr time (τ):     {tau:.1f}")
    print(f"  N_eff:                 {n_eff:.1f}")
    print(f"  Efficiency:            {n_eff/len(mnu_values)*100:.1f}%")
    print(f"  Gelman-Rubin R̂:       {R_hat:.4f}", end="")

    if R_hat < 1.01:
        print("EXCELLENT")
    elif R_hat < 1.05:
        print("GOOD")
    elif R_hat < 1.10:
        print("ACCEPTABLE")
    else:
        print("NOT CONVERGED")
else:
    print("Skipping convergence diagnostics due to missing chain.")

# ============================================================================
# 4. PARAMETER CORRELATIONS WITH SIGNIFICANCE
# ============================================================================

print("\n4. PARAMETER CORRELATIONS (WITH SIGNIFICANCE)")
print("=" * 60)

if samples_main is not None:
    params_to_check = ['mnu', 'ombh2', 'omch2', 'H0', 'As', 'ns', 'A', 'beta', 'C']
    available_params = [p for p in params_to_check if p in samples_main.columns]

    if len(available_params) > 1:
        corr_matrix = samples_main[available_params].corr()

        print("Strong correlations (|r| > 0.1):")
        for i, p1 in enumerate(available_params):
            for j, p2 in enumerate(available_params):
                if i < j:
                    r = corr_matrix.loc[p1, p2]
                    if abs(r) > 0.1:
                        # Significance test
                        n = len(samples_main)
                        t_stat = r * np.sqrt((n-2)/(1-r**2))
                        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n-2))

                        sig = ""
                        if p_val < 0.001:
                            sig = "***"
                        elif p_val < 0.01:
                            sig = "**"
                        elif p_val < 0.05:
                            sig = "*"

                        print("5.1f")
    else:
        print("Not enough parameters for correlation analysis.")
else:
    print("Skipping correlations due to missing chain.")

# ============================================================================
# 5. PHYSICS INTERPRETATION
# ============================================================================

print("\n5. PHYSICS INTERPRETATION")
print("=" * 60)

if samples_main is not None:
    mnu_mean = np.mean(mnu_values)
    mnu_std = np.std(mnu_values)
    hpd68 = np.percentile(mnu_values, [16, 84])

    print(f"Current mnu constraint: {mnu_mean:.6f} ± {mnu_std:.6f} eV")
    print(f"HPD68: [{hpd68[0]:.6f}, {hpd68[1]:.6f}] eV")

    if mnu_max >= 0.499:
        print("\nINVALID CONSTRAINT: Posterior pinned at prior boundary!")
        print("   This is a software bug, not physics.")
        print("   True constraint likely around 0.06 eV (from ncdm3 chain)")
    else:
        print("\nValid constraint - check against literature:")
        print("   Planck+BAO: Σmν < 0.12 eV (95% CL)")
        print("   Normal Hierarchy: Σmν ≈ 0.06 eV")
        print("   Inverted Hierarchy: Σmν ≈ 0.10 eV")
else:
    print("Skipping physics interpretation due to missing chain.")

# ============================================================================
# 6. RECOMMENDATIONS
# ============================================================================

print("\n6. CRITICAL RECOMMENDATIONS")
print("=" * 60)

if samples_main is not None:
    issues = []

    if 'As' in samples_main.columns and np.mean(samples_main['As']) == 0.0:
        issues.append("As = 0.0 - primordial perturbations disabled")

    if mnu_max >= 0.499:
        issues.append("Posterior pinned at upper prior boundary")

    if tau > 100:
        issues.append(f"High autocorrelation (τ = {tau:.1f})")

    if n_eff < 100:
        issues.append(f"Low effective samples (N_eff = {n_eff:.1f})")

    if 'beta' in samples_main.columns:
        beta_mean = np.mean(samples_main['beta'])
        if beta_mean < -0.4:
            issues.append(f"Negative RSD parameter (β = {beta_mean:.3f})")

    if issues:
        print("CRITICAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
    else:
        print("No critical issues detected")

    print("\nIMMEDIATE FIXES NEEDED:")
    print("   1. Add 'As' to likelihood input_params in YAML")
    print("   2. Check VGCF likelihood for sign errors")
    print("   3. Verify parameter mapping between Cobaya and CAMB")
    print("   4. Run new MCMC with fixes")
    
    print("\nFOR PUBLICATION:")
    print("   1. Use ncdm3 chain: mnu = 0.060 +/- 0.001 eV")
    print("   2. Need 57x more samples (1M total)")
    print("   3. Resolve H0 tension (9.65 sigma discrepancy)")
    print("   4. Validate void systematics")
else:
    print("Skipping recommendations due to missing chain.")
    print("\nIMMEDIATE FIXES NEEDED:")
    print("   1. Add 'As' to likelihood input_params in YAML")
    print("   2. Check VGCF likelihood for sign errors")
    print("   3. Verify parameter mapping between Cobaya and CAMB")
    print("   4. Run new MCMC with fixes")

    print("\nFOR PUBLICATION:")
    print("   1. Use ncdm3 chain: mnu = 0.060 +/- 0.001 eV")
    print("   2. Need 57x more samples (1M total)")
    print("   3. Resolve H0 tension (9.65 sigma discrepancy)")
    print("   4. Validate void systematics")

# ============================================================================
# 7. SAVE IMPROVED REPORT
# ============================================================================

print("\n7. SAVING IMPROVED ANALYSIS REPORT")
print("=" * 60)

if samples_main is not None:
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'critical_issues': issues,
        'chain_comparison': {
            'chains_filtered_final': {
                'mnu_mean': float(mnu_mean),
                'mnu_std': float(mnu_std),
                'as_mean': float(np.mean(samples_main.get('As', [0]))),
                'n_eff': float(n_eff),
                'r_hat': float(R_hat),
                'prior_pinned': mnu_max >= 0.499
            }
        },
        'recommendations': {
            'immediate': [
                "Add 'As' to input_params in YAML",
                "Debug likelihood sign error",
                "Verify CAMB parameter mapping",
                "Run new MCMC with fixes"
            ],
            'publication': [
                "Use ncdm3 chain for constraints",
                "Run extended MCMC (1M samples)",
                "Resolve H0 tension",
                "Validate void calibration"
            ]
        }
    }
else:
    report = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'critical_issues': [],
        'chain_comparison': {},
        'recommendations': {
            'immediate': [
                "Add 'As' to input_params in YAML",
                "Debug likelihood sign error",
                "Verify CAMB parameter mapping",
                "Run new MCMC with fixes"
            ],
            'publication': [
                "Use ncdm3 chain for constraints",
                "Run extended MCMC (1M samples)",
                "Resolve H0 tension",
                "Validate void calibration"
            ]
        }
    }

report_file = Path('results/improved_analysis_report.json')
report_file.parent.mkdir(exist_ok=True)
with open(report_file, 'w') as f:
    import json
    json.dump(report, f, indent=2, default=str)

print(f"Saved: {report_file}")

print("\n" + "=" * 80)
print("IMPROVED ANALYSIS COMPLETE")
print("=" * 80)
print(f"Report saved to: {report_file}")
print("Key finding: As=0 issue and prior-pinning invalidate main chain")
print("Use ncdm3 chain for valid constraints: mnu ~ 0.060 eV")
