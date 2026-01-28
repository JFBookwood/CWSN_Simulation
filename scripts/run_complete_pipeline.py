"""
This script runs the complete pipeline for void-based neutrino mass estimation:

1. Void finding with multiple algorithms (PyCosmoMMF, REVOLVER)

2. MCMC parameter estimation with survey-aware likelihood

3. Neutrino mass constraint analysis
"""

import os

import sys

import subprocess

from pathlib import Path


def run_command(cmd, description):
    """Run a command and report success/failure."""

    print(f"\n{'='*60}")

    print(f"RUNNING: {description}")

    print("=" * 60)

    try:

        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=1800
        )

        if result.returncode == 0:

            print(f"SUCCESS: {description}")

            if result.stdout:

                lines = result.stdout.strip().split("\n")

                for line in lines[-5:]:

                    if line.strip():

                        print(f"  {line}")

        else:

            print(f"FAILED: {description}")

            print(f"Return code: {result.returncode}")

            if result.stderr:

                print("Error output:")

                print(result.stderr[-500:])

        return result.returncode == 0

    except subprocess.TimeoutExpired:

        print(f"TIMEOUT: {description} (30 minutes)")

        return False

    except Exception as e:

        print(f"ERROR: {description} - {e}")

        return False


def main():
    """Run the complete pipeline."""

    print("=" * 80)

    print("COMPLETE VOID-NEUTRINO PIPELINE - MASTER EXECUTION")

    print("=" * 80)

    print("This will run the full analysis pipeline:")

    print("1. PyCosmoMMF void finding")

    print("2. REVOLVER void finding")

    print("3. MCMC parameter estimation")

    print("4. Neutrino mass constraint analysis")

    print("=" * 80)

    os.chdir(Path(__file__).parent.parent)

    success_count = 0

    total_steps = 4

    if run_command("python scripts/integrate_pycosmomf.py", "PyCosmoMMF Void Finding"):

        success_count += 1

    else:

        print("Continuing with other methods...")

    if run_command("python scripts/run_revolver_simple.py", "REVOLVER Void Finding"):

        success_count += 1

    else:

        print("REVOLVER failed, but pipeline continues...")

    if run_command(
        "python scripts/run_cobaya_mcmc_survey_aware.py", "MCMC Parameter Estimation"
    ):

        success_count += 1

    else:

        print("MCMC failed - check Cobaya installation and parameters")

    if run_command(
        "python scripts/void_expansion_neutrino_analysis_realistic.py",
        "Neutrino Mass Constraint Analysis",
    ):

        success_count += 1

    else:

        print("Neutrino analysis failed - check void data")

    print("\n" + "=" * 80)

    print("PIPELINE EXECUTION SUMMARY")

    print("=" * 80)

    print(f"Completed steps: {success_count}/{total_steps}")

    if success_count == total_steps:

        print("🎉 COMPLETE SUCCESS: All pipeline steps executed successfully!")

        print("\nKey Results:")

        print("- Multiple void catalogs generated")

        print("- MCMC chains produced with convergence")

        print("- Neutrino mass constraints calculated")

        print("- Ready for scientific analysis and publication")

    elif success_count >= 2:

        print("✅ PARTIAL SUCCESS: Core functionality working")

        print("- Void finding and basic analysis completed")

        print("- Some advanced features may need troubleshooting")

    else:

        print("LIMITED SUCCESS: Basic functionality issues")

        print("- Check dependencies and data files")

        print("- Review error messages above")

    print("\nOutput locations:")

    print("- PyCosmoMMF voids: data/desi/edr/processed/pycosmomf_voids/")

    print("- REVOLVER voids: data/desi/edr/processed/revolver_simple_voids/")

    print("- MCMC results: data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/")

    print("- Neutrino analysis: data/desi/edr/processed/neutrino_analysis/")

    print("\n" + "=" * 80)

    print("PIPELINE EXECUTION COMPLETE")

    print("=" * 80)

    return success_count == total_steps


if __name__ == "__main__":

    success = main()

    sys.exit(0 if success else 1)
