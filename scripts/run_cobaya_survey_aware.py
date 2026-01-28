"""
Run Cobaya with SURVEY-AWARE likelihood data (voids_vgcf_data_survey_aware.npz)
This uses cleaned data with survey-geometry aware void detection.
Key difference: Proper handling of DESI survey boundaries and geometry.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

COBAYA_PACK = Path(r"c:\Users\Jesse\Desktop\Experimente\CWSN\data\desi\edr\processed\voidfinder\vgcf\cobaya_pack")
TEMPLATE_YAML = COBAYA_PACK / "example_cosmo_quick_camb.yaml"
SURVEY_AWARE_DATA = COBAYA_PACK / "voids_vgcf_data_survey_aware.npz"
if not TEMPLATE_YAML.exists():
    print(f"ERROR: Template YAML not found: {TEMPLATE_YAML}")
    sys.exit(1)
if not SURVEY_AWARE_DATA.exists():
    print(f"ERROR: Survey-aware data NPZ not found: {SURVEY_AWARE_DATA}")
print("Run: python scripts/create_filtered_likelihood_npz.py")
sys.exit(1)
print("=" * 70)
print("RUNNING COBAYA WITH SURVEY-AWARE DATA")
print("=" * 70)
print(f"Data file:     {SURVEY_AWARE_DATA}")
print(f"Output prefix: chains_cosmo_quick_camb_survey_aware")
print()
with open(TEMPLATE_YAML, 'r') as f:
    yaml_content = f.read()

yaml_survey_aware = yaml_content.replace(
    'data_file: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/voids_vgcf_data.npz"',
    'data_file: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/voids_vgcf_data_survey_aware.npz"'
).replace(
    'output: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/chains_cosmo_quick_camb"',
    'output: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/chains_cosmo_quick_camb_survey_aware"'
).replace(
    'likelihood:\n  voids_vgcf:\n    python_path: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack"\n    class: "likelihood_voids_vgcf_cosmo.VoidsVGCF_APCosmo"\n    data_file: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/voids_vgcf_data.npz"\n    use_ap_from_cosmo: true\n    input_params: [A, beta, C]',
    'likelihood:\n  survey_aware_voids:\n    python_path: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack"\n    class: "survey_aware_void_likelihood.SurveyAwareVoidLikelihood"\n    data_file: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/voids_vgcf_data_survey_aware.npz"'
).replace('    prior: {min: 0.01, max: 5.0}', '    prior: {min: 0.1, max: 10.0}').replace('    prior: {min: -1.0, max: 2.0}', '    prior: {min: -2.0, max: 2.0}').replace('    prior: {min: -1.0, max: 1.0}', '    prior: {min: -5.0, max: 5.0}').replace('    ref: 0.022', '').replace('    ref: 0.12', '').replace('    ref: 67.0', '').replace('    ref: 0.06', '').replace('    ref: 2.1e-9', '').replace('    ref: 0.965', '').replace('    ref: 1.0', '').replace('    ref: 0.3', '').replace('    ref: 0.0', '').replace(
    'covmat: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/chains_cosmo_quick_camb_floor.covmat"',
    'covmat: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/chains_cosmo_quick_camb_ncdm3.covmat"'
).replace('    nchains: 4\n', '')

survey_aware_yaml_path = COBAYA_PACK / "example_cosmo_quick_camb_survey_aware.yaml"
with open(survey_aware_yaml_path, 'w') as f:
    f.write(yaml_survey_aware)
print(f"Created modified YAML: {survey_aware_yaml_path}")
print()
print("Starting Cobaya MCMC...")
print("-" * 70)
os.chdir(str(COBAYA_PACK))
try:
    result = subprocess.run(
        [sys.executable, "-m", "cobaya.run", str(survey_aware_yaml_path), "--allow-changes"],
        capture_output=False,
        text=True
    )
    if result.returncode == 0:
        print()
        print("=" * 70)
        print("SUCCESS: COBAYA EXECUTION COMPLETED!")
        print("=" * 70)
        print(f"Output files: chains_cosmo_quick_camb_survey_aware.*")
        print()
        print("Next steps:")
        print("1. Run: python scripts/compare_cobaya_results.py")
        print("   (Compare OLD vs NEW Sigma_m_nu constraints)")
        print("2. Inspect: chains_cosmo_quick_camb_survey_aware.txt")
        print("3. Check corner plots for new Sigma_m_nu posterior")
        print()
    else:
        print()
        print("ERROR: Cobaya execution failed")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
