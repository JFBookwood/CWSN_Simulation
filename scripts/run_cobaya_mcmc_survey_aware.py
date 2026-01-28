"""

Run full MCMC sampling with survey-aware void likelihood.

"""

import os
import sys
import subprocess
from pathlib import Path

COBAYA_PACK = Path(r"c:\Users\Jesse\Desktop\Experimente\CWSN\data\desi\edr\processed\voidfinder\vgcf\cobaya_pack")
DATA_FILE = COBAYA_PACK / "voids_vgcf_data_survey_aware.npz"

if not DATA_FILE.exists():
    print(f"ERROR: Survey-aware data file not found: {DATA_FILE}")
    sys.exit(1)

print("=" * 70)
print("RUNNING FULL MCMC SAMPLING WITH SURVEY-AWARE VOID LIKELIHOOD")
print("=" * 70)
print(f"Data file: {DATA_FILE}")
print()

yaml_mcmc = f'''# Full MCMC sampling with survey-aware void likelihood

packages_path: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack"

likelihood:
  survey_aware_voids:
    python_path: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack"
    class: "survey_aware_void_likelihood.SurveyAwareVoidLikelihood"
    data_file: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/voids_vgcf_data_survey_aware.npz"

params:
  A:
    prior: {{min: 0.2, max: 2.0}}
    ref: 0.77
    proposal: 0.05
  beta:
    prior: {{min: 0.0, max: 1.0}}
    ref: 0.01
    proposal: 0.01
  C:
    prior: {{min: -0.05, max: 0.05}}
    ref: 0.05
    proposal: 0.01

sampler:
  mcmc:
    covmat: "c:/Users/Jesse/Desktop/Experimente/CWSN/initial_cov.covmat"
    proposal_scale: 0.2
    learn_proposal: true
    burn_in: 500
    max_samples: 20000
    Rminus1_stop: 0.02

output: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/chains_survey_aware_mcmc"
'''

mcmc_yaml_path = COBAYA_PACK / "mcmc_survey_aware.yaml"

with open(mcmc_yaml_path, 'w') as f:
    f.write(yaml_mcmc)

print(f"Created MCMC YAML: {mcmc_yaml_path}")
print()
print("Starting full MCMC sampling...")
print("-" * 70)

os.chdir(str(COBAYA_PACK))

try:
    result = subprocess.run(
        [sys.executable, "-m", "cobaya.run", str(mcmc_yaml_path), "--allow-changes", "-f"],
        capture_output=False,
        text=True
    )
    if result.returncode == 0:
        print()
        print("=" * 70)
        print("SUCCESS: MCMC SAMPLING COMPLETED!")
        print("=" * 70)
        print("Check the output directory for chain files and analysis results.")
        print()
    else:
        print()
        print("ERROR: MCMC sampling failed")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
