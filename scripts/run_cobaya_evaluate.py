"""
Run Cobaya evaluate sampler for debugging the survey-aware likelihood.
"""

import os
import sys
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
    sys.exit(1)

print("=" * 70)
print("RUNNING COBAYA EVALUATE SAMPLER (DEBUG MODE)")
print("=" * 70)
print(f"Data file:     {SURVEY_AWARE_DATA}")
print()

yaml_eval = f'''# Survey-aware void likelihood evaluation

packages_path: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack"

likelihood:
  survey_aware_voids:
    python_path: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack"
    class: "survey_aware_void_likelihood.SurveyAwareVoidLikelihood"
    data_file: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/voids_vgcf_data_survey_aware.npz"

params:
  A:
    prior: {{min: 0.2, max: 2.0}}
    ref: 1.0
  beta:
    prior: {{min: 0.0, max: 1.0}}
    ref: 0.3
  C:
    prior: {{min: -0.05, max: 0.05}}
    ref: 0.0

sampler:
  evaluate: {{}}

output: "c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/evaluate_survey_aware"
'''

eval_yaml_path = COBAYA_PACK / "evaluate_survey_aware.yaml"

with open(eval_yaml_path, 'w') as f:
    f.write(yaml_eval)

print(f"Created evaluate YAML: {eval_yaml_path}")
print()
print("Starting Cobaya evaluate sampler...")
print("-" * 70)

os.chdir(str(COBAYA_PACK))

try:
    result = subprocess.run(
        [sys.executable, "-m", "cobaya.run", str(eval_yaml_path), "--allow-changes"],
        capture_output=False,
        text=True
    )
    if result.returncode == 0:
        print()
        print("=" * 70)
        print("SUCCESS: COBAYA EVALUATE COMPLETED!")
        print("=" * 70)
        print("Check the output for likelihood values and any errors.")
        print()
    else:
        print()
        print("ERROR: Cobaya evaluate failed")
        sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
