# -*- coding: utf-8 -*-

"""

Test the survey-aware void likelihood outside of Cobaya.

"""

import sys

import numpy as np

sys.path.insert(0, r"c:\Users\Jesse\Desktop\Experimente\CWSN\data\desi\edr\processed\voidfinder\vgcf\cobaya_pack")

from survey_aware_void_likelihood import SurveyAwareVoidLikelihood

def test_likelihood():

    """Test the likelihood with various parameter combinations."""

    print("Testing SurveyAwareVoidLikelihood...")

    like = SurveyAwareVoidLikelihood.__new__(SurveyAwareVoidLikelihood)

    like.data_file = r"c:/Users/Jesse/Desktop/Experimente/CWSN/data/desi/edr/processed/voidfinder/vgcf/cobaya_pack/voids_vgcf_data_survey_aware.npz"

    try:

        like.initialize()

        print("Likelihood initialized successfully")

    except Exception as e:

        print(f"Initialization failed: {e}")

        return

    base_params = {

        'A': 1.0,

        'C': 0.0

    }

    print("\nTesting parameter combinations:")

    for i in range(10):

        test_params = {}

        for k, v in base_params.items():

            test_params[k] = v * (1 + 0.1 * np.random.randn())

        try:

            lp = like.logp(**test_params)

            print(f"Test {i+1}: logp = {lp:.2f}")

        except Exception as e:

            print(f"Test {i+1}: failed - {e}")

    print("\nTest completed. Check likelihood_errors.log for any errors.")

if __name__ == "__main__":

    test_likelihood()