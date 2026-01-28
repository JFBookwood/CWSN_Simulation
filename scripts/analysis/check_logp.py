import sys, numpy as np

sys.path.insert(0, r"c:\Users\Jesse\Desktop\Experimente\CWSN\data\desi\edr\processed\voidfinder\vgcf\cobaya_pack")

from survey_aware_void_likelihood import SurveyAwareVoidLikelihood

like = SurveyAwareVoidLikelihood.__new__(SurveyAwareVoidLikelihood)

like.data_file = r"c:\Users\Jesse\Desktop\Experimente\CWSN\data\desi\edr\processed\voidfinder\vgcf\cobaya_pack\voids_vgcf_data_survey_aware.npz"

like.initialize()