""" Definitons for OOD analysis """

import os

# MODIS L2
extract_path = os.path.join(os.getenv("SST_OOD"), 'Extractions')
model_path = os.path.join(os.getenv("SST_OOD"), 'Models')
eval_path = os.path.join(os.getenv("SST_OOD"), 'MODIS_L2', 'Evaluations')
