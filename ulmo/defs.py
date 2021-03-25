""" Definitons for OOD analysis """

import os

# MODIS L2
if os.getenv('SST_OOD') is not None:
    modis_extract_path = os.path.join(os.getenv("SST_OOD"), 'MODIS_L2', 'Extractions')
    modis_model_path = os.path.join(os.getenv("SST_OOD"), 'MODIS_L2', 'Models')
    modis_eval_path = os.path.join(os.getenv("SST_OOD"), 'MODIS_L2', 'Evaluations')

# Main Table definitions

mtbl_dmodel = {'pp_type': dict(dtype=int, allowed=(-1, 0,1), 
                             valid=0, train=1, init=-1,
                             help='-1: illdefined, 0: valid, 1: test')
    }