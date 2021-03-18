""" Definitons for OOD analysis """

import os

# MODIS L2
if os.getenv('SST_OOD') is not None:
    extract_path = os.path.join(os.getenv("SST_OOD"), 'Extractions')
    model_path = os.path.join(os.getenv("SST_OOD"), 'Models')
    eval_path = os.path.join(os.getenv("SST_OOD"), 'MODIS_L2', 'Evaluations')

# Main Table definitions

mtbl_dmodel = {'pp_type': dict(dtype=int, allowed=(-1, 0,1), 
                             valid=0, test=1, init=-1,
                             help='-1: illdefined, 0: valid, 1: test')
    }