""" Routines to analyze patches """
import numpy as np



from ulmo.mae import patch_analysis
from ulmo.mae import enki_utils

from IPython import embed



if __name__ == "__main__":

    # VIIRS
    t=10
    p=10
    tbl_file, orig_file, recon_file, mask_file = enki_utils.set_files(
        dataset='VIIRS', t=t, p=p)
    bias = enki_utils.load_bias((t,p))

    patch_analysis.anlayze_full(
        recon_file, orig_file=orig_file, bias=bias) 
    