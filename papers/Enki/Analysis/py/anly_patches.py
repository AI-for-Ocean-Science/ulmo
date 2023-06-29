""" Routines to analyze patches """
import numpy as np



from ulmo.mae import patch_analysis
from ulmo.mae import enki_utils

from IPython import embed



if __name__ == "__main__":

    '''
    # VIIRS
    t=10
    p=10
    tbl_file, orig_file, recon_file, mask_file = enki_utils.set_files(
        dataset='VIIRS', t=t, p=p)
    bias = enki_utils.load_bias((t,p))

    patch_analysis.anlayze_full(
        recon_file, orig_file=orig_file, bias=bias) 
    
    '''

    '''
    # LLC2 no noise
    t=10
    p=10 #20
    tbl_file, orig_file, recon_file, mask_file = enki_utils.set_files(
        dataset='LLC2_nonoise', t=t, p=p)
    bias = enki_utils.load_bias((t,p), dataset='LLC2_nonoise')

    print(f"Working on: {recon_file} using orig={orig_file}")
    patch_analysis.anlayze_full(
        recon_file, orig_file=orig_file, bias=bias, nsub=100000, n_cores=12) 

    
    # LLC2 noise
    t=10
    p=10
    tbl_file, orig_file, recon_file, mask_file = enki_utils.set_files(
        dataset='LLC2_noise', t=t, p=p)
    bias = 0.
    print(f"WARNING: Using bias={bias} for {recon_file}")

    print(f"Working on: {recon_file} using orig={orig_file}")
    patch_analysis.anlayze_full(
        recon_file, orig_file=orig_file, bias=bias, nsub=100000, n_cores=12) 
    '''

    # LLC2 noise but with noiseless patches
    t=10
    p=10
    tbl_file, orig_file, recon_file, mask_file = enki_utils.set_files(
        dataset='LLC2_noise', t=t, p=p)
    _, orig_file, _, _ = enki_utils.set_files(
        dataset='LLC2_nonoise', t=t, p=p)
    bias = 0.
    print(f"WARNING: Using bias={bias} for {recon_file}")

    print(f"Working on: {recon_file} using orig={orig_file}")
    patch_analysis.anlayze_full(
        recon_file, orig_file=orig_file, bias=bias, 
        nsub=100000, n_cores=12,
        outfile='enki_noise_patches_noiseless_t10_p10.npz')