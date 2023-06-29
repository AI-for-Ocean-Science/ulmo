""" Routines to analyze patches """
import numpy as np

import h5py

from ulmo.mae import patch_analysis
from ulmo.mae import enki_utils

from ulmo import io as ulmo_io

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
    '''

    # Generate aligned orig file
    t=10
    p=10
    tbl_file, _, recon_file, mask_file = enki_utils.set_files(
        dataset='LLC2_noise', t=t, p=p)
    tbl_file2, orig_file, _, _ = enki_utils.set_files(
        dataset='LLC2_nonoise', t=t, p=p)

    # Load tables
    noise_tbl = ulmo_io.load_table(tbl_file)
    nonoise_tbl = ulmo_io.load_table(tbl_file2)

    # Load up orig_file
    f_orig = h5py.File(orig_file, 'r')
    orig_imgs = f_orig['valid'][:]

    # Ugly for loop..
    new_orig = np.zeros_like(orig_imgs)

    embed(header='89 of anly_patches.py')
    for ss in len(noise_tbl):
        if ss % 1000 == 0:
            print(f'ss: {ss}')
        if noise_tbl.iloc[ss].pp_idx < 0:
            continue
        #
        noise_idx = noise_tbl.iloc[ss].pp_idx
        nonoise_idx = nonoise_tbl.iloc[ss].pp_idx
        #
        new_orig[noise_idx] = orig_imgs[nonoise_idx]

    # Write 
    # ###################
    outfile = 'Enki_LLC_valid_noise_nonoise_preproc.h5'
    print("Writing: {}".format(outfile))
    with h5py.File(outfile, 'w') as f:
        # Validation
        f.create_dataset('valid', data=new_orig.astype(np.float32))
    print("Wrote: {}".format(outfile))
    
    # Upload
    ulmo_io.upload_file_to_s3(
        outfile, 's3://llc/mae/PreProc/'+outfile)