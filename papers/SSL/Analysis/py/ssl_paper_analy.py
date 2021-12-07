""" Run SSL Analysis specific to the paper """

from pkg_resources import resource_filename
import os
import numpy as np

import h5py
import umap

from ulmo import io as ulmo_io
from ulmo.ssl.train_util import option_preprocess
from ulmo.utils import catalog as cat_utils

from IPython import embed

if os.getenv('SST_OOD'):
    local_modis_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2/Tables/MODIS_L2_std.parquet')
    local_modis_CF_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2/Tables/MODIS_SSL_cloud_free.parquet')


def load_modis_tbl(tbl_file=None, local=False, cuts=None, CF=False,
                   region=None, percentiles=None):

    # Which file?
    if tbl_file is None:
        if CF:
            tbl_file = 's3://modis-l2/Tables/MODIS_SSL_cloud_free.parquet'
        else:
            tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'
    if local:
        if CF:
            tbl_file = local_modis_CF_file
        else:
            tbl_file = local_modis_file

    # Load
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # DT
    if 'DT' not in modis_tbl.keys():
        modis_tbl['DT'] = modis_tbl.T90 - modis_tbl.T10
    modis_tbl['logDT'] = np.log10(modis_tbl.DT)
    modis_tbl['lowDT'] = modis_tbl.mean_temperature - modis_tbl.T10
    modis_tbl['absDT'] = np.abs(modis_tbl.T90) - np.abs(modis_tbl.T10)

    # Slopes
    modis_tbl['min_slope'] = np.minimum(
        modis_tbl.zonal_slope, modis_tbl.merid_slope)

    # Cut
    goodLL = np.isfinite(modis_tbl.LL)
    if cuts is None:
        good = goodLL
    elif cuts == 'inliers':
        inliers = (modis_tbl.LL > 200.) & (modis_tbl.LL < 400)
        good = goodLL & inliers
    modis_tbl = modis_tbl[good].copy()

    # Region?
    if region is None:
        pass
    elif region == 'brazil':
        # Brazil
        in_brazil = ((np.abs(modis_tbl.lon.values + 57.5) < 10.)  & 
            (np.abs(modis_tbl.lat.values + 43.0) < 10))
        in_DT = np.abs(modis_tbl.DT - 2.05) < 0.05
        modis_tbl = modis_tbl[in_brazil & in_DT].copy()
    elif region == 'GS':
        # Gulf Stream
        in_GS = ((np.abs(modis_tbl.lon.values + 69.) < 3.)  & 
            (np.abs(modis_tbl.lat.values - 39.0) < 1))
        modis_tbl = modis_tbl[in_GS].copy()
    elif region == 'Med':
        # Mediterranean
        in_Med = ((modis_tbl.lon > -5.) & (modis_tbl.lon < 30.) &
            (np.abs(modis_tbl.lat.values - 36.0) < 5))
        modis_tbl = modis_tbl[in_Med].copy()
    else: 
        raise IOError(f"Bad region! {region}")

    # Percentiles
    if percentiles is not None:
        LL_p = np.percentile(modis_tbl.LL, percentiles)
        cut_p = (modis_tbl.LL < LL_p[0]) | (modis_tbl.LL > LL_p[1])
        modis_tbl = modis_tbl[cut_p].copy()

    return modis_tbl


def umap_subset(opt_path:str, outfile:str, DT_cut=None, local=True, CF=True, debug=False):
    opt = option_preprocess(ulmo_io.Params(opt_path))

    # Load main table
    modis_tbl = load_modis_tbl(local=local, CF=CF)
    modis_tbl['U0'] = 0.
    modis_tbl['U1'] = 0.

    # Cut down
    if DT_cut is not None:
        keep = np.abs(modis_tbl.DT - DT_cut[0]) < DT_cut[1]
    else:
        raise IOError("Need at least one cut!")

    modis_tbl = modis_tbl[keep].copy()

    # 
    if CF:
        valid = modis_tbl.ulmo_pp_type == 0
        train = modis_tbl.ulmo_pp_type == 1
        cut_prefix = 'ulmo_'
    else:
        raise IOError("Need to deal with this")


    # Prep latent_files
    latents_path = os.path.join(opt.s3_outdir, opt.latents_folder)
    latent_files = ulmo_io.list_of_bucket_files(latents_path)
    latent_files = ['s3://modis-l2/'+item for item in latent_files]

    # Load up all the latent vectors

    # Loop on em all
    if debug:
        latent_files = latent_files[0:2]

    all_latents = []
    sv_idx = []
    for latents_file in latent_files:
        basefile = os.path.basename(latents_file)
        year = int(basefile[12:16])

        # Download?
        if not os.path.isfile(basefile):
            print(f"Downloading {latents_file} (this is *much* faster than s3 access)...")
            ulmo_io.download_file_from_s3(basefile, latents_file)

        #  Load and apply
        print(f"Ingesting {basefile}")
        hf = h5py.File(basefile, 'r')

        # ##############33
        # Valid
        all_latents_valid = hf['valid'][:]
        yidx = modis_tbl.pp_file == f's3://modis-l2/PreProc/MODIS_R2019_{year}_95clear_128x128_preproc_std.h5'
        valid_idx = valid & yidx
        pp_idx = modis_tbl[valid_idx].pp_idx.values

        # Grab and save
        gd_latents = all_latents_valid[pp_idx, :]
        all_latents.append(gd_latents)
        sv_idx += np.where(valid_idx)[0].tolist()

        # Train
        train_idx = train & yidx
        if 'train' in hf.keys() and (np.sum(train_idx) > 0):
            pp_idx = modis_tbl[train_idx].pp_idx.values
            all_latents_train = hf['train'][:]
            gd_latents = all_latents_train[pp_idx, :]
            all_latents.append(gd_latents)
            sv_idx += np.where(train_idx)[0].tolist()

        hf.close()
        # Clean up
        if not debug:
            print(f"Done with {basefile}.  Cleaning up")
            os.remove(basefile)

    # Concatenate
    all_latents = np.concatenate(all_latents, axis=0)

    # UMAP me
    print(f"Running UMAP on all the files")
    reducer_umap = umap.UMAP()
    latents_mapping = reducer_umap.fit(all_latents)
    print("Done..")

    # Evaluate
    embedding = latents_mapping.transform(all_latents)

    # Save
    srt = np.argsort(sv_idx)
    gd_idx = np.zeros(len(modis_tbl), dtype=bool)
    gd_idx[sv_idx] = True
    modis_tbl.loc[gd_idx, 'U0'] = embedding[srt,0]
    modis_tbl.loc[gd_idx, 'U1'] = embedding[srt,1]

    # Remove DT
    modis_tbl.drop(columns=['DT', 'logDT', 'lowDT', 'absDT', 'min_slope'], 
                   inplace=True)
    
    # Vet
    assert cat_utils.vet_main_table(modis_tbl, cut_prefix=cut_prefix)
    # Write new table
    ulmo_io.write_main_table(modis_tbl, outfile, to_s3=False)


# 
opt_path = os.path.join(resource_filename('ulmo', 'runs'),
                        'SSL', 'MODIS', 'v2', 
                        'experiments', 'modis_model_v2', 'opts_cloud_free.json')
outfile = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2/Tables/MODIS_SSL_cloud_free_DT2.parquet')
umap_subset(opt_path, outfile, DT_cut=(2., 0.2), debug=False) # 180,000 images