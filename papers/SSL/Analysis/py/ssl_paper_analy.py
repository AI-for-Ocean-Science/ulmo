""" Run SSL Analysis specific to the paper """

from pkg_resources import resource_filename
import os
import numpy as np
import pickle

import h5py
import umap
import pandas

from ulmo import io as ulmo_io
from ulmo.ssl.train_util import option_preprocess
from ulmo.utils import catalog as cat_utils

from IPython import embed

if os.getenv('SST_OOD'):
    local_modis_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2/Tables/MODIS_L2_std.parquet')
    local_modis_CF_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2/Tables/MODIS_SSL_cloud_free.parquet')
    local_modis_CF_DT2_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2/Tables/MODIS_SSL_cloud_free_DT2.parquet')
    local_modis_96_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2/Tables/MODIS_SSL_96clear.parquet')

def load_modis_tbl(table:str=None, 
                   local=False, cuts:str=None, 
                   region:str=None, percentiles:list=None):
    """Load up the MODIS table and (usually) cut it down

    Args:
        table (str, optional): Code for the table name. Defaults to None.
            std, CF, CF_DT0, ...
        local (bool, optional): Load file on local harddrive?. Defaults to False.
        cuts (str, optional): Named cuts. Defaults to None.
            inliers: Restrict to LL = [200,400]
        region (str, optional): Cut on geographic region. Defaults to None.
            Brazil, GS, Med
        percentiles (list, optional): Cut on percentiles of LL. Defaults to None.

    Raises:
        IOError: _description_

    Returns:
        pandas.Dataframe: MODIS table
    """

    # Which file?
    if table is None:
        table = '96' 
    if table == 'std':  # Original; too many clouds
        basename = 'MODIS_L2_std.parquet'
    else:
        # Base 1
        if 'CF' in table:
            base1 = 'MODIS_SSL_cloud_free'
        elif '96' in table:
            base1 = 'MODIS_SSL_96clear'
        # DT
        if 'DT' in table:
            dtstr = table.split('_')
            base2 = '_'+dtstr
        else:
            base2 = ''
        # 
        basename = base1+base2+'.parquet'

    '''
    elif table == 'CF':
        basename = 'MODIS_SSL_cloud_free.parquet'
    elif table == 'CF_DT0':
        basename = 'MODIS_SSL_cloud_free_DT0.parquet'
    elif table == 'CF_DT1':
        basename = 'MODIS_SSL_cloud_free_DT1.parquet'
    elif table == 'CF_DT15':
        basename = 'MODIS_SSL_cloud_free_DT15.parquet'
    elif table == 'CF_DT2':
        basename = 'MODIS_SSL_cloud_free_DT2.parquet'
    elif table == 'CF_DT1_DT2':
        basename = 'UT1_2003.parquet'
    '''

    if local:
        tbl_file = os.path.join(os.getenv('SST_OOD'), 'MODIS_L2', 'Tables', basename)
    else:
        tbl_file = 's3://modis-l2/Tables/'+basename

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


def umap_subset(opt_path:str, outfile:str, DT_cut=None, 
                ntrain=200000, remove=True,
                umap_savefile:str=None,
                local=True, CF=False, debug=False):
    """Run UMAP on a subset of the data
    First 2 dimensions are written to the table

    Args:
        opt_path (str): _description_
        outfile (str): _description_
        DT_cut (_type_, optional): _description_. Defaults to None.
        ntrain (int, optional): _description_. Defaults to 200000.
        remove (bool, optional): _description_. Defaults to True.
        umap_savefile (str, optional): _description_. Defaults to None.
        local (bool, optional): _description_. Defaults to True.
        CF (bool, optional): Use cloud free (99%) set? Defaults to False.
        debug (bool, optional): _description_. Defaults to False.

    Raises:
        IOError: _description_
        IOError: _description_
    """

    opt = option_preprocess(ulmo_io.Params(opt_path))

    # Load main table
    table='CF' if CF else '96'
    modis_tbl = load_modis_tbl(local=local, 
                               table=table)
    modis_tbl['US0'] = 0.
    modis_tbl['US1'] = 0.

    # Cut down
    if DT_cut is not None:
        if DT_cut[1] < 0: # Lower limit?
            keep = modis_tbl.DT > DT_cut[0]
        else:
            keep = np.abs(modis_tbl.DT - DT_cut[0]) < DT_cut[1]
    else: # Do em all!
        keep = np.ones(len(modis_tbl), dtype=bool)

    modis_tbl = modis_tbl[keep].copy()
    print(f"After the cuts, we have {len(modis_tbl)} cutouts to work on.")

    # 
    if table in ['CF', '96']:
        valid = modis_tbl.ulmo_pp_type == 0
        train = modis_tbl.ulmo_pp_type == 1
        cut_prefix = 'ulmo_'
    else:
        raise IOError("Need to deal with this")

    # Prep latent_files
    print(f"Loading latents from this folder: {opt.latents_folder}")
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
        if not debug and remove:
            print(f"Done with {basefile}.  Cleaning up")
            os.remove(basefile)

    # Concatenate
    all_latents = np.concatenate(all_latents, axis=0)
    nlatents = all_latents.shape[0]

    # UMAP me
    ntrain = min(ntrain, nlatents)
    print(f"Training UMAP on a random {ntrain} set of the files")
    random = np.random.choice(np.arange(nlatents), size=ntrain, 
                              replace=False)
    reducer_umap = umap.UMAP()
    latents_mapping = reducer_umap.fit(all_latents[random,...])
    print("Done..")

    # Save?
    if umap_savefile is not None:
        pickle.dump(latents_mapping, open(umap_savefile, "wb" ) )
        print(f"Saved UMAP to {umap_savefile}")

    # Evaluate all of the latents
    print("Embedding all of the latents..")
    embedding = latents_mapping.transform(all_latents)

    # Save
    srt = np.argsort(sv_idx)
    gd_idx = np.zeros(len(modis_tbl), dtype=bool)
    gd_idx[sv_idx] = True
    modis_tbl.loc[gd_idx, 'US0'] = embedding[srt,0]
    modis_tbl.loc[gd_idx, 'US1'] = embedding[srt,1]

    # Remove DT
    modis_tbl.drop(columns=['DT', 'logDT', 'lowDT', 'absDT', 'min_slope'], 
                   inplace=True)
    
    # Vet
    assert cat_utils.vet_main_table(modis_tbl, cut_prefix=cut_prefix)
    # Write new table
    ulmo_io.write_main_table(modis_tbl, outfile, to_s3=False)
