""" UMAP utitlies for SSL """
import numpy as np
import os, shutil
import pickle

import h5py
import pandas
import umap

from ulmo import io as ulmo_io
from ulmo.ssl.train_util import option_preprocess
from ulmo.utils import catalog as cat_utils
from ulmo.ssl import defs as ssl_defs

def umap_subset(modis_tbl:pandas.DataFrame, 
                opt_path:str, outfile:str, DT_cut=None, 
                ntrain=200000, remove=True,
                umap_savefile:str=None,
                local=True, CF=False, debug=False):
    """Run UMAP on a subset of the data
    First 2 dimensions are written to the table

    Args:
        modis_tbl (pandas.DataFrame): MODIS table
        opt_path (str): _description_
        outfile (str): _description_
        DT_cut (str, optional): DT40 cut to apply. Defaults to None.
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
    modis_tbl['US0'] = 0.
    modis_tbl['US1'] = 0.

    # Cut down on DT
    if DT_cut is not None:
        DT_cuts = ssl_defs.umap_DT[DT_cut]
        if DT_cuts[1] < 0: # Lower limit?
            keep = modis_tbl.DT40 > DT_cuts[0]
        else:
            keep = np.abs(modis_tbl.DT40 - DT_cuts[0]) < DT_cuts[1]
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
            # Try local if local is True
            if local:
                local_file = latents_file.replace('s3://modis-l2/SSL',
                    os.path.join(os.getenv('SST_OOD'),
                                                  'MODIS_L2', 'SSL'))
                print(f"Copying {local_file}")
                shutil.copyfile(local_file, basefile)
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
    to_s3 = True if 's3' in outfile else False
    ulmo_io.write_main_table(modis_tbl, outfile, to_s3=to_s3)
