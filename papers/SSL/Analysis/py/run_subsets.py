""" Run UMAP on subsets of the data (DT) """
import os
import numpy as np
from pkg_resources import resource_filename

import h5py

from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils

import ssl_paper_analy

from IPython import embed

opt_path_CF = os.path.join(resource_filename('ulmo', 'runs'),
                        'SSL', 'MODIS', 'v2', 
                        'experiments', 'modis_model_v2', 'opts_cloud_free.json')
opt_path_96 = os.path.join(resource_filename('ulmo', 'runs'),
                        'SSL', 'MODIS', 'v3', 
                        'opts_96clear_ssl.json')

def run_subset(subset, remove=True, CF=False):                    
    if subset == 'DT0':
        DT_cut = (0.25, 0.25)
    elif subset == 'DT10':
        DT_cut = (1.0, 0.05)
    elif subset == 'DT1':
        DT_cut = (0.75, 0.25)
    elif subset == 'DT15':
        DT_cut = (1.25, 0.25)
    elif subset == 'DT2':
        DT_cut = (2.0, 0.5)
    elif subset == 'DT4':
        DT_cut = (3.25, 0.75)
    elif subset == 'DT5':
        DT_cut = (5.0, -1)
    elif subset == 'all':
        DT_cut = None

    # Prep
    if CF:
        base1 = 'cloud_free'
        opt_path = opt_path_CF
    else:
        base1 = '96clear'
        opt_path = opt_path_96
    outfile = os.path.join(
        os.getenv('SST_OOD'), 
        f'MODIS_L2/Tables/MODIS_SSL_{base1}_{subset}.parquet')
    umap_savefile = os.path.join(
        os.getenv('SST_OOD'), 
        f'MODIS_L2/UMAP/MODIS_SSL_{base1}_{subset}_UMAP.pkl')

    # Run
    ssl_paper_analy.umap_subset(opt_path, outfile, 
                                DT_cut=DT_cut, debug=False,
                                umap_savefile=umap_savefile,
                                remove=remove, CF=CF)

def build_portal_images(subset, local=True, CF=False, debug=False):

    if CF:
        base1 = 'cloud_free'
        opt_path = opt_path_CF
    else:
        base1 = '96clear'
        opt_path = opt_path_96

    # Grab table (local)
    tbl_file = os.path.join(os.getenv('SST_OOD'),
                            'MODIS_L2', 'Tables', 
                            f'MODIS_SSL_{base1}_{subset}.parquet')
    subset_tbl = ulmo_io.load_main_table(tbl_file)
    subset_tbl.reset_index(inplace=True, drop=True)

    # Reindex!

    # Init
    subset_tbl['portal_pp_idx'] = -1

    outfile = os.path.join(os.getenv('SST_OOD'),
                            'MODIS_L2', 'Portal', 
                            f'MODIS_SSL_{base1}_{subset}_preproc.h5')

    # Loop on PreProc files
    pp_files = np.unique(subset_tbl.pp_file)
    valid_imgs = []

    valid_id0 = 0
    for pp_file in pp_files:
        ipp = subset_tbl.pp_file == pp_file
        # Local?
        if local:
            dpath = os.path.join(os.getenv('SST_OOD'), 'MODIS_L2', 'PreProc')
            ifile = os.path.join(dpath, os.path.basename(pp_file))
        else:
            embed(header='Not setup for this')
        # Open
        print(f"Working on: {ifile}")
        hf = h5py.File(ifile, 'r')
        # Faster to load em all up
        all_ulmo_valid = hf['valid'][:]

        # Type
        train_img_pp = subset_tbl.ulmo_pp_type == 1
        valid_img_pp = subset_tbl.ulmo_pp_type == 0

        # Valid (Ulmo)
        if np.any(valid_img_pp & ipp):
            # Fastest to grab em all
            iidx = np.where(valid_img_pp & ipp)[0]
            idx = subset_tbl.pp_idx.values[iidx]
            valid_imgs += [all_ulmo_valid[idx, 0, :, :]]
            # Set portal_pp_idx
            subset_tbl.loc[iidx, 'portal_pp_idx'] = valid_id0 + np.arange(idx.size)
            valid_id0 += idx.size

        # Train (Ulmo)
        #if np.any(train_img_pp & ipp):
        #    # Fastest to grab em all
        #    iidx = np.where(train_img_pp & ipp)[0]
        #    idx = img_tbl.pp_idx.values[iidx]
        #    n_new = len(idx)
        #    train_imgs[itrain:itrain+n_new, ...] = all_ulmo_valid[idx, 0, :, :]
        #    itrain += n_new

        del all_ulmo_valid

        hf.close()
        # 
        if debug and pp_file == pp_files[1]:
            break

    # Write
    valid_imgs = np.concatenate(valid_imgs, axis=0)
    valid_imgs = valid_imgs[:, None, :, :]
    if not debug:
        out_h5 = h5py.File(outfile, 'w')
        #out_h5.create_dataset('train', data=train_imgs.reshape((train_imgs.shape[0],1,img_shape[0], img_shape[1]))) 
        #out_h5.create_dataset('train_indices', data=all_ulmo_valid_idx[train])  # These are the cloud free indices
        out_h5.create_dataset('valid', data=valid_imgs.astype(np.float32))
        #out_h5.create_dataset('valid_indices', data=all_ulmo_valid_idx[valid])  # These are the cloud free indices
        out_h5.close()
        print(f"Wrote: {outfile}")

    assert cat_utils.vet_main_table(subset_tbl, cut_prefix=['ulmo_', 'portal_'])

    if not debug:
        ulmo_io.write_main_table(subset_tbl, tbl_file, to_s3=False)
        print("Push the Table to s3 to keep the portal info")

# All
#run_subset('all', remove=False)

# DT cuts
# DT0  0 - 0.5 :: 656783 cutouts
#run_subset('DT0', remove=False)

# DT1  0.5 - 1 :: 3579807 cutouts
#run_subset('DT1', remove=False)

# DT15 1 - 1.5
#run_subset('DT15', remove=False)

# DT2  1.5 - 2.5
#run_subset('DT2', remove=False)

# DT4  2.5 - 4 # > 200000
#run_subset('DT4', remove=False)

# DT5  >5  # 16000 cutouts
#run_subset('DT5', remove=False) 

# Build portal image files
build_portal_images('DT0')
build_portal_images('DT1')
#build_portal_images('DT15')
build_portal_images('DT2')
build_portal_images('DT4')
build_portal_images('DT5')


# CF
# IF WE REDO, SET CF=True