""" Methods for MODIS pre-processing """

import numpy as np
import os

import h5py
import pandas

from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


from ulmo.preproc import io as pp_io
from ulmo.preproc import utils as pp_utils
from ulmo import io as ulmo_io

from IPython import embed

def preproc_tbl(modis_tbl:pandas.DataFrame, valid_fraction:float, 
                s3_bucket:str,
                preproc_root='standard',
                debug=False, 
                extract_folder='Extract',
                preproc_folder='PreProc',
                clobber_local=True):
    nsub_fields = 10000
    n_cores = 10

    # Preprocess options
    pdict = pp_io.load_options(preproc_root)

    # Setup for parallel
    map_fn = partial(pp_utils.preproc_image, pdict=pdict,
                     use_mask=True)

    # Prep table
    modis_tbl = pp_utils.prep_table_for_preproc(modis_tbl, 
                                                preproc_root)
    
    # Folders
    if not os.path.isdir(extract_folder):
        os.mkdir(extract_folder)                                            
    if not os.path.isdir(preproc_folder):
        os.mkdir(preproc_folder)                                            

    # Unique extraction files
    uni_ex_files = np.unique(modis_tbl.ex_filename)

    for ex_file in uni_ex_files:
        print("Working on Exraction file: {}".format(ex_file))

        # Download to local
        local_file = os.path.join(extract_folder, os.path.basename(ex_file))
        ulmo_io.download_file_from_s3(local_file, ex_file)

        # Output file
        local_outfile = local_file.replace('inpaint', 
                                           'preproc_'+preproc_root).replace(
                                               extract_folder, preproc_folder)
        s3_file = os.path.join(s3_bucket, preproc_folder, os.path.basename(local_outfile))

        # Find the matches
        gd_exfile = modis_tbl.ex_filename == ex_file
        ex_idx = np.where(gd_exfile)[0]

        # 
        nimages = np.sum(gd_exfile)
        nloop = nimages // nsub_fields + ((nimages % nsub_fields) > 0)

        # Write the file locally
            
        # Process them all, then deal with train/validation
        pp_fields, meta, img_idx = [], [], []
        for kk in range(nloop):
            f = h5py.File(local_file, mode='r')

            # Load the images into memory
            i0 = kk*nsub_fields
            i1 = min((kk+1)*nsub_fields, nimages)
            print('Fields: {}:{} of {}'.format(i0, i1, nimages))
            fields = f['fields'][i0:i1]
            shape =fields.shape
            masks = f['masks'][i0:i1].astype(np.uint8)
            sub_idx = np.arange(i0, i1).tolist()

            # Convert to lists
            print('Making lists')
            fields = np.vsplit(fields, shape[0])
            fields = [field.reshape(shape[1:]) for field in fields]

            masks = np.vsplit(masks, shape[0])
            masks = [mask.reshape(shape[1:]) for mask in masks]

            items = [item for item in zip(fields,masks,sub_idx)]

            print('Process time')
            # Do it
            with ProcessPoolExecutor(max_workers=n_cores) as executor:
                chunksize = len(items) // n_cores if len(items) // n_cores > 0 else 1
                answers = list(tqdm(executor.map(map_fn, items,
                                                chunksize=chunksize), total=len(items)))

            # Deal with failures
            answers = [f for f in answers if f is not None]

            # Slurp
            pp_fields += [item[0] for item in answers]
            img_idx += [item[1] for item in answers]
            meta += [item[2] for item in answers]

            del answers, fields, masks, items
            f.close()

        # Remove local_file
        os.remove(local_file)
        print("Removed: {}".format(local_file))

        # Write
        modis_tbl = pp_utils.write_pp_fields(pp_fields, 
                                 meta, 
                                 modis_tbl, 
                                 ex_idx, 
                                 img_idx,
                                 valid_fraction, 
                                 s3_file, 
                                 local_outfile)

        # Write to s3
        ulmo_io.upload_file_to_s3(local_outfile, s3_file)

    print("Done with generating pre-processed files..")
    return modis_tbl