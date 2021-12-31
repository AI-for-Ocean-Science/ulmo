""" Script to run Web Portal"""

import enum
from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Run Web Portal')
    parser.add_argument("table_file", type=str, help="Input MODIS table file")
    parser.add_argument("outfile", type=str, help="Input MODIS table file")
    parser.add_argument('--nimages', type=int, help="Limit to the first nimages of the Table")
    parser.add_argument('--image_path', type=str, help="Full path to the individual image files")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(pargs):
    """ Run
    """
    import numpy as np
    import json
    import os

    import h5py

    from ulmo import io as ulmo_io

    from IPython import embed

    # Path
    if pargs.image_path is None:
        image_path = os.path.join(os.getenv('SST_OOD'),
                                  'MODIS_L2', 'PreProc')
    else:
        image_path = pargs.image_path

    # Table
    main_tbl = ulmo_io.load_main_table(pargs.table_file)

    # Cut down?
    if pargs.nimages is not None:
        keep = np.array([False]*len(main_tbl))
        keep[np.arange(min(pargs.nimages, len(main_tbl)))] = True 
        main_tbl = main_tbl[keep].copy()
    
    valid = main_tbl.ulmo_pp_type == 0
    train = main_tbl.ulmo_pp_type == 1

    # Load images 
    print("Loading images..")

    # Loop on unique filenames
    uni_files = np.unique(main_tbl.pp_file)
    images = None
    for ss, uni_file in enumerate(uni_files):
        print(f"Loading images from {uni_file}")
        valid_images = None
        train_images = None
        in_year = main_tbl.pp_file == uni_file
        # Read locally
        local_file = os.path.join(image_path, 
                                  os.path.basename(uni_file))
        f = h5py.File(local_file, 'r') 
        # Valid
        if np.any(valid & in_year):
            valid_images = f['valid'][main_tbl[valid & in_year].pp_idx.values,...]
        if valid_images is None:
            pass
        elif images is None:
            images = valid_images
        else:
            images = np.concatenate([images, valid_images])
        # Train
        if np.any(train & in_year):
            train_images = f['train'][main_tbl[train & in_year].pp_idx.values,...]
        if train_images is None:
            pass 
        elif images is None:
            images = train_images
        else:
            images = np.concatenate([images, train_images])
        # Close
        f.close()
    print("Done")

    assert images.shape[0] == len(main_tbl)
    # Write em!
    with h5py.File(pargs.outfile, 'w') as f:
        f.create_dataset('images', data=images)
    print(f"Wrote: {pargs.outfile}!")