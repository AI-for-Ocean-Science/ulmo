"""
Simple script to run Evals
"""

import os
import numpy as np

import xarray

import h5py

from ulmo import io as ulmo_io
from ulmo.preproc import extract
from ulmo.preproc import utils as pp_utils
from ulmo.ood import ood

from IPython import embed

dpath = '/home/xavier/Projects/Oceanography/AI/OOD/Scan'


def scan_granule(mask_file, data_file, outfile):
    CC_max = 0.05
    nadir_offset = 480
    field_size = (128, 128)
    # Load mask
    xd_mask = xarray.load_dataset(os.path.join(dpath, mask_file))
    msk = xd_mask.new_mask.data.astype(int)

    # Load data
    sst_tot, qual, latitude, longitude = ulmo_io.load_nc(os.path.join(dpath,data_file))

    # Restrict to near nadir
    nadir_pix = sst_tot.shape[1] // 2
    lb = nadir_pix - nadir_offset
    ub = nadir_pix + nadir_offset
    sst = sst_tot[:, lb:ub]
    mask = msk[:, lb:ub].astype(np.uint8)

    # Clear regions
    CC_mask = extract.uniform_filter(mask.astype(float), field_size[0], mode='constant', cval=1.)
    # Clear
    mask_edge = np.zeros_like(mask)  # , dtype=bool)
    mask_edge[:field_size[0] // 2, :] = True
    mask_edge[-field_size[0] // 2:, :] = True
    mask_edge[:, -field_size[0] // 2:] = True
    mask_edge[:, :field_size[0] // 2] = True
    clear = (CC_mask < CC_max) & np.logical_not(mask_edge)

    # Indices
    idx_clear = np.where(clear)
    nclear = idx_clear[0].size

    # Offset
    picked_row = idx_clear[0] - field_size[0] // 2
    picked_col = idx_clear[1] - field_size[0] // 2

    # Extract
    rows = picked_row
    cols = picked_col
    clear_fracs = CC_mask[idx_clear]

    masks = mask.copy()

    # Loop
    fields, field_masks = [], []
    metadata = []
    count = 0
    inpaint = True
    for r, c, clear_frac in zip(rows, cols, clear_fracs):
        if (count % 100) == 0:
            print(count)
        # Inpaint?
        field = sst[r:r + field_size[0], c:c + field_size[1]]
        mask = masks[r:r + field_size[0], c:c + field_size[1]]
        if inpaint:
            field, _ = pp_utils.preproc_field(field, mask, only_inpaint=True)
        if field is None:
            continue
        # Append SST and mask
        fields.append(field.astype(np.float32))
        field_masks.append(mask)
        # meta
        row, col = r, c + lb
        lat = latitude[row + field_size[0] // 2, col + field_size[1] // 2]
        lon = longitude[row + field_size[0] // 2, col + field_size[1] // 2]
        metadata.append([data_file, str(row), str(col), str(lat), str(lon), str(clear_frac)])
        count += 1

    # Stack
    fields = np.stack(fields)
    field_masks = np.stack(field_masks)
    metadata = np.stack(metadata)

    # Write
    columns = ['filename', 'row', 'column', 'latitude', 'longitude', 'clear_fraction']

    with h5py.File(outfile, 'w') as f:
        # f.create_dataset('fields', data=fields.astype(np.float32))
        f.create_dataset('fields', data=fields)
        f.create_dataset('masks', data=field_masks.astype(np.uint8))
        dset = f.create_dataset('metadata', data=metadata.astype('S'))
        dset.attrs['columns'] = columns

    print("Wrote: {}".format(outfile))


def run_evals(datadir, filepath, data_file, log_prob_file, clobber=False):

    # Load model
    #if flavor == 'loggrad':
    #    datadir = './Models/R2019_2010_128x128_loggrad'
    #    filepath = 'PreProc/MODIS_R2019_2010_95clear_128x128_preproc_loggrad.h5'
    pae = ood.ProbabilisticAutoencoder.from_json(datadir + '/model.json',
                                                 datadir=datadir,
                                                 filepath=filepath,
                                                 logdir=datadir)
    pae.load_autoencoder()
    pae.load_flow()

    print("Model loaded!")

    # Input
    # Check
    if not os.path.isfile(data_file):
        raise IOError("This data file does not exist! {}".format(data_file))

    # Output
    if os.path.isfile(log_prob_file) and not clobber:
        print("Eval file {} exists! Skipping..".format(log_prob_file))
        return

    # Run
    pae.compute_log_probs(data_file, 'valid', log_prob_file, csv=True)


def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Preproc images in an H5 file.')
    parser.add_argument("years", type=str, help="Begin, end year:  e.g. 2010,2012")
    parser.add_argument("flavor", type=str, help="Model (std, loggrad)")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs




# Command line execution
if __name__ == '__main__':

    #scan_granule('AQUA_MODIS_20100619T172508_L2_SST_pcc_Mask.nc',
    #             'AQUA_MODIS.20100619T172508.L2.SST.nc',
    #             'LLscan_20100619T172508_L2.h5')

    # Run
    # Night
    #run_evals('./Models/R2019_2010_128x128_std',
    #          'PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5',
    #          'Scan/LL_map_preproc.h5', 'Scan/LL_map_log_prob.h5')
    # Day
    run_evals('./Models/R2019_2010_128x128_std',
              'PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5',
              'Scan/LLscan_20100619T172508_L2_preproc.h5',
              'Scan/LLscan_20100619T172508_L2_log_prob.h5')

