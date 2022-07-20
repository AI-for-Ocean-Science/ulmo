""" Analysis related methods """

import os
import numpy as np

from ulmo import io as ulmo_io

s3_llc_match_table_file = 's3://llc/Tables/llc_viirs_match.parquet'
s3_llc_uniform_table_file = 's3://llc/Tables/LLC_uniform144_r0.5.parquet'
s3_viirs_table_file = 's3://viirs/Tables/VIIRS_all_98clear_std.parquet'

def load_table(dataset, local=False, cut_lat=57.):

    if dataset[0:3] == 'llc':
        if dataset == 'llc_match':
            s3_file = s3_llc_match_table_file
        elif dataset == 'llc_uniform':
            s3_file = s3_llc_uniform_table_file
        else:
            raise IOError("Bad llc dataset!")
        if local:
            tbl_file = os.path.join(os.getenv('SST_OOD'),
                'LLC', 'Tables', os.path.basename(s3_file))
        else:
            tbl_file = s3_file
    elif dataset == 'viirs':
        if local:
            tbl_file = os.path.join(os.getenv('SST_OOD'),
                'VIIRS', 'Tables', os.path.basename(s3_viirs_table_file))
        else:
            tbl_file = s3_viirs_table_file
    # Load
    tbl = ulmo_io.load_main_table(tbl_file)

    # Cut?
    if cut_lat is not None:
        tbl = tbl[tbl.lat < cut_lat].copy()
        tbl.reset_index(drop=True, inplace=True)

    # Expunge Nan
    finite = np.isfinite(tbl.LL)
    tbl = tbl[finite]
    tbl.reset_index(drop=True, inplace=True)

    # Return
    return tbl