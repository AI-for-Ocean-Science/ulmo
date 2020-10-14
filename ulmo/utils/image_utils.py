import os
import numpy as np
import h5py

import pandas

from IPython import embed

eval_path = os.path.join(os.getenv("SST_OOD"), 'Evaluations')
extract_path = os.path.join(os.getenv("SST_OOD"), 'Extractions')
preproc_path = os.path.join(os.getenv("SST_OOD"), 'PreProc')


def grab_img(example, itype, ptype='std'):

    year = example.date.year

    if itype == 'Extracted':
        print("Extracting")
        # Grab out of Extraction file
        extract_file = os.path.join(extract_path,
                                    'MODIS_R2019_{}_95clear_128x128_inpaintT.h5'.format(year))
        f = h5py.File(extract_file, mode='r')
        key = 'metadata'
        meta = f[key]
        df_ex = pandas.DataFrame(meta[:].astype(np.unicode_), columns=meta.attrs['columns'])

        # Find the match
        imt = (df_ex.filename.values == example.filename) & (
                df_ex.row.values.astype(int) == example.row) & (
                      df_ex.column.values.astype(int) == example.column)
        assert np.sum(imt) == 1
        index = df_ex.iloc[imt].index[0]

        # Grab image + mask
        field = f['fields'][index]
        mask = f['masks'][index]
        f.close()
    elif itype == 'PreProc':
        # Grab out of PreProc file
        preproc_file = os.path.join(preproc_path,
                                    'MODIS_R2019_{}_95clear_128x128_preproc_{}.h5'.format(year, ptype))
        f = h5py.File(preproc_file, mode='r')
        key = 'valid_metadata'
        meta = f[key]

        df_pp = pandas.DataFrame(meta[:].astype(np.unicode_), columns=meta.attrs['columns'])

        # Find the match
        imt = (df_pp.filename.values == example.filename) & (
                df_pp.row.values.astype(int) == example.row) & (
                      df_pp.column.values.astype(int) == example.column)
        assert np.sum(imt) == 1
        index = df_pp.iloc[imt].index[0]

        # Grab image + mask
        field = f['valid'][index]
        mask = None
        f.close()

    # Return
    return field, mask


