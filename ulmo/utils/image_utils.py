import os
import numpy as np
import h5py

import pandas

from IPython import embed

sst_path = '/Volumes/Aqua-1/MODIS/uri-ai-sst/OOD' if os.getenv('SST_OOD') is None else os.getenv('SST_OOD')
eval_path = os.path.join(sst_path, 'Evaluations')
extract_path = os.path.join(sst_path, 'Extractions')
preproc_path = os.path.join(sst_path, 'PreProc')


def grab_img(example, itype, ptype='std', preproc_file=None):
    """

    Parameters
    ----------
    example : pandas.Row
    itype : str  Type of image
        Extracted =
        PreProc =
    ptype : str, optional
        Processing step
    preproc_file : str, optional
        If not provided, the year of the example is used + SST defaults

    Returns
    -------

    """


    if itype == 'Extracted':
        year = example.date.year
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
        if preproc_file is None:
            year = example.date.year
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


