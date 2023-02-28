import os
import numpy as np
import h5py
import healpy as hp

import pandas

from ulmo import io as ulmo_io

from IPython import embed

# Deal with paths

# MODIS L2
def set_paths(root):
    paths = {}
    paths['sst'] = '/Volumes/Aqua-1/MODIS/uri-ai-sst/OOD' if os.getenv('SST_OOD') is None else os.getenv('SST_OOD')
    paths['eval'] = os.path.join(sst_path, root, 'Evaluations')
    paths['extract'] = os.path.join(sst_path, root, 'Extractions')
    paths['preproc'] = os.path.join(sst_path, root, 'PreProc')
    return paths


def grab_modis_l2_img(example, itype, ptype='std', preproc_file=None):
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
    paths = set_paths('MODIS_L2')

    if itype == 'Extracted':
        year = example.date.year
        print("Extracting")
        # Grab out of Extraction file
        extract_file = os.path.join(paths['extract'],
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
            preproc_file = os.path.join(paths['preproc'],
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



def evals_to_healpix(eval_tbl, nside, log=False, mask=True, 
                     extras=None):
    """
    Generate a healpix map of where the input
    MHW Systems are located on the globe

    Parameters
    ----------
    mhw_sys : pandas.DataFrame
    nside : int
    mask : bool, optional
    extras : list, optional

    Returns
    -------
    healpix_array, lats, lons, extras (optional) : hp.ma, np.ndarray, 
        np.ndarray, dict

    """
    # Grab lats, lons
    lats = eval_tbl.lat.values
    lons = eval_tbl.lon.values

    # Healpix coords
    theta = (90 - lats) * np.pi / 180.
    phi = lons * np.pi / 180.
    idx_all = hp.pixelfunc.ang2pix(nside, theta, phi)

    # Count events
    npix_hp = hp.nside2npix(nside)
    all_events = np.ma.masked_array(np.zeros(npix_hp, dtype='int'))
    for idx in idx_all:
        all_events[idx] += 1
    
    # Extras?
    if extras is not None:
        # Init
        hp_extras = {}
        for extra in extras:
            hp_extras[extra] = np.zeros(npix_hp)
        # Loop me
        uidx = np.unique(idx_all)
        for iuidx in uidx:
            mt = idx_all == iuidx
            # Loop on extras
            for extra in extras:
                hp_extras[extra][iuidx] = np.median(eval_tbl[extra][mt])


    # Zero me
    zero = all_events == 0
    if log:
        float_events = np.zeros_like(all_events).astype(float)
        float_events[~zero] = np.log10(all_events[~zero].astype(float))
    else:
        float_events = all_events.astype(float)


    # Mask
    hpma = hp.ma(float_events)
    if mask:
        hpma.mask = zero

    # Angles
    hp_lons, hp_lats = hp.pixelfunc.pix2ang(nside, np.arange(npix_hp), 
                                            lonlat=True)

    # Return
    if extras is None:
        return hpma, hp_lons, hp_lats
    else:
        return hpma, hp_lons, hp_lats, hp_extras


def grab_image(cutout:pandas.core.series.Series, 
               close=True, pp_hf=None, local_file=None):
    """Grab a cutout image

    Args:
        cutout (pandas.core.series.Series): [description]
        close (bool, optional): [description]. Defaults to True.
        pp_hf ([type], optional): Pointer to the HDF5 file. Defaults to None.
        local_file (str, optional): Use this file, if provided

    Returns:
        [type]: [description]
    """
    if local_file is not None:
        pp_hf = h5py.File(local_file, 'r')
    # Open?
    if pp_hf is None:
        with ulmo_io.open(cutout.pp_file, 'rb') as f:
            pp_hf = h5py.File(f, 'r')
    img = pp_hf['valid'][cutout.pp_idx, 0, ...]

    # Close?
    if close:
        pp_hf.close()
        return img
    else:
        return img, pp_hf