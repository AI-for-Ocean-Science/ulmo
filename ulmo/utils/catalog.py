""" Catalog utility methods """
import numpy as np

from ulmo import defs

import pandas

from IPython import embed

def match_ids(IDs, match_IDs, require_in_match=True):
    """ Match input IDs to another array of IDs (usually in a table)
    Return the rows aligned with input IDs
    Parameters
    ----------
    IDs : ndarray
    match_IDs : ndarray
    require_in_match : bool, optional
      Require that each of the input IDs occurs within the match_IDs
    Returns
    -------
    rows : ndarray
      Rows in match_IDs that match to IDs, aligned
      -1 if there is no match
    """
    rows = -1 * np.ones_like(IDs).astype(int)
    # Find which IDs are in match_IDs
    in_match = np.in1d(IDs, match_IDs)
    if require_in_match:
        if np.sum(~in_match) > 0:
            raise IOError("qcat.match_ids: One or more input IDs not in match_IDs")
    rows[~in_match] = -1
    #
    IDs_inmatch = IDs[in_match]
    # Find indices of input IDs in meta table -- first instance in meta only!
    xsorted = np.argsort(match_IDs)
    ypos = np.searchsorted(match_IDs, IDs_inmatch, sorter=xsorted)
    indices = xsorted[ypos]
    rows[in_match] = indices
    return rows


def vet_main_table(table:pandas.DataFrame, cut_prefix=None):

    # Loop on the keys
    for key in table.keys():
        # Allow for cut prefix
        if cut_prefix is not None and len(key) > len(cut_prefix) and (
                key[:len(cut_prefix)] == cut_prefix):
            skey = key[len(cut_prefix):]
        else:
            skey = key
        assert skey in defs.mtbl_dmodel.keys(), f'{key} not in data model'
        # Check datat type
        assert isinstance(table.iloc[0][key], 
                          defs.mtbl_dmodel[skey]['dtype']), \
                            f'Bad data type for {key}'
    # Return
    return True