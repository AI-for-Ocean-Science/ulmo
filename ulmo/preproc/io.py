""" Module for I/O related to pre-processing"""
import os
from pkg_resources import resource_filename

import json

preproc_path = os.path.join(resource_filename('ulmo', 'preproc'), 'options')

def load_options(root):
    filename = 'preproc_{}.json'.format(root)
    # Tuples
    with open(os.path.join(preproc_path, filename), 'rt') as fh:
        pdict = json.load(fh)
    # Tuple me
    for key in ['med_size', 'dscale_size']:
        if key in pdict:
            pdict[key] = tuple(pdict[key])
    # Return
    return pdict

