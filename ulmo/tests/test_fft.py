"""
Module to run tests on FFT
"""
import os

import pytest
import numpy as np

import pandas

from ulmo.analysis import fft 
from ulmo.tests import tst_utils

from IPython import embed


def test_ffft():
    """ Test fast fft and slopes """

    # Load up a cutout image
    img = np.load(tst_utils.data_path('cutout_image.npy'))

    # Analyze
    zonal_results, merid_results = fft.analyze_cutout(img, dtdm=True)

    # Test
    assert 'slope_large_err' in zonal_results.keys()
    assert 'slope_large_err' in merid_results.keys()