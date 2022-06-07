"""
Module to run tests on modules
"""
import os

import pytest
import numpy as np

import pandas

from ulmo.models import io as model_io
from IPython import embed

def test_ssh():
    """ SSH model test"""
    pae = model_io.load_ssh()
    pytest.set_trace()

def test_modis_l2():
    """ MODIS L2 test"""
    pae = model_io.load_modis_l2()
    # Test
    assert pae.datadir == 's3://modis-l2/Models/R2019_2010_128x128_std'
