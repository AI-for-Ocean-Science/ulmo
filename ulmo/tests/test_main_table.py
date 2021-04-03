"""
Module to run tests on arcoadd
"""
import os

import pytest
import numpy as np

import pandas

from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils

from IPython import embed

def data_path(filename):
    data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'files')
    return os.path.join(data_dir, filename)


def test_vet_tbl():
    # Load
    tbl = ulmo_io.load_main_table(data_path('tst_table.parquet'))
    # Vet
    assert cat_utils.vet_main_table(tbl)
