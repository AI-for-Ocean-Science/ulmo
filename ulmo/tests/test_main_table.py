"""
Module to run tests on arcoadd
"""
import os

import pytest
import numpy as np

import pandas

from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils
from ulmo.tests import tst_utils

from IPython import embed


def test_vet_tbl():
    # Load
    tbl = ulmo_io.load_main_table(tst_utils.data_path('tst_table.parquet'))
    # Vet
    assert cat_utils.vet_main_table(tbl)
