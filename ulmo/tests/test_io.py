"""
Module to run tests on arcoadd
"""
import os

import pytest
import numpy as np

import pandas

from ulmo import io as ulmo_io
from ulmo.utils import image_utils 
from ulmo.tests import tst_utils
from IPython import embed

def test_pandas_s3():
    """ Test we can write to s3 """
    df = pandas.DataFrame()
    df['tmp'] = np.arange(1000)
    #
    outfile = 's3://test/tst.feather'
    # Delete
    ulmo_io.s3.meta.client.delete_object(Bucket='test', 
                                         Key='tst.feather')
    # Write
    ulmo_io.write_pandas_to_s3_feather(df, outfile)
    # Test
    #files = ulmo_io.s3.meta.client.list_objects(Bucket='test')
    bucket = ulmo_io.s3.Bucket('test')
    objs = bucket.objects.all()
    files = [obj.key for obj in objs]
    assert 'tst.feather' in files

    # Remove
    ulmo_io.s3.meta.client.delete_object(Bucket='test', 
                                         Key='tst.feather')

def test_grab_image():
    modis_tbl = ulmo_io.load_main_table('s3://modis-l2/Tables/MODIS_L2_std.parquet')
    # Random
    cutout = modis_tbl.iloc[10]
    # Grab image
    img = image_utils.grab_image(cutout)

    assert isinstance(img, np.ndarray)
    assert img.shape[0] == 64

    # Save to files -- Leave this commented out
    #np.save(tst_utils.data_path('cutout_image.npy'), img)