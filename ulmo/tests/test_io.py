"""
Module to run tests on arcoadd
"""
import os

import pytest
import numpy as np

import pandas

from ulmo import io as ulmo_io
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
