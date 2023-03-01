""" Bokeh portal for Ocean Sciences.  Based on Itamar Reiss' code 
and further modified by Kate Storrey-Fisher"""
from bokeh.models.widgets.tables import StringFormatter
import numpy as np
import os

import h5py
import pandas

from bokeh.server.server import Server
from bokeh.colors import RGB

from ulmo.webpage_dynamic import os_portal

# For the geography figure
from IPython import embed

# CODE HERE AND DOWN IS FOR TESTING
def get_test_os_session(doc):
    images, objids, metric, umapd = grab_dum_data()
    # Instantiate
    sess = os_portal.OSPortal(images, objids, metric, umapd)
    return sess(doc)

def get_modis_subset_os_session(doc):
    data_dict = grab_modis_subset()
    # Instantiate
    sess = os_portal.OSPortal(data_dict)
    return sess(doc)

def grab_dum_data():
    nobj = 100
    dum_images = np.random.uniform(size=(nobj, 64, 64))
    dum_objids = np.arange(nobj)
    dum_umap = np.random.uniform(size=(nobj,2)) 
    dum_LL = np.random.uniform(low=0., high=100., size=nobj)
    dum_metric = dict(LL=dum_LL)
    dum_umapd = dict(UMAP=dum_umap)
    embed(header='NEED TO REFACTOR')
    #
    return dum_images, dum_objids, dum_metric, dum_umapd

def grab_modis_subset():
    # Load up 
    sst_dir='/data/Projects/Oceanography/AI/OOD/SST/MODIS_L2/PreProc/'
    data_file = os.path.join(sst_dir, 
                             'MODIS_R2019_2010_95clear_128x128_preproc_std.h5')
    results_path = '/data/Projects/Oceanography/AI/SSL/portal/'
    umaps_path = os.path.join(results_path, 'embeddings')

    # Images
    nimgs = 100000
    sub_idx = np.arange(nimgs)
    f = h5py.File(data_file, 'r') 
    images = f["valid"][sub_idx,0,:,:]
    f.close()

    # UMAP
    umap_file = os.path.join(umaps_path, 'UMAP_2010_valid_v1.npz')
    f = np.load(umap_file, allow_pickle=False)
    e1, e2 = f['e1'], f['e2']

    # Metrics
    results_file = os.path.join(results_path, 'ulmo_2010_valid_v1.parquet')
    res = pandas.read_parquet(results_file)
    metric_dict = {'LL': res.LL.values[sub_idx], 
                   'lat': res.lat.values[sub_idx],
                   'lon': res.lon.values[sub_idx],
                   'avgT': res.mean_temperature.values[sub_idx],
                   'DT': (res.T90-res.T10).values[sub_idx],
                   'obj_ID': sub_idx,
    }

    # Repack
    data_dict = {
        'images': images,
        'xy_scatter': dict(UMAP=(np.array([e1, e2]).T)[sub_idx]),
        'metrics': metric_dict,
    }

    return data_dict


def main(flg):
    flg = int(flg)

    # Deprecated test
    if flg & (2 ** 0):
        pass
    
    # Test main class
    if flg & (2 ** 1):
        dum_images, dum_objids, dum_metric, dum_umapd = grab_dum_data()
        sess = os_portal.OSPortal(dum_images, dum_objids, dum_metric, dum_umapd)
        print("Success!")

    # Real deal
    if flg & (2 ** 2):
        server = Server({'/': get_test_os_session}, num_procs=1)
        server.start()
        print('Opening Bokeh application for test data on http://localhost:5006/')

        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()

    # Test modis subset
    if flg & (2 ** 3):
        data_dict = grab_modis_subset()
        sess = os_portal.OSPortal(data_dict)

    if flg & (2 ** 4):
        server = Server({'/': get_modis_subset_os_session}, num_procs=1)
        server.start()
        print('Opening Bokeh application for MODIS subset on http://localhost:5006/')

        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # Test bokeh
        #flg += 2 ** 1  # Test object
        #flg += 2 ** 2  # Full Test 
        #flg += 2 ** 3  # Test load MODIS subset
        flg += 2 ** 4  # Full MODIS subset
    else:
        flg = sys.argv[1]

    main(flg)
