""" Script to run Web Portal"""

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Run Web Portal')
    parser.add_argument("input_file", type=str, help="Input JSON file")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(pargs):
    """ Run
    """
    import numpy as np
    import os
    import json

    import h5py
    import pandas

    from ulmo.webpage_dynamic import os_portal

    from IPython import embed

    # Parse the JSON file
    with open(pargs.input_file, 'rt') as fh:
        idict = json.load(fh)
    
    # Load images 
    sub_idx = np.arange(idict['Nimages'])
    f = h5py.File(idict['image_file'], 'r') 
    images = f[idict['image_key']][sub_idx,0,:,:]
    f.close()

    # xy
    if idict['xy_values'] == 'UMAP':
        f = np.load(idict['UMAP_file'], allow_pickle=False)
        e1, e2 = f['e1'], f['e2']
    else:
        raise IOError("Not ready for multiple x,y")

    # Metrics
    res = pandas.read_parquet(idict['metric_file'])
    metric_dict = dict(obj_ID=sub_idx)
    for metric in idict['metrics']:
        if metric[0] == 'DT':
            metric_dict[metric[0]] = (res.T90-res.T10).values[sub_idx]
        else:
            metric_dict[metric[0]] = res[metric[1]].values[sub_idx]

    # Repack
    data_dict = {
        'images': images,
        'xy_scatter': dict(UMAP=(np.array([e1, e2]).T)[sub_idx]),
        'metrics': metric_dict,
    }

    # Odd work-around
    def get_session(doc):
        sess = os_portal.os_web(data_dict)
        return sess(doc)

    # Do me!
    server = os_portal.Server({'/': get_session}, num_procs=1)
    server.start()
    print('Opening Bokeh application for OS data on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
