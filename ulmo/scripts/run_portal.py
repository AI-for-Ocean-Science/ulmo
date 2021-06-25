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
    from ulmo import io as ulmo_io

    from IPython import embed

    # Parse the JSON file
    with open(pargs.input_file, 'rt') as fh:
        idict = json.load(fh)
    

    # Table
    main_tbl = ulmo_io.load_main_table(idict['table_file'])

    # Load images 
    if idict['Nimages'] == 0:
        sub_idx = np.arange(len(main_tbl))
    else:
        sub_idx = np.arange(idict['Nimages'])
    print(f"Loading {sub_idx.size} images..")

    f = h5py.File(idict['image_file'], 'r') 
    images = f[idict['image_key']][sub_idx,0,:,:]
    f.close()
    print("Done")

    # Metrics
    metric_dict = dict(obj_ID=sub_idx)
    for metric in idict['metrics']:
        if metric[0] == 'DT':
            metric_dict[metric[0]] = (main_tbl.T90-main_tbl.T10).values[sub_idx]
        else:
            metric_dict[metric[0]] = main_tbl[metric[1]].values[sub_idx]

    # Repack
    data_dict = {
        'images': images,
        'xy_values': idict['xy_values'],
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
