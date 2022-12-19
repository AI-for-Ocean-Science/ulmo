""" Script to run Web Portal"""

import enum
from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Run Web Portal')
    parser.add_argument("--table_file", type=str, help="Run the Web Portal on this Table")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(pargs):
    """ Run
    """
    import numpy as np
    import json
    import os

    import h5py

    from ulmo.webpage_dynamic import os_portal
    from ulmo import io as ulmo_io
    from ulmo import defs 

    from IPython import embed

    # Parse the JSON file
    with open(pargs.input_file, 'rt') as fh:
        idict = json.load(fh)
    

    # Table
    main_tbl = ulmo_io.load_main_table(idict['table_file'])

    # Cut on image key


    # Cut if train/valid
    if idict['image_key'] in ['valid', 'train']:
        pp_type = 'pp_type' if 'pp_type' not in idict.keys() else idict['pp_type']
        keep = main_tbl[pp_type] == defs.mtbl_dmodel['pp_type'][idict['image_key']]
        main_tbl = main_tbl[keep].copy()

    # Cut down further?
    if idict['Nimages'] == 0:
        keep = np.array([True]*len(main_tbl))
    else:
        keep = np.array([False]*len(main_tbl))
        keep[np.arange(min(idict['Nimages'], 
                       len(main_tbl)))] = True 
    main_tbl = main_tbl[keep].copy()

    # Indices
    pp_idx = 'pp_idx' if 'pp_idx' not in idict.keys() else idict['pp_idx']
    sub_idx = main_tbl[pp_idx].values
    
    print("Loading images")

    f = h5py.File(idict['image_file'], 'r') 
    images = f[idict['image_key']][sub_idx, 0,:,:]
    f.close()
    print("Done")

    # Metrics
    metric_dict = dict(obj_ID=sub_idx)
    for metric in idict['metrics']:
        if metric[0] == 'DT':
            metric_dict[metric[0]] = (main_tbl.T90-main_tbl.T10).values[sub_idx]
        elif metric[1] == 'min_slope':
            metric_dict[metric[0]] = np.minimum(main_tbl.merid_slope.values,
                                                main_tbl.zonal_slope.values)[sub_idx]
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
        sess = os_portal.OSPortal(data_dict)
        return sess(doc)

    # Do me!
    server = os_portal.Server({'/': get_session}, num_procs=1)
    server.start()
    print('Opening Bokeh application for OS data on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
