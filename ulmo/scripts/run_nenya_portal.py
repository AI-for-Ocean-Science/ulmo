""" Script to run SSL Web Portal"""

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Run SSL Web Portal')
    parser.add_argument("--table_file", type=str, help="Run the Web Portal on this Table")
    parser.add_argument("--image_file", type=str, help="Run the Web Portal on this image")
    parser.add_argument("--Uvalues", type=str, help="Comma separated U values. (e.g. 2.35,-1.2) [not to be used with --image_file]")

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

    from ulmo.ssl import portal
    from ulmo.ssl import analyze_image

    from IPython import embed

    # No input image
    if pargs.image_file is None:
        inp_Image = None
        if pargs.table_file is None:
            raise IOError("Need a table file until we figure it out from the image")
        table_file = pargs.table_file
    else:
        # Load
        if os.path.splitext(pargs.image_file)[1] == '.npy':
            img = np.load(pargs.image_file)
        else:
            raise IOError("Not ready for this image type")
        # Latents and UMAP
        embedding, pp_img, table_file, DT = analyze_image.umap_image('v4', img)
        Us = embedding[0,:]
        # Prep
        inp_Image = portal.Image(
            pp_img[0,0,...], Us.tolist(), 
            DT, lat=150., lon=150.)

    if pargs.Uvalues is not None:
        Uvalues = [float(item) for item in pargs.Uvalues.split(',')]
    else:
        Uvalues = None

    # Odd work-around
    def get_session(doc):
        sess = portal.OSSinglePortal(
            table_file, input_Image=inp_Image, 
            init_Us=Uvalues)
        return sess(doc)

    # Do me!
    server = portal.Server({'/': get_session}, num_procs=1)
    server.start()
    print('Opening Bokeh application for OS data on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
