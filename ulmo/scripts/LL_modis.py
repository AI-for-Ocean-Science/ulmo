""" Script to calculate LL for a field in a MODIS image"""

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='LL for a MODIS image')
    parser.add_argument("file", type=str, help="MODIS filename")
    parser.add_argument("row", type=int, help="Row for the field")
    parser.add_argument("col", type=int, help="Column for the field")
    parser.add_argument("-s","--show", default=False, action="store_true", help="Show pre-processed image?")
    #parser.add_argument("-g", "--galaxy_options", type=str, help="Options for fg/host building (photom,cigale)")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(pargs):
    """ Run
    """
    import warnings
    from matplotlib import pyplot as plt
    import seaborn as sns

    from ulmo import io as ulmo_io
    from ulmo.preproc import utils as pp_utils
    from ulmo.plotting import plotting

    # Load the image
    sst, qual, latitude, longitude = ulmo_io.load_nc(pargs.file, verbose=False)

    # Generate the masks
    masks = pp_utils.build_mask(sst, qual)

    # Grab the field and mask
    field_size = (128, 128)
    row, col = pargs.row, pargs.col

    field = sst[row:row + field_size[0], col:col + field_size[1]]
    mask = masks[row:row + field_size[0], col:col + field_size[1]]

    print("This field has {:0.1f}% cloud coverage".format(100*mask.sum()/field.size))

    # Pre-process
    pp_field, mu = pp_utils.preproc_field(field, mask)

    # Show?
    if pargs.show:
        pal, cm = plotting.load_palette()
        plt.clf()
        ax = plt.gca()
        sns.heatmap(pp_field, ax=ax, xticklabels=[], yticklabels=[], cmap=cm, vmin=-5, vmax=5)
        plt.show()
