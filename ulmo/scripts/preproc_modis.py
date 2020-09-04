""" Script to calculate LL for a field in a MODIS image"""

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Preproc list of MODIS fields. Filename, row, col')
    parser.add_argument("file", type=str, help="ASCII list of fields for pre-processing")
    parser.add_argument("outfile", type=str, help="Output file.  Should have .h5 extension")
    parser.add_argument("--field_size", type=str, default='128,128', help='Field size, before downscale.  e.g. 128,128')
    parser.add_argument("--skip_inpaint", default=False, action="store_true",
                        help="Skip inpainting?")
    parser.add_argument("-v", "--verbose", default=False, action="store_true",
                        help="Verbose messaging?")

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
    import warnings
    import h5py

    import pandas

    from ulmo import io as ulmo_io
    from ulmo.preproc import utils as pp_utils

    # Init
    field_size = tuple([int(isz) for isz in pargs.field_size.split(',')])

    # Load the list
    tbl = pandas.read_csv(pargs.file, names=['file', 'row', 'col'])

    # For saving
    pp_fields, CCs, mus = [], [], []

    # Pre-processing dict
    pdict = dict(inpaint=True, median=True, med_size=(3, 1),
                      downscale=True, dscale_size=(2, 2))

    # Loop on the rows
    for iid, tfield in tbl.iterrows():
        # Load the image
        sst, qual, latitude, longitude = ulmo_io.load_nc(tfield.file, verbose=False)

        # Generate the masks
        masks = pp_utils.build_mask(sst, qual)

        # Grab the field and mask
        row, col = tfield.row, tfield.col

        # Snip
        field = sst[row:row + field_size[0], col:col + field_size[1]]
        mask = masks[row:row + field_size[0], col:col + field_size[1]]
        CC = 100*mask.sum()/field.size
        if pargs.verbose:
            print("This {} field has {:0.1f}% cloud coverage".format(field_size, CC))

        # Pre-process
        pp_field, mu = pp_utils.preproc_field(field, mask, **pdict)
        if pp_field is None:
            print("Field at {},{} in {} failed preprocessing. Saving a null image".format(
                tfield.row, tfield.col, tfield.file))
            pp_field = np.zeros((field_size[0]/2, field_size[0]/2))

        # Save
        pp_fields.append(pp_field)
        CCs.append(CC)
        mus.append(mu)

    # Add to pandas table
    tbl['CC'] = CCs
    tbl['mu'] = mus

    # Write to disk
    tbl.to_hdf(pargs.outfile, 'meta', mode='w')

    f = h5py.File(pargs.outfile, mode='a')
    # Add pre-processing steps
    for key in pdict:
        f.attrs[key] = pdict[key]
    # Add images
    dset = f.create_dataset("pp_fields", compression='gzip', data=pp_fields)
    f.close()

    print("Wrote: {}".format(pargs.outfile))
