""" Script to genreate a reconstructed cutout """

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='Reconstruct an image')
    parser.add_argument("preproc_file", type=str, help="File+path to Preproc file")
    parser.add_argument("table_file", type=str, help="Evaluation table filename (.csv)")
    parser.add_argument("row", type=int, help="Row in the Table (not the cutout row)")
    parser.add_argument("outroot", type=str, help="Filename for the output numpy file without the .npy")
    parser.add_argument("-s","--show", default=False, action="store_true", help="Show pre-processed image?")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(pargs):
    """ Run
    """
    import numpy as np
    from matplotlib import pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    import pandas

    from ulmo.plotting import plotting
    from ulmo.utils import image_utils
    from ulmo.utils import models as model_utils

    # Load the table
    print("Loading the table..")
    df = pandas.read_csv(pargs.table_file)
    example = df.iloc[pargs.row]

    # Grab it
    field, mask = image_utils.grab_img(example, 'PreProc', ptype='std',
                                       preproc_file=pargs.preproc_file)
    fields = np.reshape(field, (1,1,64,64))

    # Load up the model
    pae = model_utils.load('standard')
    # Reconstruct
    recons = pae.reconstruct(fields)

    # Save em
    np.save(pargs.outroot+'_orig.npy', field, allow_pickle=False)
    np.save(pargs.outroot+'_recon.npy', recons[0,0,...], allow_pickle=False)
    print("Wrote: {}".format(pargs.outfile))

    # Show?
    if pargs.show:
        vmnx=(-5,5)
        # Plot
        fig = plt.figure(figsize=(10, 4))
        pal, cm = plotting.load_palette()
        plt.clf()
        gs = gridspec.GridSpec(1, 2)

        # Original
        ax1 = plt.subplot(gs[0])
        sns.heatmap(field[0, ...], ax=ax1, xticklabels=[], yticklabels=[], cmap=cm,
                    vmin=vmnx[0], vmax=vmnx[1])

        # Reconstructed
        ax2 = plt.subplot(gs[1])
        sns.heatmap(recons[0, 0, ...], ax=ax2, xticklabels=[], yticklabels=[], cmap=cm,
                    vmin=vmnx[0], vmax=vmnx[1])
        plt.show()

