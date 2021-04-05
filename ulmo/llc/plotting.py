""" Plotting routines"""
# Hiding cartopy

import pandas

from ulmo.plotting import plotting
from ulmo.llc import io as llc_io


def show_cutout(cutout:pandas.core.series.Series): 
    """Simple wrapper for showing the input cutout

    Args:
        cutout (pandas.core.series.Series): Cutout to display
    """


    # Load image
    img = llc_io.grab_image(cutout, close=True)

    # Plot
    plotting.show_image(img)
