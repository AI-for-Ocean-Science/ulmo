""" Plotting routines"""
# Hiding cartopy

import pandas

from ulmo.plotting import plotting
from ulmo.utils import image_utils


def show_cutout(cutout:pandas.core.series.Series,
                local_file:str=None,
                ax=None): 
    """Simple wrapper for showing the input cutout

    Args:
        cutout (pandas.core.series.Series): Cutout to display
        local_file (str, optional): Local file to use. Defaults to None.
        ax (matplotlib.axes._subplots.AxesSubplot, optional): 
            Axes to use. Defaults to None.
    """


    # Load image
    img = image_utils.grab_image(cutout, close=True, local_file=local_file)

    # Plot
    plotting.show_image(img, ax=ax)
