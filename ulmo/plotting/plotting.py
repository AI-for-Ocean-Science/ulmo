""" Plotting utilities """
from pkg_resources import resource_filename
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec


def load_palette(pfile=None):
    """ Load the color pallette

    Args:
        pfile (str, optional): Filename of the pallette. Defaults to None.

    Returns:
        color pallette, LinearSegmentedColormap: pallette for sns, colormap
    """
    
    if pfile is None:
        pfile = os.path.join(resource_filename('ulmo', 'plotting'), 'color_palette.txt')
    # Load me up
    with open(pfile, 'r') as f:
        colors = np.array([l.split() for l in f.readlines()]).astype(np.float32)
        pal = sns.color_palette(colors)
        boundaries = np.linspace(0, 1, 64)
        colors = list(zip(boundaries, colors))
        cm = LinearSegmentedColormap.from_list(name='rainbow', colors=colors)
    return pal, cm


def grid_plot(nrows, ncols):
    """ Grid plot

    Args:
        nrows (int): Number of rows in the grid
        ncols (int): Number of cols in the grid

    Returns:
        plt.Figure, plt.axis: Plot and axes
    """
    
    # Make plot grid
    n, m = nrows, ncols # rows, columns
    t, b = 0.9, 0.1 # 1-top space, bottom space
    msp, sp = 0.1, 0.5 # minor spacing, major spacing

    offs = (1+msp)*(t-b)/(2*n+n*msp+(n-1)*sp) # grid offset
    hspace = sp+msp+1 #height space per grid

    gso = GridSpec(n, m, bottom=b+offs, top=t, hspace=hspace)
    gse = GridSpec(n, m, bottom=b, top=t-offs, hspace=hspace)
    
    width, height = (ncols/4)*18, (nrows/4)*25
    fig = plt.figure(figsize=(width, height))
    axes = []
    for i in range(n*m):
        axes.append((fig.add_subplot(gso[i]), fig.add_subplot(gse[i])))
    
    return fig, axes

def show_cutout(img:np.ndarray, cm=None, cbar=True):
    """Dispay the cutout image

    Args:
        img (np.ndarray): cutout image
        cm ([type], optional): Color map to use. Defaults to None.
            If None, load the heatmap above
        cbar (bool, optional): If True, show a color bar. Defaults to True.

    Returns:
        matplotlib.Axis: axis containing the plot
    """
    if cm is None:
        _, cm = load_palette()
    #
    ax = sns.heatmap(img, xticklabels=[], yticklabels=[], cmap=cm,
             cbar=cbar)
    #
    return ax