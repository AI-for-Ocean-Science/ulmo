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

def show_image(img:np.ndarray, cm=None, cbar=True, flipud=True,
               vmnx=(None,None)):
    """Dispay the cutout image

    Args:
        img (np.ndarray): cutout image
        cm ([type], optional): Color map to use. Defaults to None.
            If None, load the heatmap above
        cbar (bool, optional): If True, show a color bar. Defaults to True.
        flipud (bool, optional): If True, flip the image up/down. Defaults to True.
        vmnx (tuple, optional): Set vmin, vmax. Defaults to None

    Returns:
        matplotlib.Axis: axis containing the plot
    """
    if cm is None:
        _, cm = load_palette()
    #
    ax = sns.heatmap(np.flipud(img), xticklabels=[], 
                     vmin=vmnx[0], vmax=vmnx[1],
                     yticklabels=[], cmap=cm, cbar=cbar)
    #
    return ax


def set_fontsize(ax, fsz):
    """
    Set the fontsize throughout an Axis

    Args:
        ax (Matplotlib Axis):
        fsz (float): Font size

    Returns:

    """
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsz)


def umap_gallery(main_tbl, outfile=None, point_sz_scl=1., width=800, 
                 height=800, vmnx=(-1000.,None)):

    num_samples = len(main_tbl)
    point_size = point_sz_scl / np.sqrt(num_samples)
    dpi = 100

    # New plot
    plt.figure(figsize=(width//dpi, height//dpi))
    ax = plt.gca()
    img = ax.scatter(main_tbl.U0, main_tbl.U1,
            s=point_size, c=main_tbl.LL, 
            cmap='jet', vmin=vmnx[0], vmax=vmnx[1])
    cb = plt.colorbar(img, pad=0.)
    cb.set_label('LL', fontsize=20.)
    #
    ax.set_xlabel(r'$U_0$')
    ax.set_ylabel(r'$U_1$')
    set_fontsize(ax, 15.)

    if outfile is not None:
        plt.savefig(outfile, dpi=300)
    #