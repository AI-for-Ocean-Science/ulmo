""" Plotting utilities """
from ulmo.utils import image_utils
from IPython.terminal.embed import embed
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
               vmnx=(None,None), show=False, set_aspect=None, clbl=None):
    """Dispay the cutout image

    Args:
        img (np.ndarray): cutout image
        cm ([type], optional): Color map to use. Defaults to None.
            If None, load the heatmap above
        cbar (bool, optional): If True, show a color bar. Defaults to True.
        flipud (bool, optional): If True, flip the image up/down. Defaults to True.
        vmnx (tuple, optional): Set vmin, vmax. Defaults to None
        set_aspect (str, optional):
            Passed to ax.set_aspect() if provided

    Returns:
        matplotlib.Axis: axis containing the plot
    """
    if cm is None:
        _, cm = load_palette()
    #
    ax = sns.heatmap(np.flipud(img), xticklabels=[], 
                     vmin=vmnx[0], vmax=vmnx[1],
                     yticklabels=[], cmap=cm, cbar=cbar, cbar_kws={'label': clbl, 'fontsize': 20})
    plt.savefig('image', dpi=600)
    
    if show:
        plt.show()
    if set_aspect is not None:
        ax.set_aspect(set_aspect)
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


def umap_gallery(main_tbl, outfile=None, point_sz_scl=1., 
                 width=800, 
                 height=800, vmnx=(-1000.,None), dxdy=(0.3, 0.3),
                 Nx=20, debug=False, skip_scatter=False,
                 fsz=15., ax=None):
    """Generate a UMAP plot and overplot a gallery
    of cutouts

    Args:
        main_tbl (pandas.DataFrame): Table of quantities
        outfile (str, optional): Outfile for the figure. Defaults to None.
        point_sz_scl (float, optional): Point size for UMAP points. Defaults to 1..
        width (int, optional): Width of the figure. Defaults to 800.
        height (int, optional): Height of the figure. Defaults to 800.
        vmnx (tuple, optional): Color bar vmin,vmax. Defaults to (-1000.,None).
        dxdy (tuple, optional): Amount to pad the xlim, ylim by. Defaults to (0.3, 0.3).
        Nx (int, optional): Number of cutout images in x to show. Defaults to 20.
        skip_scatter (bool, optional): Skip the scatter plot?
        debug (bool, optional): Debug? Defaults to False.
        ax (matplotlib.plt.Axes, optional): Use this axis!
        fsz (float, optional): fontsize

    Returns:
        matplotlib.plt.Axes: Axis
    """

    _, cm = load_palette()

    num_samples = len(main_tbl)
    point_size = point_sz_scl / np.sqrt(num_samples)
    dpi = 100

    # New plot
    if ax is None:
        plt.figure(figsize=(width//dpi, height//dpi))
        ax = plt.gca()
    if not skip_scatter:
        img = ax.scatter(main_tbl.U0, main_tbl.U1,
                s=point_size, c=main_tbl.LL, 
                cmap='jet', vmin=vmnx[0], vmax=vmnx[1])
        cb = plt.colorbar(img, pad=0.)
        cb.set_label('LL', fontsize=20.)
        #

    ax.set_xlabel(r'$U_0$')
    ax.set_ylabel(r'$U_1$')

    # Set boundaries
    xmin, xmax = main_tbl.U0.min()-dxdy[0], main_tbl.U0.max()+dxdy[0]
    ymin, ymax = main_tbl.U1.min()-dxdy[1], main_tbl.U1.max()+dxdy[1]
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # ###################
    # Gallery time

    # Grid
    xval = np.linspace(xmin, xmax, num=Nx)
    dxv = xval[1]-xval[0]
    yval = np.arange(ymin, ymax+dxv, step=dxv)

    # Ugly for loop
    pp_hf = None
    ndone = 0
    if debug:
        nmax = 100
    else:
        nmax = 1000000000
    for x in xval[:-1]:
        for y in yval[:-1]:
            pts = np.where((main_tbl.U0 >= x) & (main_tbl.U0 < x+dxv) & (
                main_tbl.U1 >= y) & (main_tbl.U1 < y+dxv))[0]
            if len(pts) == 0:
                continue

            # Pick a random one
            ichoice = np.random.choice(len(pts), size=1)
            idx = int(pts[ichoice])
            cutout = main_tbl.iloc[idx]

            # Image
            axins = ax.inset_axes(
                    [x, y, 0.9*dxv, 0.9*dxv], 
                    transform=ax.transData)
            try:
                cutout_img, pp_hf = image_utils.grab_image(cutout, 
                                                       pp_hf=pp_hf,
                                                       close=False)
            except:
                embed(header='198 of plotting')                                                    
            _ = sns.heatmap(np.flipud(cutout_img), xticklabels=[], 
                     #vmin=vmnx[0], vmax=vmnx[1],
                     yticklabels=[], cmap=cm, cbar=False,
                     ax=axins)
            ndone += 1
            print(f'ndone= {ndone}, LL={cutout.LL}')
            if ndone > nmax:
                break
        if ndone > nmax:
            break

    set_fontsize(ax, fsz)
    ax.set_aspect('equal', 'datalim')
    # Finish
    if outfile is not None:
        plt.savefig(outfile, dpi=300)
        print(f"Wrote: {outfile}")

    return ax