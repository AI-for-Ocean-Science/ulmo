import torch
import skimage
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec


def load_palette():
    with open('./color_palette.txt', 'r') as f:
        colors = np.array([l.split() for l in f.readlines()]).astype(np.float32)
        pal = sns.color_palette(colors)
        boundaries = np.linspace(0, 1, 64)
        colors = list(zip(boundaries, colors))
        cm = LinearSegmentedColormap.from_list(name='rainbow', colors=colors)
    return pal, cm


def grid_plot(nrows, ncols):
    
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