import torch
import skimage
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def load_palette():
    with open('./color_palette.txt', 'r') as f:
        colors = np.array([l.split() for l in f.readlines()]).astype(np.float32)
        pal = sns.color_palette(colors)
        boundaries = np.linspace(0, 1, 64)
        colors = list(zip(boundaries, colors))
        cm = LinearSegmentedColormap.from_list(name='rainbow', colors=colors)
    return pal, cm