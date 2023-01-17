""" Figures for SSL paper on MODIS """
from dataclasses import replace
from datetime import datetime
import os, sys
import numpy as np
import scipy
from scipy import stats
from urllib.parse import urlparse
import datetime

import argparse

import healpy as hp

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Ellipse


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas
import seaborn as sns

import h5py

from ulmo import plotting
from ulmo.utils import utils as utils

from ulmo import io as ulmo_io
from ulmo.ssl import single_image as ssl_simage
from ulmo.ssl import ssl_umap
from ulmo.ssl import defs as ssl_defs
from ulmo.utils import image_utils

from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import ssl_paper_analy
sys.path.append(os.path.abspath("../Figures/py"))
import fig_ssl_modis


def fig_uniform_gallery(outfile:str, table:str, 
                     umap_dim=2,
                     umap_comp='S0,S1',
                     seed=1235,
                     min_pts=1000,
                     in_vmnx=(-2., 2.),
                     nxy=4):

    local=True
    if seed is not None:
        np.random.seed(seed)

    # Load
    modis_tbl = ssl_paper_analy.load_modis_tbl(
        local=local, table=table)

    # UMAP
    umap_keys = ssl_paper_analy.gen_umap_keys(
        umap_dim, umap_comp)

    # Outfile


    # Grab the cutouts
    modis_tbl, cutouts, umap_grid = ssl_umap.cutouts_on_umap_grid(
        modis_tbl, nxy, umap_keys, min_pts=min_pts)
    ncutouts = len(cutouts)

    cutouts = [item for item in cutouts if item is not None]

    # Pick 9 at random
    choices = np.random.choice(len(cutouts), size=nxy*nxy, replace=False)

    _, cm = plotting.load_palette()
    fig = plt.figure(figsize=(10,9))
    plt.clf()
    gs = gridspec.GridSpec(nxy,nxy)

    cbar_kws = dict(label=r'$\delta T$ (K)')

    # Color bar
    for ii,choice in enumerate(choices):
        cutout = cutouts[choice]

        # Axis
        ax = plt.subplot(gs[ii])

        if ii == len(choices)-1:
            plt_cbar = True
            ax_cbar = ax.inset_axes([1.1,0.05,0.15,0.8])
        else:
            plt_cbar = False
            ax_cbar = None

        # This is only local
        parsed_s3 = urlparse(cutout.pp_file)
        local_file = os.path.join(os.getenv('SST_OOD'),
                                    'MODIS_L2',
                                    parsed_s3.path[1:])
        cutout_img = image_utils.grab_image(
            cutout, close=True, local_file=local_file)
                # Limits
        if in_vmnx[0] == -999:
            DT = cutout.T90 - cutout.T10
            vmnx = (-1*DT, DT)
        elif in_vmnx is not None:
            vmnx = in_vmnx
        else:
            imin, imax = cutout_img.min(), cutout_img.max()
            amax = max(np.abs(imin), np.abs(imax))
            vmnx = (-1*amax, amax)

        # Plot
        sns_ax = sns.heatmap(np.flipud(cutout_img), 
                        xticklabels=[], 
                    vmin=vmnx[0], vmax=vmnx[1],
                    yticklabels=[], cmap=cm, 
                    cbar=plt_cbar,
                    cbar_ax=ax_cbar, 
                    cbar_kws=cbar_kws,
                    ax=ax)
        sns_ax.set_aspect('equal', 'datalim')

    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_regional_with_gallery(geo_region:str, outfile:str, table:str, 
                     umap_comp='S0,S1', min_pts=200, cut_to_inner=None,
                     in_vmnx=None,
                     umap_dim=2, cmap='bwr', nxy=8):

    # Load
    local=True
    modis_tbl = ssl_paper_analy.load_modis_tbl(
        local=local, table=table)

    # UMAP
    umap_keys = ssl_paper_analy.gen_umap_keys(
        umap_dim, umap_comp)

    # UMAP the region
    counts, counts_geo, modis_tbl, umap_grid, xedges, yedges, = ssl_umap.regional_analysis(
        geo_region, modis_tbl, nxy, umap_keys, min_counts=min_pts)
    rtio_counts = counts_geo / counts

    # Figure
    fig = plt.figure(figsize=(12,5.3))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    ax_regional = plt.subplot(gs[0])

    values = rtio_counts.transpose()
    lbl = r'Relative Frequency ($f_b$)'
    vmin, vmax = 0, 2.
    mplt = ax_regional.pcolormesh(xedges, yedges, values, 
                         cmap=cmap, vmin=vmin, vmax=vmax) 
    cbaxes = plt.colorbar(mplt, pad=0., fraction=0.030)
    cbaxes.set_label(lbl, fontsize=15.)

    # Title
    if geo_region == 'eqpacific':
        title = f'Pacific ECT: '
    elif geo_region == 'eqindian':
        title = 'Equatorial Indian Ocean: '
    elif geo_region == 'gulfstream':
        title = 'Gulf Stream: '
    elif geo_region == 'coastalcali':
        title = 'Coastal California: '
    else:
        embed(header='777 of figs')

    # Add lon, lat
    lons = ssl_defs.geo_regions[geo_region]['lons']
    lats = ssl_defs.geo_regions[geo_region]['lats']
    title += f'lon={ssl_paper_analy.lon_to_lbl(lons[0])},'
    title += f'{ssl_paper_analy.lon_to_lbl(lons[1])};'
    title += f' lat={ssl_paper_analy.lat_to_lbl(lats[0])},'
    title += f'{ssl_paper_analy.lat_to_lbl(lats[1])}'
    ax_regional.set_title(title)

    # ##############################################3
    # Gallery
    _, cm = plotting.load_palette()
    ax_gallery = plt.subplot(gs[1])


    # Color bar
    xmin, xmax = umap_grid['xmin'], umap_grid['xmax']
    ymin, ymax = umap_grid['ymin'], umap_grid['ymax']
    dxv = umap_grid['dxv']
    dyv = umap_grid['dyv']
    xval = umap_grid['xval']
    yval = umap_grid['yval']


    ax_gallery.set_xlim(xmin, xmax+dxv)
    ax_gallery.set_ylim(ymin, ymax)

    plt_cbar = True
    ax_cbar = ax_gallery.inset_axes(
                    [xmax + dxv + dxv/10, ymin, dxv/2, (ymax-ymin)*0.2],
                    transform=ax_gallery.transData)
    cbar_kws = dict(label=r'$\Delta T$ (K)')

    ndone = 0
    for ii, x in enumerate(xval[:-1]):
        for jj, y in enumerate(yval[:-1]):
            pts = np.where((modis_tbl[umap_keys[0]] >= x) & (
                modis_tbl[umap_keys[0]] < x+dxv) & (
                modis_tbl[umap_keys[1]] >= y) & (modis_tbl[umap_keys[1]] < y+dxv)
                           & np.isfinite(modis_tbl.LL))[0]
            if len(pts) < min_pts or counts[ii,jj] <= 0.:
                continue

            # Pick a random one
            ichoice = np.random.choice(len(pts), size=1)
            idx = int(pts[ichoice])
            cutout = modis_tbl.iloc[idx]

            # Image
            axins = ax_gallery.inset_axes(
                    [x+0.05*dxv, y+0.05*dyv, 0.9*dxv, 0.9*dyv], 
                    transform=ax_gallery.transData)
            # Load
            try:
                if local:
                    parsed_s3 = urlparse(cutout.pp_file)
                    local_file = os.path.join(os.getenv('SST_OOD'),
                                              'MODIS_L2',
                                              parsed_s3.path[1:])
                    cutout_img = image_utils.grab_image(
                        cutout, close=True, local_file=local_file)
                else:
                    cutout_img = image_utils.grab_image(cutout, close=True)
            except:
                embed(header='598 of plotting')                                                    
            # Cut down?
            if cut_to_inner is not None:
                imsize = cutout_img.shape[0]
                x0, y0 = [imsize//2-cut_to_inner//2]*2
                x1, y1 = [imsize//2+cut_to_inner//2]*2
                cutout_img = cutout_img[x0:x1,y0:y1]
            # Limits
            if in_vmnx[0] == -999:
                DT = cutout.T90 - cutout.T10
                vmnx = (-1*DT, DT)
            elif in_vmnx is not None:
                vmnx = in_vmnx
            else:
                imin, imax = cutout_img.min(), cutout_img.max()
                amax = max(np.abs(imin), np.abs(imax))
                vmnx = (-1*amax, amax)
            # Plot
            sns_ax = sns.heatmap(np.flipud(cutout_img), 
                            xticklabels=[], 
                     vmin=vmnx[0], vmax=vmnx[1],
                     yticklabels=[], cmap=cm, cbar=plt_cbar,
                     cbar_ax=ax_cbar, cbar_kws=cbar_kws,
                     ax=axins)
            sns_ax.set_aspect('equal', 'datalim')
            # Only do this once
            if plt_cbar:
                plt_cbar = False
            ndone += 1
            print(f'ndone= {ndone}, LL={cutout.LL}, npts={len(pts)}')

    # Fonts
    fsz = 15.
    plotting.set_fontsize(ax_gallery, fsz)
    plotting.set_fontsize(ax_regional, fsz)


    # Outline
    #https://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph
    region = (values > 1.5) & np.isfinite(values)
    segments = mk_segments(region, xmax-xmin+dxv, ymax-ymin,
                           x0=xmin, y0=ymin)
    ax_gallery.plot(segments[:, 0], segments[:, 1], color='k', linewidth=2.0,
                    zorder=10)


    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def mk_segments(mapimg,dx,dy,x0=-0.5,y0=-0.5):
    # a vertical line segment is needed, when the pixels next to each other horizontally
    #   belong to diffferent groups (one is part of the mask, the other isn't)
    # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates
    ver_seg = np.where(mapimg[:, 1:] != mapimg[:, :-1])

    # the same is repeated for horizontal segments
    hor_seg = np.where(mapimg[1:, :] != mapimg[:-1, :])

    # if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
    #   (2,7) and (2,8), i.e. from (2,8)..(3,8)
    # in order to draw a discountinuous line, we add Nones in between segments
    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0] + 1))
        l.append((p[1] + 1, p[0] + 1))
        l.append((np.nan, np.nan))

    # and the same for vertical segments
    for p in zip(*ver_seg):
        l.append((p[1] + 1, p[0]))
        l.append((p[1] + 1, p[0] + 1))
        l.append((np.nan, np.nan))

    # now we transform the list into a numpy array of Nx2 shape
    segments = np.array(l)

    # now we need to know something about the image which is shown
    #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
    #   drawn with origin='lower'
    # with this information we can rescale our points
    try:
        segments[:, 0] = x0 + dx * segments[:, 0] / mapimg.shape[1]
        segments[:, 1] = y0 + dy * segments[:, 1] / mapimg.shape[0]
    except:
        embed(header='2346 of figs')

    return segments

#### ########################## #########################
def main(flg_fig):
    if flg_fig == 'all':
        flg_fig = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_fig = int(flg_fig)

    # Uniform, non UMAP gallery for DTall
    if flg_fig & (2 ** 0):
        fig_uniform_gallery('fig_uniform_gallery_DTall.png',
            '96clear_v4_DTall', seed=1235)

    # Uniform, non UMAP gallery for DT1
    if flg_fig & (2 ** 1):
        fig_uniform_gallery('fig_uniform_gallery_DT1.png',
                            '96clear_v4_DT1', seed=1236,
                     in_vmnx=(-1, 1.))

    # Uniform, non UMAP gallery for DT4
    if flg_fig & (2 ** 2):
        fig_uniform_gallery('fig_uniform_gallery_DT4.png', 
                            '96clear_v4_DT4', seed=1236, 
                            in_vmnx=(-2, 2.))

    # All the galleries
    if flg_fig & (2 ** 3):
        for vmnx, table, outfile in zip(
            [(-0.5,0.5), 
             (-0.75,0.75),
             (-1.,1.),
             (-1.5,1.5),
             (-2.,2.),
             (-3.,3.),
             ],
            ['96clear_v4_DT0', 
             '96clear_v4_DT1',
             '96clear_v4_DT15',
             '96clear_v4_DT2',
             '96clear_v4_DT4',
             '96clear_v4_DT5',
             ],
            ['fig_umap_gallery_DT0.png',
             'fig_umap_gallery_DT1.png',
             'fig_umap_gallery_DT15.png',
             'fig_umap_gallery_DT2.png',
             'fig_umap_gallery_DT4.png',
             'fig_umap_gallery_DT5.png',
             ]):
            #if 'DT5' not in outfile:
            #    continue
            fig_ssl_modis.fig_umap_gallery(
                in_vmnx=vmnx, table=table, local=True, outfile=outfile,
                umap_dim=2, umap_comp='S0,S1',
                skip_incidence=True, min_pts=200, cut_to_inner=40, nxy=8)

    # Regional, Pacific ECT
    if flg_fig & (2 ** 4):
        fig_regional_with_gallery(
            'eqpacific',
            'fig_regional_with_gallery_eqpacific.png',
            '96clear_v4_DT1', in_vmnx=(-0.75, 0.75))

    # Regional, Pacific ECT
    if flg_fig & (2 ** 5):
        fig_regional_with_gallery(
            'coastalcali',
            'fig_regional_with_gallery_coastalcali.png',
            '96clear_v4_DT1', in_vmnx=(-0.75, 0.75))


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # Gallery of 16 with DT = all
        #flg_fig += 2 ** 1  # Gallery of 16 with DT = 1
        #flg_fig += 2 ** 2  # Gallery of 16 with DT = 4
        #flg_fig += 2 ** 3  # Full set of UMAP galleries
        #flg_fig += 2 ** 4  # Regional + Gallery -- Pacific ECT
        flg_fig += 2 ** 5  # Regional + Gallery -- Coastal california
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)