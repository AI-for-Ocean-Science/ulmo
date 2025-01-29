""" Figures for LLC analysis in development"""
import os, sys
from typing import IO
import numpy as np
import glob

from urllib.parse import urlparse

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from pandas.core.frame import DataFrame
import xarray

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas
import seaborn as sns

import healpy as hp

from ulmo import plotting
from ulmo.utils import image_utils
from ulmo.utils import utils as utils
from ulmo.analysis import figures as ulmo_figs
from ulmo.llc import io as llc_io
from ulmo.llc import kinematics

from ulmo import io as ulmo_io

from IPython import embed

local_modis_file = '/home/xavier/Projects/Oceanography/AI/OOD/MODIS_L2/Tables/MODIS_L2_std.parquet'


def fig_brazil(outfile='fig_brazil.png'):
    """
    Brazil

    Parameters
    ----------

    Returns
    -------

    """
    # Load LLC
    tbl_test_noise_file = 's3://llc/Tables/test_noise_modis2012.parquet'
    llc_table = ulmo_io.load_main_table(tbl_test_noise_file)

    # Add in DT
    if 'DT' not in llc_table.keys():
        llc_table['DT'] = llc_table.T90 - llc_table.T10

    # Brazil
    in_brazil = ((np.abs(llc_table.lon.values + 57.5) < 10.)  & 
        (np.abs(llc_table.lat.values + 43.0) < 10))
    in_DT = np.abs(llc_table.DT - 2.05) < 0.05
    evals_bz = llc_table[in_brazil & in_DT].copy()
    
    # Rectangles
    #R2 = dict(lon=-60., dlon=1.,
    #    lat=-41.5, dlat=1.5)
    R2 = dict(lon=-61.0, dlon=1.,
        lat=-45., dlat=2.2)
    R1 = dict(lon=-56.5, dlon=1.5,
        lat=-45, dlat=2.2)

    logL = evals_bz.LL.values

    lowLL_val = np.percentile(logL, 10.)
    hiLL_val = np.percentile(logL, 90.)


    # Plot
    fig = plt.figure(figsize=(8, 8))
    plt.clf()
    gs = gridspec.GridSpec(11,11)

    tformP = ccrs.PlateCarree()
    ax_b = plt.subplot(gs[:5, :6], projection=tformP)

    ax_b.text(0.05, 1.03, '(a)', transform=ax_b.transAxes,
              fontsize=15, ha='left', color='k')

    # LL near Argentina!
    psize = 0.5
    cm = plt.get_cmap('coolwarm')
    img = plt.scatter(
        x=evals_bz.lon,
        y=evals_bz.lat,
        s=psize,
        c=evals_bz.LL,
        cmap=cm,
        vmin=lowLL_val, 
        vmax=hiLL_val,
        transform=tformP)
    plt.ylabel('Latitude')
    plt.xlabel('Longitude')
    # Color bar
    cb = plt.colorbar(img, fraction=0.020, pad=0.04)
    cb.ax.set_title('LL', fontsize=11.)

    # Draw rectangles
    for lbl, R, ls in zip(['R1', 'R2'], [R1, R2], ['k-', 'k--']):
        xvals = R['lon']-R['dlon'], R['lon']+R['dlon'], R['lon']+R['dlon'], R['lon']-R['dlon'], R['lon']-R['dlon']
        yvals = R['lat']-R['dlat'], R['lat']-R['dlat'], R['lat']+R['dlat'], R['lat']+R['dlat'], R['lat']-R['dlat']
        ax_b.plot(xvals, yvals, ls, label=lbl)

    gl = ax_b.gridlines(crs=ccrs.PlateCarree(), linewidth=1,
        color='black', alpha=0.5, linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_left = True
    gl.ylabels_right=False
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'black'}# 'weight': 'bold'}
    gl.ylabel_style = {'color': 'black'}# 'weight': 'bold'}
    #gl.xlocator = mticker.FixedLocator([-180., -160, -140, -120, -60, -20.])
    #gl.xlocator = mticker.FixedLocator([-240., -180., -120, -65, -60, -55, 0, 60, 120.])
    #gl.ylocator = mticker.FixedLocator([0., 15., 30., 45, 60.])

    plt.gca().coastlines()

    '''
    # Bathymetry
    df_200 = pandas.read_csv('Patagonian_Bathymetry_200m.txt')
    cut_df200 = (df_200.lat > -50.) & (df_200.lon > -65.) & (df_200.lat < -33.)
    img2 = plt.scatter(
        x=df_200[cut_df200].lon,
        y=df_200[cut_df200].lat,
        s=0.05,
        color='green',
        transform=tformP, label='200m')
    '''


    legend = plt.legend(loc='upper left', scatterpoints=1, borderpad=0.3,
                        handletextpad=0.3, fontsize=11, numpoints=1)

    # ######################################################################33
    # ######################################################################33
    # Histograms
    in_R1, in_R2 = [((np.abs(evals_bz.lon.values - R['lon']) < R['dlon'])  & 
        (np.abs(evals_bz.lat.values - R['lat']) < R['dlat'])) for R in [R1,R2]]
    evals_bz['Subsample'] = 'null'
    evals_bz['Subsample'][in_R1] = 'R1'
    evals_bz['Subsample'][in_R2] = 'R2'

    df_rects = pandas.DataFrame(dict(
        LL=evals_bz.LL.values[in_R1 | in_R2],
        Subsample=evals_bz.Subsample.values[in_R1 | in_R2]))

    ax_h = plt.subplot(gs[:5, 8:])
    ax_h.text(0.05, 1.03, '(b)', transform=ax_h.transAxes,
              fontsize=15, ha='left', color='k')

    sns.histplot(data=df_rects, x='LL',
        hue='Subsample', hue_order=['R1', 'R2'], ax=ax_h)
    ax_h.set_xlim(-800, 500)
    ax_h.set_xlabel('Log Likelihood (LL)')#, fontsize=fsz)
    #plt.ylabel('Probability Density', fontsize=fsz)

    # Gallery
    #nGal = 25
    nGal = 9
    vmin, vmax = None, None
    vmin, vmax = -1, 1
    pal, cm = plotting.load_palette()
    grid_size = 3
    row_off = 5 - grid_size

    # R1
    print("We have {} in R1".format(np.sum(in_R1)))
    idx_R1 = np.where(in_R1)[0]
    rand_R1 = np.random.choice(idx_R1, nGal, replace=False)

    pp_hf = None
    for ss in range(nGal):
        example = evals_bz.iloc[rand_R1[ss]]
        field, pp_hf = llc_io.grab_image(example, close=False, pp_hf=pp_hf) 
        # Axis
        row = grid_size+1 + ss//grid_size + row_off
        col = grid_size+1 + ss % grid_size
        #
        ax_0 = plt.subplot(gs[row, col])
        sns.heatmap(field, ax=ax_0, xticklabels=[], yticklabels=[], cmap=cm,
                    vmin=vmin, vmax=vmax, cbar=False)

    # R2
    print("We have {} in R2".format(np.sum(in_R2)))
    idx_R2 = np.where(in_R2)[0]
    rand_R2 = np.random.choice(idx_R2, nGal, replace=False)

    for ss in range(nGal):
        example = evals_bz.iloc[rand_R2[ss]]
        field, _ = llc_io.grab_image(example, pp_hf=pp_hf, close=False)

        # Axis
        row = grid_size + 1 + ss//grid_size + row_off
        col = ss % grid_size
        #
        ax_0 = plt.subplot(gs[row, col])
        sns.heatmap(field, ax=ax_0, xticklabels=[], yticklabels=[], cmap=cm,
                    vmin=vmin, vmax=vmax, cbar=False)

    # Layout and save
    #plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_brazil_save(nGal=9, outdir='Brazil', seed=1234):
    """
    Save Brazil files locally for convenience

    Parameters
    ----------

    Returns
    -------

    """
    rstate = np.random.RandomState(seed=seed)
    # Load LLC
    tbl_test_noise_file = 's3://llc/Tables/test_noise_modis2012.parquet'
    llc_table = ulmo_io.load_main_table(tbl_test_noise_file)

    # Add in DT
    if 'DT' not in llc_table.keys():
        llc_table['DT'] = llc_table.T90 - llc_table.T10

    # Brazil
    in_brazil = ((np.abs(llc_table.lon.values + 57.5) < 10.)  & 
        (np.abs(llc_table.lat.values + 43.0) < 10))
    in_DT = np.abs(llc_table.DT - 2.05) < 0.05
    evals_bz = llc_table[in_brazil & in_DT].copy()
    
    # Rectangles
    R2 = dict(lon=-61.0, dlon=1., lat=-45., dlat=2.2)
    R1 = dict(lon=-56.5, dlon=1.5, lat=-45, dlat=2.2)

    logL = evals_bz.LL.values

    in_R1, in_R2 = [((np.abs(evals_bz.lon.values - R['lon']) < R['dlon'])  & 
        (np.abs(evals_bz.lat.values - R['lat']) < R['dlat'])) for R in [R1,R2]]
    evals_bz['Subsample'] = 'null'
    evals_bz['Subsample'][in_R1] = 'R1'
    evals_bz['Subsample'][in_R2] = 'R2'

    # R1
    idx_R1 = np.where(in_R1)[0]
    idx_R2 = np.where(in_R2)[0]

    pp_hf = None
    sv_lbls = []
    sv_idx = []
    for smpl, idx in zip(['R1', 'R2'],
                          [idx_R1, idx_R2]):
        rand = rstate.choice(idx, nGal, replace=False)
        for ss in range(nGal):
            example = evals_bz.iloc[rand[ss]]
            # Save
            sv_idx.append(rand[ss])
            sv_lbls.append(smpl+'_{}'.format(str(ss).zfill(3)))
            # Load velocity
            U, V, SST, Salt = llc_io.grab_velocity(example, add_SST=True,
                                             add_Salt=True)
            # Generate ds
            ds = xarray.Dataset({'U': U, 'V': V, 'Theta': SST,
                                 'Salt': Salt})
            # Outfile
            outfile = os.path.join(outdir, '{}.nc'.format(sv_lbls[-1]))
            ds.to_netcdf(outfile)
            print("Wrote: {}".format(outfile))
    
    # Write a table too
    df = pandas.DataFrame(dict(file=sv_lbls, idx=sv_idx))
    df.to_csv(os.path.join(outdir, 'images.csv'))
    evals_bz.to_csv(os.path.join(outdir, 'brazil.csv'))

def load_brazil(nGal=9, indir='Brazil', use_files=True):
    """ Method to load up images from the Brazil-Malvanis Channel

    Args:
        nGal (int, optional): _description_. Defaults to 9.
        indir (str, optional): _description_. Defaults to 'Brazil'.
        use_files (bool, optional): _description_. Defaults to True.

    Returns:
        tuple: show_R1, show_R2, R1_dict, R2_dict
    """
    # Load LLC
    tbl_test_noise_file = 's3://llc/Tables/test_noise_modis2012.parquet'
    llc_table = ulmo_io.load_main_table(tbl_test_noise_file)

    # Add in DT
    if 'DT' not in llc_table.keys():
        llc_table['DT'] = llc_table.T90 - llc_table.T10

    # Brazil
    in_brazil = ((np.abs(llc_table.lon.values + 57.5) < 10.)  & 
        (np.abs(llc_table.lat.values + 43.0) < 10))
    in_DT = np.abs(llc_table.DT - 2.05) < 0.05
    evals_bz = llc_table[in_brazil & in_DT].copy()
    
    # Rectangles
    #R2 = dict(lon=-60., dlon=1.,
    #    lat=-41.5, dlat=1.5)
    R2 = dict(lon=-61.0, dlon=1.,
        lat=-45., dlat=2.2)
    R1 = dict(lon=-56.5, dlon=1.5,  # Dynamic region
        lat=-45, dlat=2.2)

    logL = evals_bz.LL.values

    lowLL_val = np.percentile(logL, 10.)
    hiLL_val = np.percentile(logL, 90.)

    in_R1, in_R2 = [((np.abs(evals_bz.lon.values - R['lon']) < R['dlon'])  & 
        (np.abs(evals_bz.lat.values - R['lat']) < R['dlat'])) for R in [R1,R2]]
    evals_bz['Subsample'] = 'null'
    evals_bz['Subsample'][in_R1] = 'R1'
    evals_bz['Subsample'][in_R2] = 'R2'

    # Load up input files?
    if use_files:
        keys = ['U', 'V', 'Theta', 'Salt']
        R1_dict, R2_dict = {}, {}
        for smpl, R_dict in zip(['R1', 'R2'],
                                [R1_dict, R2_dict]):
            # Init
            for key in keys:
                R_dict[key] = []
            for ss in range(nGal):
                root = smpl+'_{}.nc'.format(str(ss).zfill(3))
                infile = os.path.join(indir, root)
                ds = xarray.open_dataset(infile)
                for key in keys:
                    R_dict[key].append(ds[key])
        show_R1 = np.arange(nGal)
        show_R2 = np.arange(nGal)
    else:                
        # R1
        print("We have {} in R1".format(np.sum(in_R1)))
        idx_R1 = np.where(in_R1)[0]
        show_R1 = np.random.choice(idx_R1, nGal, replace=False)
        # R2
        print("We have {} in R2".format(np.sum(in_R2)))
        idx_R2 = np.where(in_R2)[0]
        show_R2 = np.random.choice(idx_R2, nGal, replace=False)
    # Return
    return show_R1, show_R2, R1_dict, R2_dict


def fig_brazil_kin_distrib(mode, nGal=9, indir='Brazil', use_files=True):
    # Load up
    show_R1, show_R2, R1_dict, R2_dict = load_brazil(
        nGal=nGal, indir=indir, use_files=use_files)
    if mode == 'full':
        outfile='fig_brazil_kin_distrib_full.png'
    elif mode in ['mean', 'median', 'std']:
        outfile='fig_brazil_kin_distrib_{}.png'.format(mode)
    else:
        raise IOError("Bad mode!")

    # Calc it all
    kin_dict = {}
    kin_dict['R1'] = dict(rel_vort=[], okubo=[], strain=[], div=[])
    kin_dict['R2'] = dict(rel_vort=[], okubo=[], strain=[], div=[])

    for smpl, idx, R_dict in zip(
            ['R1', 'R2'], [show_R1, show_R2], [R1_dict, R2_dict]):
        # Loop on images
        for ss in range(nGal):
            U = R_dict['U'][idx[ss]]
            V = R_dict['V'][idx[ss]]
            for kin in kin_dict[smpl].keys():
                if kin == 'rel_vort':
                    stat = kinematics.calc_curl(U.data, V.data)
                elif kin == 'okubo':
                    stat = kinematics.calc_okubo_weiss(U.data, V.data)
                elif kin == 'strain':
                    stat = kinematics.calc_lateral_strain_rate(U.data, V.data)
                elif kin == 'div':
                    stat = kinematics.calc_div(U.data, V.data)
                else:
                    raise ValueError("Bad kin!!")
                # Save it
                if mode == 'full':
                    kin_dict[smpl][kin] += stat.flatten().tolist()
                elif mode == 'mean':
                    kin_dict[smpl][kin] += [np.mean(stat)]
                elif mode == 'median':
                    kin_dict[smpl][kin] += [np.median(stat)]
                elif mode == 'std':
                    kin_dict[smpl][kin] += [np.std(stat)]

    # Build the Table
    R1_tbl = pandas.DataFrame(kin_dict['R1'])
    R1_tbl['Sample'] = 'R1'
    R2_tbl = pandas.DataFrame(kin_dict['R2'])
    R2_tbl['Sample'] = 'R2'
    stat_tbl = pandas.concat([R1_tbl, R2_tbl])

    # Figure time 
    fig = plt.figure(figsize=(12, 6))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    for ss, kin in enumerate(kin_dict['R1'].keys()):
        ax = plt.subplot(gs[ss])
        sns.histplot(data=stat_tbl, x=kin, hue='Sample', ax=ax)
        # Limits
        if kin != 'strain' and mode == 'full':
            std = np.std(stat_tbl[stat_tbl.Sample == 'R1'][kin])
            ax.set_xlim(-2.5*std, 2.5*std)
        # Label
        if ss == 0 and not (mode == 'full'):
            ax.text(0.05, 1.03, mode, transform=ax.transAxes,
              fontsize=15, ha='left', color='k')

    # Layout and save
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
    plt.savefig(outfile, dpi=400)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_brazil_kin_imgs(outroot='fig_brazil_',
                        nGal=9, indir='Brazil', use_files=True):
    """
    Brazil

    Parameters
    ----------

    Returns
    -------

    """
    # Load up
    show_R1, show_R2, R1_dict, R2_dict = load_brazil(
        nGal=nGal, indir=indir, use_files=use_files)

    # Gallery
    grid_size = 3
    row_off = 0
    pal, cm = plotting.load_palette()

    def mk_figure(metric):
        outfile = outroot+'{}.png'.format(metric)
        fig = plt.figure(figsize=(12, 6))
        plt.clf()
        gs = gridspec.GridSpec(3,6)

        for coff, smpl, idx, R_dict in zip(
            [grid_size,0],
            ['R1', 'R2'],
            [show_R1, show_R2],
            [R1_dict, R2_dict]):
            for ss in range(nGal):
                U = R_dict['U'][idx[ss]]
                V = R_dict['V'][idx[ss]]
                SST = R_dict['Theta'][idx[ss]]
                Salt = R_dict['Salt'][idx[ss]]
                # Axis
                row = ss//grid_size + row_off
                col = coff + ss % grid_size
                #
                ax = plt.subplot(gs[row, col])
                # 
                if metric == 'vel':
                    ax.quiver(U, V, color='b')
                elif metric == 'SST':
                    sns.heatmap(np.flipud(SST.data - np.mean(SST.data)), 
                                ax=ax, cmap=cm,
                                vmin=-1, vmax=1, cbar=False)
                elif metric == 'div':
                    div = kinematics.calc_div(U.data, V.data)
                    sns.heatmap(np.flipud(div), ax=ax, cmap='seismic',
                                vmin=-0.2, vmax=0.2, cbar=False)
                elif metric == 'curl':  # aka relative vorticity
                    curl = kinematics.calc_curl(U.data, V.data)
                    sns.heatmap(np.flipud(curl), ax=ax, cmap='seismic',
                                vmin=-0.2, vmax=0.2, cbar=False)
                elif metric == 'okubo':  # aka relative vorticity
                    okubo = kinematics.calc_okubo_weiss(U.data, V.data)
                    sns.heatmap(np.flipud(okubo), ax=ax, cmap='seismic',
                        cbar=False, vmin=-0.02, vmax=0.02)
                elif metric == 'strain_rate':  
                    strain = kinematics.calc_lateral_strain_rate(U.data, V.data)
                    sns.heatmap(np.flipud(strain), ax=ax, cmap='Blues',
                        cbar=False, vmin=0., vmax=0.1)
                elif metric == 'F_s':  
                    F_s = kinematics.calc_F_s(U.data, V.data, 
                                                 SST.data, Salt.data)
                    sns.heatmap(np.flipud(F_s), ax=ax, cmap='seismic',
                        cbar=False, vmin=-0.003, vmax=0.003)
                else: 
                    raise IOError("Bad choice")
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])

        # Layout and save
        plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
        plt.savefig(outfile, dpi=400)
        plt.close()
        print('Wrote {:s}'.format(outfile))

    # Frontogenesis
    mk_figure('F_s')
    # Strain rate
    mk_figure('strain_rate')
    # Okubo
    mk_figure('okubo')
    # SST
    mk_figure('SST')
    # Velocity
    mk_figure('vel')
    # Divergence
    mk_figure('div')
    # Curl
    mk_figure('curl')

def fig_outlier_distribution(outfile='fig_outlier_distribution.png'): 
    # Load LLC
    tbl_test_noise_file = 's3://llc/Tables/test_noise_modis2012.parquet'
    llc_table = ulmo_io.load_main_table(tbl_test_noise_file)
    low_LL = llc_table.LL < -1000.

    # Spatial plot
    ax = ulmo_figs.show_spatial(llc_table[low_LL], lbl='low LL',
                               show=False)
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()

def fig_LL_distribution(outfile, LL_source='LLC', nside=64, 
                        func='mean', vmnx=None,
                        min_sample=5, normalize=False):
    """Spatial maps of LL metrics

        outfile ([type]): [description]
        nside (int, optional): [description]. Defaults to 64.
        func (str, optional): [description]. Defaults to 'mean'.
            diff_mean -- Subtract <MODIS> - <LLC>
    """
    # Inputs
    if vmnx is None:
        vmnx = (-2000., None)
    # Load Table
    table_file = 's3://llc/Tables/test_noise_modis2012.parquet'
    ulmo_table = ulmo_io.load_main_table(table_file)

    # Do the analysis

    if normalize:
        mean_LL_MODIS = np.mean(ulmo_table.modis_LL.values)
        std_LL_MODIS = np.std(ulmo_table.modis_LL.values)
        mean_LL_LLC = np.mean(ulmo_table.LL.values)
        std_LL_LLC = np.std(ulmo_table.LL.values)
        # Scale the LLC
        Nstd = (ulmo_table.LL.values-mean_LL_LLC)/std_LL_LLC
        ulmo_table['new_LL_LLC'] = mean_LL_MODIS + Nstd*std_LL_MODIS
        # Check
        if False:
            df = pandas.concat(axis=0, ignore_index=True,
                        objs=[
                            pandas.DataFrame.from_dict(dict(LL=ulmo_table.new_LL_LLC,
                                                            Data='New LLC')),
                            pandas.DataFrame.from_dict(dict(LL=ulmo_table.modis_LL,
                                                            Data='MODIS (2012)')),
                        ]
                        )
            ax = sns.histplot(data=df, x='LL', hue='Data')
            embed(header='276 of figs')

    # Grab lats, lons
    lats = ulmo_table.lat.values
    lons = ulmo_table.lon.values

    # Healpix coords
    theta = (90 - lats) * np.pi / 180.
    phi = lons * np.pi / 180.
    hp_idx = hp.pixelfunc.ang2pix(nside, theta, phi)

    # Count events
    npix_hp = hp.nside2npix(nside)

    LL_stat = np.zeros(npix_hp)
    LL_rms = np.zeros(npix_hp)

    uni_idx = np.unique(hp_idx)
    # Looping
    print("Looping...")
    for idx in uni_idx:
        in_hp = idx == hp_idx
        if np.sum(in_hp) < min_sample:
            continue
        # Grab data 
        if normalize:
            LL_LLC = ulmo_table[in_hp].new_LL_LLC.values
        else:
            LL_LLC = ulmo_table[in_hp].LL.values
        LL_MODIS = ulmo_table[in_hp].modis_LL.values
        # Proceed
        if LL_source == 'LLC':
            LL = LL_LLC
        elif LL_source == 'MODIS':
            LL = LL_MODIS
        # Stats
        if func == 'mean':
            LL_stat[idx] = np.mean(LL)
        elif func == 'median':
            LL_stat[idx] = np.median(LL)
        elif func == 'diff_mean':
            LL_stat[idx] = np.mean(LL_MODIS) - np.mean(LL_LLC) 
        else:
            raise IOError("Bad function")
        LL_rms[idx] = np.std(LL)

    # Angles
    hp_lons, hp_lats = hp.pixelfunc.pix2ang(nside, np.arange(npix_hp), lonlat=True)

    # Nan me
    good = LL_stat != 0.

    # Figure time
    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    ax = plt.axes(projection=tformP)

    cm = plt.get_cmap('seismic')
    #img = ax.tricontourf(hp_lons, hp_lats, LL_stat, 
    #                     transform=tformM,
    #                     levels=20, cmap=cm)#, zorder=10)
    img = plt.scatter(x=hp_lons[good],
            y=hp_lats[good],
            c=LL_stat[good],
            cmap=cm,
            s=1,
            vmin=vmnx[0],
            vmax=vmnx[1],
            transform=tformP)

    # Colorbar
    cb = plt.colorbar(img, orientation='horizontal', pad=0.)
    clbl=f'{func} LL'
    cb.set_label(clbl, fontsize=20.)
    cb.ax.tick_params(labelsize=17)

    # Coast lines
    ax.coastlines(zorder=10)
    ax.set_global()

    # Layout and save
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_LLC_vs_MODIS(outfile='fig_LLC_vs_MODIS.png'): 

    # Load LLC
    tbl_test_noise_file = 's3://llc/Tables/test_noise_modis2012.parquet'
    llc_table = ulmo_io.load_main_table(tbl_test_noise_file)

    # Load MODIS
    #modisl2_table = ulmo_io.load_main_table(local_modis_file)

    # Dummy table for plotting
    df = pandas.concat(axis=0, ignore_index=True,
                       objs=[
                           pandas.DataFrame.from_dict(dict(LL=llc_table.LL,
                                                            Data='LLC')),
                           pandas.DataFrame.from_dict(dict(LL=llc_table.modis_LL,
                                                            Data='MODIS (2012)')),
                       ]
                       )

    palette = sns.color_palette(["#4c72b0","#55a868"])

    fig = plt.figure(figsize=(12, 8))
    plt.clf()

    ax = sns.histplot(data=df, x='LL', hue='Data', palette=palette)
    #_ = sns.histplot(data=llc_table, x='modis_LL', ax=ax, color='g')
    ax.set_xlim(-3000., 1500.)
    ax.set_ylim(0., 10000.)

    #ax.legend()
    # Fonts
    set_fontsize(ax, 19.)

    # Layout and save
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_umap_gallery(outfile='fig_umap_gallery.png',
                     version=1, local=True, restrict_DT=False): 
    if version == 1:                    
        tbl_file = 's3://llc/Tables/LLC_MODIS2012_SSL_v1.parquet'
    if local:
        parsed_s3 = urlparse(tbl_file)
        tbl_file = os.path.basename(parsed_s3.path[1:])
    # Load
    llc_tbl = ulmo_io.load_main_table(tbl_file)

    # Restrict on DT?
    if restrict_DT:
        llc_tbl['DT'] = llc_tbl.T90 - llc_tbl.T10
        llc_tbl = llc_tbl[ll]
    ax = plotting.umap_gallery(llc_tbl, outfile=outfile)



def set_fontsize(ax,fsz):
    '''
    Generate a Table of columns and so on
    Restrict to those systems where flg_clm > 0

    Parameters
    ----------
    ax : Matplotlib ax class
    fsz : float
      Font size
    '''
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsz)


#### ########################## #########################
def main(flg_fig):
    if flg_fig == 'all':
        flg_fig = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_fig = int(flg_fig)

    # LL distributions
    if flg_fig & (2 ** 0):
        fig_LLC_vs_MODIS()

    # Outlier locations
    if flg_fig & (2 ** 1):
        fig_outlier_distribution()

    # Brazil
    if flg_fig & (2 ** 2):
        fig_brazil()

    # LL spatial metrics
    if flg_fig & (2 ** 3):
        #fig_LL_distribution('fig_LL_spatial_LLC.png', LL_source='LLC')
        #fig_LL_distribution('fig_LL_spatial_MODIS.png', LL_source='MODIS')
        #fig_LL_distribution('fig_LL_spatial_diff.png', 
        #                    func='diff_mean', vmnx=(-1000., 1000))
        fig_LL_distribution('fig_LL_spatial_diff_norm.png', 
                            func='diff_mean', vmnx=(-1000., 1000),
                            normalize=True)

    # Brazil velocity
    if flg_fig & (2 ** 4):
        fig_brazil_kin_imgs(use_files=True)

    # Brazil kinematic distributions
    if flg_fig & (2 ** 5):
        #fig_brazil_kin_distrib('full', use_files=True)
        fig_brazil_kin_distrib('mean', use_files=True)
        fig_brazil_kin_distrib('median', use_files=True)
        fig_brazil_kin_distrib('std', use_files=True)

    # UMAP gallery
    if flg_fig & (2 ** 6):
        fig_umap_gallery()

    # Save Brazil data to disk
    if flg_fig & (2 ** 7):
        fig_brazil_save()


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # LL for LLC vs. MODIS (matched on 2012)
        #flg_fig += 2 ** 1  # Outlier distribution (2012 matched)
        #flg_fig += 2 ** 2  # Brazil
        #flg_fig += 2 ** 3  # Spatial LL metrics
        #flg_fig += 2 ** 4  # Brazil kinematic images
        #flg_fig += 2 ** 5  # Brazil kinematic distributions
        #flg_fig += 2 ** 6  # UMAP SSL gallery
        flg_fig += 2 ** 7  # Generate Brazil cutouts of ocean model data
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)

