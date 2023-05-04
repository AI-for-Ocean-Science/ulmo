""" Figures for MAE paper """
import os, sys
import numpy as np
import scipy
from pkg_resources import resource_filename

import healpy as hp

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs
import cartopy

mpl.rcParams['font.family'] = 'stixgeneral'

import pandas
import seaborn as sns

import h5py

from ulmo import plotting
from ulmo.mae import mae_utils
from ulmo import io as ulmo_io
from ulmo.utils import image_utils
try:
    from ulmo.mae import models_mae
except ModuleNotFoundError:
    print("Not able to load the models")
else:    
    from ulmo.mae import reconstruct
from ulmo.mae import plotting as mae_plotting


from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import anly_patches
#sys.path.append(os.path.abspath("../Figures/py"))
#import fig_ssl_modis

# Globals

#preproc_path = os.path.join(os.getenv('OS_AI'), 'MAE', 'PreProc')
#recon_path = os.path.join(os.getenv('OS_AI'), 'MAE', 'Recon')
#orig_file = os.path.join(preproc_path, 'MAE_LLC_valid_nonoise_preproc.h5')

sst_path = os.getenv('OS_SST')
ogcm_path = os.getenv('OS_OGCM')
enki_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki')

def fig_clouds(outfile:str, analy_file:str,
                 local=False, 
                 debug=False, 
                 color='bwr', vmax=None): 
    """ Global geographic plot of the UMAP select range

    Args:
        outfile (str): 
        table (str): 
            Which table to use
        umap_rngs (list): _description_
        local (bool, optional): _description_. Defaults to False.
        nside (int, optional): _description_. Defaults to 64.
        umap_comp (str, optional): _description_. Defaults to 'S0,S1'.
        umap_dim (int, optional): _description_. Defaults to 2.
        debug (bool, optional): _description_. Defaults to False.
        color (str, optional): _description_. Defaults to 'bwr'.
        vmax (_type_, optional): _description_. Defaults to None.
        min_counts (int, optional): Minimum to show in plot.
        show_regions (str, optional): Rectangles for the geographic regions of this 
            Defaults to False.
        absolute (bool, optional):
            If True, show absolute counts instead of relative
    """

    # Load
    data = np.load(analy_file)
    nside = int(data['nside'])

    # Angles
    npix_hp = hp.nside2npix(nside)
    hp_lons, hp_lats = hp.pixelfunc.pix2ang(nside, np.arange(npix_hp), 
                                            lonlat=True)

    rlbl = r"$\log_{10} \; \rm Counts$"
    vmax = None
    color = 'Blues'


   # Figure
    fig = plt.figure(figsize=(12,8))
    plt.clf()
    gs = gridspec.GridSpec(2,2)

    tformM = ccrs.Mollweide()
    tformP = ccrs.PlateCarree()

    CC_plt = [0.05, 0.1, 0.2, 0.5]
    #CC_values = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 
    #             0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    for kk, CC in enumerate(CC_plt):
        mt = np.where(np.isclose(data['CC_values'],CC))[0][0]
        #mt = np.where(np.isclose(CC_values,CC))[0][0]
        ax = plt.subplot(gs[kk], projection=tformM)

        hp_events = np.ma.masked_array(data['hp_pix_CC'][:,mt])
        # Mask
        hp_events.mask = [False]*hp_events.size
        bad = hp_events <= 0
        hp_events.mask[bad] = True
        hp_events.data[bad] = 0

        # Proceed
        hp_plot = np.log10(hp_events)

        cm = plt.get_cmap(color)
        # Cut
        good = np.invert(hp_plot.mask)
        img = plt.scatter(x=hp_lons[good],
            y=hp_lats[good],
            c=hp_plot[good], 
            cmap=cm,
            vmin=0.,
            vmax=vmax, 
            s=1,
            transform=tformP)

        # Colorbar
        cb = plt.colorbar(img, orientation='horizontal', pad=0.)
        lbl = rlbl + f'  (CC={CC:.2f})'
        cb.set_label(lbl, fontsize=15.)
        cb.ax.tick_params(labelsize=17)

        # Coast lines
        ax.coastlines(zorder=10)
        ax.add_feature(cartopy.feature.LAND, 
            facecolor='gray', edgecolor='black')
        ax.set_global()

        gl = ax.gridlines(crs=tformP, linewidth=1, 
            color='black', alpha=0.5, linestyle=':', draw_labels=True)
        gl.xlabels_top = False
        gl.ylabels_left = True
        gl.ylabels_right=False
        gl.xlines = True
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'color': 'black'}# 'weight': 'bold'}
        gl.ylabel_style = {'color': 'black'}# 'weight': 'bold'}

        plotting.set_fontsize(ax, 19.)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_numhp_clouds(outfile:str, analy_file:str):

    # Load
    data = np.load(analy_file)
    nside = int(data['nside'])
    CC_values = data['CC_values']

    N_mins = [10, 30, 100, 300]
    # Figure
    fig = plt.figure(figsize=(12,8))
    plt.clf()

    ax = plt.gca()

    for N_min in N_mins:
        num_hp = []
        for kk in range(CC_values.size):
            gd = np.sum(data['hp_pix_CC'][:,kk] >= N_min)
            num_hp.append(gd)
        # Plot
        ax.plot(CC_values, num_hp, label=f'N_min={N_min}')

    ax.legend(fontsize=15.)
    ax.set_xlabel('Cloud Cover')
    ax.set_ylabel('Number')
    plotting.set_fontsize(ax, 17.)
    ax.set_yscale('log')
    ax.set_ylim(1., 7e4)

    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_bias(outfile:str):

    # Bias file
    bias_file = os.path.join(
        resource_filename('ulmo', 'runs'),
        'MAE', 'enki_bias_LLC.csv')
    bias = pandas.read_csv(bias_file)

    # Figure
    fig = plt.figure(figsize=(10,8))
    plt.clf()

    ax = plt.gca()

    for t in np.unique(bias.t.values):
        all_t = bias.t == t
        # Plot
        ax.plot(bias[all_t].p, bias[all_t]['median'], 
                label=f't={t}')

    ax.legend(fontsize=19.)
    ax.set_xlabel('p (%)')
    ax.set_ylabel('Median Bias')
    plotting.set_fontsize(ax, 21.)
    #ax.set_yscale('log')
    #ax.set_ylim(1., 7e4)

    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_patch_ij_binned_stats(metric:str,
    stat:str, patch_file:str, nbins:int=16):

    t_per, p_per = mae_utils.parse_mae_img_file(patch_file)

    # Outfile
    outfile = f'fig_{metric}_{stat}_t{t_per}_p{p_per}_patch_ij_binned_stats.png'
    # Load
    patch_file = os.path.join(os.getenv("OS_DATA"),
                              'MAE', 'Recon', patch_file)
    f = np.load(patch_file)
    data = f['data']
    data = data.reshape((data.shape[0]*data.shape[1], 
                         data.shape[2]))

    items = f['items']
    tbl = pandas.DataFrame(data, columns=items)

    #metric = 'abs_median_diff'
    #metric = 'median_diff'
    #metric = 'std_diff'
    #stat = 'median'
    #stat = 'mean'
    #stat = 'std'

    values, lbl = anly_patches.parse_metric(
        tbl, metric)

    # Do it
    median, x_edge, y_edge, ibins = scipy.stats.binned_statistic_2d(
    tbl.i_patch, tbl.j_patch, values,
        statistic=stat, expand_binnumbers=True, 
        bins=[nbins,nbins])

    # Figure
    fig = plt.figure(figsize=(10,8))
    plt.clf()
    ax = plt.gca()

    cmap = 'Blues'
    cm = plt.get_cmap(cmap)
    mplt = ax.pcolormesh(x_edge, y_edge, 
                    median.transpose(),
                    cmap=cm, 
                    vmax=None) 
    # Color bar
    cbaxes = plt.colorbar(mplt, pad=0., fraction=0.030)
    cbaxes.set_label(f'{stat}({lbl})', fontsize=17.)
    cbaxes.ax.tick_params(labelsize=15)

    # Axes
    ax.set_xlabel(r'i')
    ax.set_ylabel(r'j')
    ax.set_aspect('equal')

    plotting.set_fontsize(ax, 15)


    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_explore_bias(outfile:str='fig_explore_bias.png',
                     nimg:int=50000, debug:bool=False,
                     bias_file:str='bias.csv',
                     clobber:bool=False):

    if os.path.isfile(bias_file) and not clobber:
        df = pandas.read_csv(bias_file)
        print(f"Loaded: {bias_file}")
    else:
        # Load
        f_orig = h5py.File(orig_file, 'r')
        orig_img = f_orig['valid'][0:nimg,0,...]

        # Analyze
        result_dict = dict(t=[], p=[], median_bias=[], mean_bias=[])
        for t in [10, 35, 75]:
            if debug and t > 35:
                break
            for p in [10, 20, 30, 40, 50]:
                if debug and p > 30:
                    break
                print(f'Working on t={t} p={p}')
                #
                recon_file = mae_utils.img_filename(t, p, mae_img_path=recon_path)
                mask_file = mae_utils.mask_filename(t, p, mae_mask_path=recon_path)
                # Load
                f_recon = h5py.File(recon_file, 'r')
                f_mask = h5py.File(mask_file, 'r')

                # Load
                recon_img = f_recon['valid'][0:nimg,0,...]
                mask_img = f_mask['valid'][0:nimg,0,...].astype(int)

                # Do it
                diff_true = recon_img - orig_img 

                patches = mask_img == 1

                median_bias = np.median(diff_true[patches])
                mean_bias = np.mean(diff_true[patches])
                #mean_img = np.mean(orig_img[np.isclose(mask_img,0.)])

                # Save
                result_dict['t'].append(t)
                result_dict['p'].append(p)
                result_dict['median_bias'].append(median_bias)
                result_dict['mean_bias'].append(mean_bias)

        # Write
        df = pandas.DataFrame(result_dict)
        df.to_csv(bias_file, index=False)
        print(f'Wrote: {bias_file}')


    # Figure
    sns.set_style("whitegrid")

    fig = plt.figure(figsize=(10,8))
    plt.clf()

    # Plot em
    ax = sns.scatterplot(data=df, x='p', y='median_bias', hue='t',
                         palette='deep', 
                         s=100, markers='o')
                         #size='p', sizes=(100, 1000))

    # Axes
    ax.set_xlabel('Patch Fraction')
    #ax.set_ylabel(r'j')
    #ax.set_aspect('equal')

    plotting.set_fontsize(ax, 15)

    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_viirs_example(outfile:str, t:int, idx:int=0): 
    """ Show an example of VIIRS reconstruction

    Args:
        outfile (str): _description_
        t (int): _description_
        idx (int, optional): _description_. Defaults to 0.
    """


    # VIIRS table
    viirs_file = os.path.join(sst_path, 'VIIRS', 'Tables', 
                              'VIIRS_all_98clear_std.parquet')
    viirs = ulmo_io.load_main_table(viirs_file)
    all_clear = np.isclose(viirs.clear_fraction, 0.)
    clear_viirs = viirs[all_clear].copy()
    
    # Grab a clear image
    pp_file = os.path.join(sst_path, 'VIIRS', 'PreProc', 'VIIRS_2012_95clear_192x192_preproc_viirs_std.h5')
    f_pp2012 = h5py.File(pp_file, 'r')
    cutout = clear_viirs.iloc[0]
    img, _ = image_utils.grab_image(cutout, pp_hf=f_pp2012, close=False)

    # Load model
    chkpt_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'MAE', 'models', f'mae_t{t}_399.pth')
    model = models_mae.prepare_model(chkpt_file)

    # Reconstruct
    rimg = np.resize(img, (64,64,1))
    mask, x, y = reconstruct_LLC.run_one_image(rimg, model, 0.3)

    # Numpy me
    np_mask = mask.detach().cpu().numpy().reshape(64,64)
    np_x = x.detach().cpu().numpy().reshape(64,64)
    np_y = y.detach().cpu().numpy().reshape(64,64)

    # Prep fig
    fig = plt.figure(figsize=(7, 3.1))
    plt.clf()
    gs = gridspec.GridSpec(1,3)

    mae_plotting.plot_recon(np_y, np_x, np_mask, gs=gs,
                            res_vmnx=(-0.2,0.2))

    # Finish
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_llc_inpainting(outfile:str, t:int, p:int, 
                       debug:bool=False):

    # Files
    local_enki_table = os.path.join(
        enki_path, 'Tables', 
        'MAE_LLC_valid_nonoise.parquet')
    local_mae_valid_nonoise_file = os.path.join(
        enki_path, 'PreProc', 
        'MAE_LLC_valid_nonoise_preproc.h5')
    local_orig_file = local_mae_valid_nonoise_file
    inpaint_file = os.path.join(
        ogcm_path, 'LLC', 'Enki', 
        'Recon', f'LLC_inpaint_t{t}_p{p}.h5')
    recon_file = mae_utils.img_filename(t,p, local=True)
    mask_file = mae_utils.mask_filename(t,p, local=True)

    # Load up
    enki_tbl = ulmo_io.load_main_table(local_enki_table)
    f_orig = h5py.File(local_orig_file, 'r')
    f_recon = h5py.File(recon_file, 'r')
    f_inpaint = h5py.File(inpaint_file, 'r')
    f_mask = h5py.File(mask_file, 'r')


    # Grab the images
    if debug:
        nimgs = 1000
    else:
        nimgs = 50000
    orig_imgs = f_orig['valid'][:nimgs,0,...]
    mask_imgs = f_mask['valid'][:nimgs,0,...]

    # Allow for various shapes (hack)
    recon_imgs = f_recon['valid'][:nimgs,0,...]


    rms_enki = rms_images(orig_imgs, recon_imgs, mask_imgs)
    del recon_imgs
    
    inpaint_imgs = f_inpaint['inpainted'][:nimgs,...]
    rms_inpaint = rms_images(orig_imgs, inpaint_imgs, mask_imgs)

    # Cut and add table
    cut = (enki_tbl.pp_idx >= 0) & (enki_tbl.pp_idx < nimgs)
    enki_tbl = enki_tbl[cut].copy()
    enki_tbl.sort_values(by=['pp_idx'], inplace=True)

    enki_tbl['rms_enki'] = rms_enki
    enki_tbl['rms_inpaint'] = rms_inpaint
    enki_tbl['delta_rms'] = rms_inpaint - rms_enki
    enki_tbl['log10DT'] = np.log10(enki_tbl.DT)
    enki_tbl['frac_rms'] = enki_tbl.delta_rms / enki_tbl.DT

    nbad = np.sum(enki_tbl.delta_rms < 0.)
    print(f"There are {100*nbad/len(enki_tbl)}% with DeltaRMS < 0")

    # Plot
    # Prep fig
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2,2)
    plt.clf()

    ax0 = plt.subplot(gs[1])
    _ = sns.histplot(data=enki_tbl, x='DT',
                    y='delta_rms', log_scale=(True,True),
                    color='purple', ax=ax0)
    ax0.set_xlabel(r'$\Delta T$ (K)')                
    ax0.set_ylabel(r'$\Delta$RMSE = RMSE$_{\rm biharmonic}$ - RMSE$_{\rm Enki}$ (K)')

    # Delta RMS / Delta T
    ax1 = plt.subplot(gs[2])
    sns.histplot(data=enki_tbl, x='DT',
                 y='frac_rms', log_scale=(True,False),
                 color='gray', ax=ax1) 
    ax1.set_xlabel(r'$\Delta T$ (K)')                
    ax1.set_ylabel(r'$\Delta$RMSE / $\Delta T$')
    ax1.set_ylim(-0.1, 1)

    # RMS_biharmonic vs. RMS_Enki
    ax2 = plt.subplot(gs[3])
    scat = ax2.scatter(enki_tbl.rms_enki, 
                enki_tbl.rms_inpaint, s=0.1,
                c=enki_tbl.log10DT, cmap='jet')
    ax2.set_ylabel(r'RMSE$_{\rm biharmonic}$')
    ax2.set_xlabel(r'RMSE$_{\rm Enki}$')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    cbaxes = plt.colorbar(scat)#, pad=0., fraction=0.030)
    cbaxes.set_label(r'$\log_{10} \, \Delta T$ (K)')#, fontsize=17.)
    #cbaxes.ax.tick_params(labelsize=15)

    ax2.plot([1e-3, 10], [1e-3,10], 'k--')

    # RMS_Enki vs. DT
    nobj = len(enki_tbl)
    hack = pandas.concat([enki_tbl,enki_tbl])
    hack['Model'] = ['Enki']*nobj + ['Biharmonic']*nobj
    hack['RMSE'] = np.concatenate(
        [enki_tbl.rms_enki.values[0:nobj],
        enki_tbl.rms_inpaint.values[0:nobj]])

    ax3 = plt.subplot(gs[0])
    sns.histplot(data=hack, x='DT',
                 y='RMSE', 
                 hue='Model',
                 log_scale=(True,True),
                 ax=ax3) 
    #sns.histplot(data=enki_tbl, x='DT',
    #             y='rms_enki', 
    #             log_scale=(True,True),
    #             color='blue', ax=ax3) 
    ax3.set_xlabel(r'$\Delta T$ (K)')                
    #ax3.set_ylabel(r'RMSE$_{\rm Enki}$ (K)')

    # Polish
    #fg.ax.minorticks_on()
    for ax in [ax0, ax1, ax2, ax3]:
        plotting.set_fontsize(ax, 14.)

    #plt.title(f'Enki vs. Inpaiting: t={t}, p={p}')

    # Finish
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

#### ########################## #########################
def main(flg_fig):
    if flg_fig == 'all':
        flg_fig = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_fig = int(flg_fig)

    # Uniform, non UMAP gallery for DTall
    if flg_fig & (2 ** 0):
        fig_clouds('fig_clouds.png', 'modis_2020_cloudcover.npz')

    if flg_fig & (2 ** 1):
        fig_numhp_clouds('fig_numhp_clouds.png', 'modis_2020_cloudcover.npz')

    if flg_fig & (2 ** 2):
        #fig_patch_ij_binned_stats('abs_median_diff', 'median',
        #                          'mae_patches_t75_p20.npz')
        #fig_patch_ij_binned_stats('median_diff', 'mean',
        #                          'mae_patches_t75_p20.npz')
        fig_patch_ij_binned_stats('median_diff', 'median',
                                  'mae_patches_t75_p20.npz')

    # Explore the bias
    if flg_fig & (2 ** 3):
        fig_explore_bias(clobber=False)

    # VIIRS recon example
    if flg_fig & (2 ** 4):
        fig_viirs_example('fig_viirs_example.png', 75)

    # VIIRS full recon analysis
    if flg_fig & (2 ** 5):
        fig_viirs_recon_rmse('fig_viirs_recon.png', 10, 10)

    # VIIRS inpainting analysis
    if flg_fig & (2 ** 6):
        fig_llc_inpainting('fig_llcinpainting.png', 10, 10)#, debug=True)

    # VIIRS inpainting analysis
    if flg_fig & (2 ** 7):
        fig_bias('fig_bias.png')



# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # Clouds on the sphere
        #flg_fig += 2 ** 1  # Number satisfying
        #flg_fig += 2 ** 2  # Binned stats
        #flg_fig += 2 ** 3  # Bias
        #flg_fig += 2 ** 4  # VIIRS example
        #flg_fig += 2 ** 5  # VIIRS reocn analysis
        #flg_fig += 2 ** 6  # LLC inpainting
        flg_fig += 2 ** 7  # Bias
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)