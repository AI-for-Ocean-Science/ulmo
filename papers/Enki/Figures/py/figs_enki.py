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
from ulmo.mae import enki_utils
from ulmo.mae.cutout_analysis import rms_images
from ulmo import io as ulmo_io
from ulmo.utils import image_utils
try:
    from ulmo.mae import models_mae
except (ModuleNotFoundError, ImportError):
    print("Not able to load the models")
else:    
    from ulmo.mae import reconstruct
from ulmo.mae import plotting as mae_plotting


from IPython import embed

# Local
sys.path.append(os.path.abspath("../../MAE/Analysis/py"))
import anly_patches
sys.path.append(os.path.abspath("../Analysis/py"))
import enki_anly_rms
#sys.path.append(os.path.abspath("../Figures/py"))
#import fig_ssl_modis

# Globals

#preproc_path = os.path.join(os.getenv('OS_AI'), 'MAE', 'PreProc')
#recon_path = os.path.join(os.getenv('OS_AI'), 'MAE', 'Recon')
#orig_file = os.path.join(preproc_path, 'MAE_LLC_valid_nonoise_preproc.h5')

sst_path = os.getenv('OS_SST')
ogcm_path = os.getenv('OS_OGCM')
enki_path = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki')

smper = r'$m_\%$'
stper = r'$t_\%$'

def fig_patches(outfile:str, patch_file:str):

    fig = plt.figure(figsize=(12,7))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # Spatial
    ax0 = plt.subplot(gs[0])
    fig_patch_ij_binned_stats('std_diff', 'median',
                              patch_file, ax=ax0)
    lsz = 16.
    ax0.set_title('(a)', fontsize=lsz, color='k', loc='left')

    # RMSE
    ax1 = plt.subplot(gs[1])
    fig_patch_rmse(patch_file, ax=ax1)
    ax1.set_title('(b)', fontsize=lsz, color='k', loc='left')

    # Finish
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_cutouts(outfile:str):

    fig = plt.figure(figsize=(13,6))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # Spatial
    ax0 = plt.subplot(gs[0])
    rmse = figs_rmse_vs_LL(ax=ax0)

    lsz = 16.
    ax0.set_title('(a)', fontsize=lsz, color='k')

    # RMSE
    ax1 = plt.subplot(gs[1])
    fig_rmse_models(ax=ax1, rmse=rmse)
    ax1.set_title('(b)', fontsize=lsz, color='k')

    # Finish
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_patch_ij_binned_stats(metric:str,
    stat:str, patch_file:str, nbins:int=16, ax=None):
    """ Binned stats for patches

    Args:
        metric (str): _description_
        stat (str): _description_
        patch_file (str): _description_
        nbins (int, optional): _description_. Defaults to 16.
    """

    # Parse
    t_per, p_per = enki_utils.parse_enki_file(patch_file)

    # Outfile
    outfile = f'fig_{metric}_{stat}_t{t_per}_p{p_per}_patch_ij_binned_stats.png'
    # Load
    patch_file = os.path.join(os.getenv("OS_OGCM"),
        'LLC', 'Enki', 'Recon', patch_file)
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
    if ax is None:
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
    cbaxes = plt.colorbar(mplt, pad=0., fraction=0.030, orientation='horizontal') #location='left')
    cbaxes.set_label(f'{stat}({lbl})', fontsize=17.)
    cbaxes.ax.tick_params(labelsize=15)

    # Axes
    ax.set_xlabel(r'$i$')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel(r'$j$')
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_aspect('equal')

    # Finish
    if ax is None:
        plotting.set_fontsize(ax, 15)
        plt.savefig(outfile, dpi=300)
        plt.close()
        print('Wrote {:s}'.format(outfile))


def fig_patch_rmse(patch_file:str, nbins:int=16, ax=None):
    """ Binned stats for patches

    Args:
        patch_file (str): _description_
        nbins (int, optional): _description_. Defaults to 16.
    """
    lims = (-5,1)

    # Parse
    t_per, p_per = enki_utils.parse_enki_file(patch_file)

    # Outfile
    outfile = f'fig_patch_rmse_t{t_per}_p{p_per}.png'

    # Load
    patch_file = os.path.join(os.getenv("OS_OGCM"),
        'LLC', 'Enki', 'Recon', patch_file)

    f = np.load(patch_file)
    data = f['data']
    data = data.reshape((data.shape[0]*data.shape[1], 
                         data.shape[2]))

    items = f['items']
    tbl = pandas.DataFrame(data, columns=items)

    nbins = 32
    metric = 'log10_std_diff'
    stat = 'median'

    x_metric = 'log10_stdT'
    xvalues, x_lbl = anly_patches.parse_metric(tbl, x_metric)

    values, lbl = anly_patches.parse_metric(tbl, metric)

    good = np.isfinite(xvalues.values)

    # Do it
    eval_stats, x_edge, ibins = scipy.stats.binned_statistic(
        xvalues.values[good], values.values[good], statistic=stat, bins=nbins)

    # Figure
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        plt.clf()
        ax = plt.gca()

    # Patches
    plt_x = (x_edge[:-1]+x_edge[1:])/2
    ax.plot(plt_x, eval_stats, 'b', label='Patches')

    # PMC Model
    consts = (0.01, 8.)
    xval = np.linspace(lims[0], lims[1], 1000)
    yval = np.log10((10**xval + consts[0])/consts[1])
    ax.plot(xval, yval, 'r:', label=r'$\log_{10}( \, (\sigma_T + '+f'{consts[0]})/{consts[1]}'+r')$')

    # Axes
    ax.set_xlabel(x_lbl)
    ax.set_ylabel(f'{stat}({lbl})')

    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[0], lims[1])

    # Labeling
    fsz = 17.
    ax.text(0.05, 0.9, stper+f'={t_per}, '+smper+f'={p_per}',
            transform=ax.transAxes,
              fontsize=fsz, ha='left', color='k')

    ax.legend(loc='lower right', fontsize=fsz-2)

    plotting.set_fontsize(ax, fsz)

    # 1-1
    ax.plot(lims, lims, 'k--')

    # Grid
    ax.grid()

    if ax is None:
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
                       debug:bool=False, single:bool=False):
    """_summary_

    Args:
        outfile (str): _description_
        t (int): _description_
        p (int): _description_
        debug (bool, optional): _description_. Defaults to False.
        single (bool, optional): 
            Show a single panel.  Defaults to False.
    """

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
    recon_file = enki_utils.img_filename(t,p, local=True)
    mask_file = enki_utils.mask_filename(t,p, local=True)

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
    #orig_imgs = f_orig['valid'][:nimgs,0,...]
    #mask_imgs = f_mask['valid'][:nimgs,0,...]

    # Allow for various shapes (hack)
    #recon_imgs = f_recon['valid'][:nimgs,0,...]


    #rms_enki = rms_images(orig_imgs, recon_imgs, mask_imgs)
    rms_enki = rms_images(f_orig, f_recon, f_mask, nimgs=nimgs)
    
    #inpaint_imgs = f_inpaint['inpainted'][:nimgs,...]
    #rms_inpaint = rms_images(orig_imgs, inpaint_imgs, mask_imgs)
    rms_inpaint = rms_images(f_orig, f_inpaint, f_mask, nimgs=nimgs,
                             keys=['valid', 'inpainted', 'valid'])

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
    if single:
        gs = gridspec.GridSpec(1,1)
    else:
        gs = gridspec.GridSpec(2,2)
    plt.clf()

    axes = []
    if not single:
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
        # Add axes
        axes.append(ax0)
        axes.append(ax1)

    # RMS_biharmonic vs. RMS_Enki
    if single:
        ax2 = plt.subplot(gs[0])
    else:
        ax2 = plt.subplot(gs[3])
    #embed(header='429 of figs')
    scat = ax2.scatter(enki_tbl.rms_enki, 
                enki_tbl.rms_inpaint, s=0.1,
                c=enki_tbl.LL, cmap='jet')
                #c=enki_tbl.log10DT, cmap='jet')
    ax2.set_ylabel(r'RMSE$_{\rm biharmonic}$')
    ax2.set_xlabel(r'RMSE$_{\rm Enki}$')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    cbaxes = plt.colorbar(scat)#, pad=0., fraction=0.030)
    #cbaxes.set_label(r'$\log_{10} \, \Delta T$ (K)', fontsize=17.)
    cbaxes.set_label(r'$LL_{\rm Ulmo}$', fontsize=17.)
    cbaxes.ax.tick_params(labelsize=15)

    ax2.plot([1e-3, 10], [1e-3,10], 'k--')
    axes.append(ax2)

    # RMS_Enki vs. DT
    if not single:
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
        axes.append(ax3)

    # Polish
    #fg.ax.minorticks_on()
    lsz = 17 if single else 14
    for ax in axes:
        plotting.set_fontsize(ax, lsz)

    #plt.title(f'Enki vs. Inpaiting: t={t}, p={p}')

    # Finish
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def figs_rmse_vs_LL(outfile='rmse_t10only.png', ax=None):

                    
    # load rmse
    rmse = enki_anly_rms.create_table()
    
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        plt.clf()
        gs = gridspec.GridSpec(1,1)
        ax = plt.subplot(gs[0])
    
    models = [10,35,50,75]
    masks = [10,20,30,40,50]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    plt_labels = []
        
    i=0
        
    # plot
    for p, c in zip(masks, colors):
        x = rmse['median_LL']
        y = rmse['rms_t{t}_p{p}'.format(t=models[i], p=p)]
        plt_labels.append(smper+f'={p}')
        plt.scatter(x, y, color=c)

    #ax.set_ylim([0, 0.20])
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
    fsz = 17
    plt.legend(labels=plt_labels, title='Patch Mask Ratio',
                title_fontsize=fsz+1, fontsize=fsz, fancybox=True)
    plt.title('Training Ratio: t={}'.format(models[i]))
    plt.xlabel(r"Median $LL_{\rm Ulmo}$")
    plt.ylabel("Average RMSE (K)")

    plotting.set_fontsize(ax, 19)
                         
    #fig.tight_layout()
    #fig.subplots_adjust(top=0.92)
    #fig.subplots_adjust(wspace=0.2)
    #fig.suptitle('RMSE vs LL', fontsize=16)
    
    if ax is None:
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f'Wrote: {outfile}')
    return rmse

def fig_rmse_models(outfile='fig_rmse_models.png', ax=None, rmse=None):
                         
    # load rmse
    if rmse is None:
        rmse = enki_anly_rms.create_table()
    
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        plt.clf()
        gs = gridspec.GridSpec(1,1)
        ax = plt.subplot(gs[0])
    
    models = [10,35,50,75]
    masks = [10,20,30,40,50]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    plt_labels = []
        
    
    # plot
    for i in range(4):
        avg_RMSEs = []
        for p, c in zip(masks, colors):
            y = rmse['rms_t{t}_p{p}'.format(t=models[i], p=p)]
            avg_RMSE = np.mean(y)
            avg_RMSEs.append(avg_RMSE)
        # Plot
        plt_labels.append(stper+f'={models[i]}')
        plt.plot(masks, avg_RMSEs, 's', ms=10, color=colors[i])

    ax.set_ylim([0, 0.25])
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
    fsz = 17
    plt.legend(labels=plt_labels, title='Training Percentile ('+stper+')',
                title_fontsize=fsz+1, fontsize=fsz, fancybox=True)
    #plt.xlabel("Training Ratio")
    plt.xlabel("Patch Masking Percentile ("+smper+")")
    plt.ylabel("Average RMSE (K)")

    plotting.set_fontsize(ax, 19)
                         
    #fig.tight_layout()
    #fig.subplots_adjust(top=0.92)
    #fig.subplots_adjust(wspace=0.2)
    #fig.suptitle('RMSE vs LL', fontsize=16)
    
    if ax is None:
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f'Wrote: {outfile}')
    return
    

#### ########################## #########################
def main(flg_fig):
    if flg_fig == 'all':
        flg_fig = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_fig = int(flg_fig)

    # Patches
    if flg_fig & (2 ** 0):
        fig_patches('fig_patches_t10_p20.png',
                    'mae_patches_t10_p20.npz')

    # Cutouts
    if flg_fig & (2 ** 1):
        fig_cutouts('fig_cutouts.png')

    # LLC (Enki vs inpainting)
    if flg_fig & (2 ** 2):
        fig_llc_inpainting('fig_llcinpainting.png', 10, 10, single=True)#, debug=True)

# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # patches
        #flg_fig += 2 ** 1  # cutouts
        flg_fig += 2 ** 2  # LLC (Enki vs inpainting)
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)