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
from ulmo.mae import patch_analysis
from ulmo.mae import plotting as enki_plotting
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

valid_tbl_file = os.path.join(enki_path, 'Tables',
                              'MAE_LLC_valid_nonoise.parquet')
valid_img_file = os.path.join(enki_path, 'PreProc',
                              'MAE_LLC_valid_nonoise_preproc.h5')

smper = r'$m_\%$'
stper = r'$t_\%$'

def fig_reconstruct(outfile:str='fig_reconstruct.png', t:int=10, p:int=20,
                    patch_sz:int=4):

    # Load
    tbl = ulmo_io.load_main_table(valid_tbl_file)
    bias = enki_utils.load_bias((t,p))

    # Pick one
    LL = 0.
    imin = np.argmin(np.abs(tbl.LL - LL))
    cutout = tbl.iloc[imin]

    # Load the images
    recon_file = enki_utils.img_filename(t,p, local=True)
    mask_file = enki_utils.mask_filename(t,p, local=True)

    f_orig = h5py.File(valid_img_file, 'r')
    f_recon = h5py.File(recon_file, 'r')
    f_mask = h5py.File(mask_file, 'r')

    orig_img = f_orig['valid'][cutout.pp_idx, 0, :, :]
    recon_img = f_recon['valid'][cutout.pp_idx, 0, :, :]
    mask_img = f_mask['valid'][cutout.pp_idx, 0, :, :]

    # Figure time
    enki_plotting.plot_recon_four(orig_img, recon_img, mask_img, 
                             bias=bias, outfile=outfile,
                             res_vmnx=(-0.1,0.1))
    #enki_plotting.plot_recon_three(orig_img, recon_img, mask_img, 
    #                         bias=bias, outfile=outfile,
    #                         res_vmnx=(-0.1,0.1))

    # Stats
    res = recon_img - orig_img - bias
    # Ignore boundary
    mask_img[0:patch_sz, :] = 0
    mask_img[-patch_sz:, :] = 0
    mask_img[:, 0:patch_sz] = 0
    mask_img[:, -patch_sz:] = 0

    max_res = np.max(np.abs(res*mask_img))
    results = patch_analysis.patch_stats_img([orig_img, recon_img, mask_img],
                                             bias=bias)
    max_rmse = np.max(results['std_diff'])

    print(f'Residual max: {max_res:.3f}, RMSE max: {max_rmse:.3f}')


def fig_patches(outfile:str, patch_file:str):
    lsz = 16.

    fig = plt.figure(figsize=(12,7))
    plt.clf()
    gs = gridspec.GridSpec(1,2)

    # Spatial
    ax0 = plt.subplot(gs[0])
    fig_patch_ij_binned_stats('std_diff', 'median',
                              patch_file, ax=ax0)
    ax0.set_title('(a)', fontsize=lsz, color='k', loc='left')

    # RMSE
    ax1 = plt.subplot(gs[1])
    fig_patch_rmse(patch_file, in_ax=ax1)
    ax1.set_title('(b)', fontsize=lsz, color='k', loc='left')

    # Finish
    plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))


def fig_cutouts(outfile:str):

    fig = plt.figure(figsize=(14,6.5))
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
    plt.tight_layout(pad=0.05, h_pad=0.1, w_pad=0.3)
    plt.savefig(outfile, dpi=300)
    plt.close()
    print('Wrote {:s}'.format(outfile))

def fig_patch_ij_binned_stats(metric:str,
    stat:str, patch_file:str, nbins:int=16, in_ax=None):
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

    values, lbl = enki_utils.parse_metric(
        tbl, metric)

    # Do it
    median, x_edge, y_edge, ibins = scipy.stats.binned_statistic_2d(
    tbl.i_patch, tbl.j_patch, values,
        statistic=stat, expand_binnumbers=True, 
        bins=[nbins,nbins])

    # Figure
    if in_ax is None:
        fig = plt.figure(figsize=(10,8))
        plt.clf()
        ax = plt.gca()
    else:
        ax = in_ax

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
    if in_ax is None:
        plotting.set_fontsize(ax, 15)
        plt.savefig(outfile, dpi=300)
        plt.close()
        print('Wrote {:s}'.format(outfile))


def fig_patch_rmse(patch_file:str, in_ax=None, outfile:str=None,
                   tp:tuple=None):
    """ Binned stats for patches

    Args:
        patch_file (str): _description_
    """
    lims = (-5,1)

    # Parse
    if tp is None:
        t_per, p_per = enki_utils.parse_enki_file(patch_file)
    else:
        t_per, p_per = tp

    # Outfile
    if outfile is None:
        outfile = f'fig_patch_rmse_t{t_per}_p{p_per}.png'

    # Analysis
    x_edge, eval_stats, stat, x_lbl, lbl, popt = enki_anly_rms.anly_patches(
        patch_file)

    # Figure
    if in_ax is None:
        fig = plt.figure(figsize=(8, 8))
        plt.clf()
        ax = plt.gca()
    else:
        ax = in_ax

    # Patches
    plt_x = (x_edge[:-1]+x_edge[1:])/2
    ax.plot(plt_x, eval_stats, 'o', color='b', label='Patches')

    # PMC Model
    #consts = (0.01, 8.)
    consts = popt
    xval = np.linspace(lims[0], lims[1], 1000)
    yval = np.log10((10**xval + consts[0])/consts[1])
    ax.plot(xval, yval, 'r:', 
            label=r'$\log_{10}( \, (\sigma_T + '+f'{consts[0]:0.3f})/{consts[1]:0.1f}'+r')$')

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

    if in_ax is None:
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
    """ Compare LLC inpainting to Enki

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
                c=enki_tbl.LL, cmap='jet', vmin=-1000.)
                #c=enki_tbl.log10DT, cmap='jet')
    ax2.set_ylabel(r'RMSE$_{\rm inpaint}$')
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
    rmse = enki_anly_rms.create_llc_table()
    
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
    plt.title('Training Percentile: t={}'.format(models[i]))
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
        rmse = enki_anly_rms.create_llc_table()
    
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
    

def fig_viirs_rmse(outfile='fig_viirs_rmse.png', 
                   t:int=10, p:int=10, in_ax=None, 
                   nbatch:int=10, e_llc=None):
                         
    # load rmse
    llc = ulmo_io.load_main_table(os.path.join(
        os.getenv('OS_OGCM'), 'LLC', 'Enki', 
        'Tables', 'MAE_LLC_valid_nonoise.parquet'))
    
    if in_ax is None:
        fig = plt.figure(figsize=(10, 10))
        plt.clf()
        gs = gridspec.GridSpec(1,1)
        ax = plt.subplot(gs[0])

    # VIIRS
    viirs_file = os.path.join(sst_path, 'VIIRS', 'Enki', 'Tables',
        'VIIRS_all_100clear_std.parquet')
    viirs = ulmo_io.load_main_table(viirs_file)

    # LL
    rmse_viirs, starts, ends = enki_anly_rms.calc_median_LL(
        viirs, nbatch=nbatch)

    # RMSE
    rmse_viirs[f'rms_t{t}_p{p}'] = enki_anly_rms.calc_batch_RMSE(
        viirs, t, p, batch_percent=100./nbatch)
    rmse_viirs[f'rms_inpaint_t{t}_p{p}'] = enki_anly_rms.calc_batch_RMSE(
        viirs, t, p, batch_percent=100./nbatch, inpaint=True)

    # VIIRS plot
    x = rmse_viirs['median_LL']
    y = rmse_viirs[f'rms_t{t}_p{p}']
    y_inpaint = rmse_viirs[f'rms_inpaint_t{t}_p{p}']
    plt.scatter(x, y, marker='s', color='k', label='VIIRS Enki')
    plt.scatter(x, y_inpaint, marker='s', color='b', label='VIIRS Inpaint')

    # LLC analysis
    x_llc, y_llc = [], []
    for start, end in zip(starts, ends):
        in_llc = llc.LL.between(start, end)
        x_llc.append(np.median(llc[in_llc].LL))
        y_llc.append(np.median(llc[in_llc][f'RMS_t{t}_p{p}']))
        
    # LLC
    plt.scatter(x_llc, y_llc, marker='*', color='r', label='LLC Enki')
        
    fsz = 17
    plt.legend(title_fontsize=fsz+1, fontsize=fsz, 
               fancybox=True)
    #plt.title('Training Percentile: t={}'.format(models[i]))
    plt.xlabel(r"Median $LL_{\rm Ulmo}$")
    plt.ylabel("Average RMSE (K)")

    plotting.set_fontsize(ax, 19)
    ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
    plt.title(f't={t}, p={p}')
                         
    ax.set_yscale('log')
    
    if in_ax is None:
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f'Wrote: {outfile}')
    return

def fig_chk_valid(outfile='fig_chk_valid.png'):

    # load rmse
    rmse1 = enki_anly_rms.create_llc_table()
    rmse2 = enki_anly_rms.create_llc_table(
        models=[10],
        data_filepath=os.path.join(
        os.getenv('OS_OGCM'), 
        'LLC', 'Enki', 'Tables', 'Enki_LLC_valid_nonoise.parquet'))
    
    fig = plt.figure(figsize=(10, 10))
    plt.clf()
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])
    
    models = [10]
    masks = [10,20,30,40,50]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    plt_labels = []
        
    i=0
        
    # plot
    for p, c in zip(masks, colors):
        x1 = rmse1['median_LL']
        y1 = rmse1['rms_t{t}_p{p}'.format(t=models[i], p=p)]
        x2 = rmse2['median_LL']
        y2 = rmse2['rms_t{t}_p{p}'.format(t=models[i], p=p)]
        #
        plt_labels.append(smper+f'={p}')
        if p == 10:
            lbl1 = 'Valid 1'
            lbl2 = 'Valid 2'
        else:
            lbl1, lbl2 = None, None
        plt.scatter(x1, y1, marker='s', color=c, label=lbl1)
        plt.scatter(x2, y2, marker='o', color=c, label=lbl2)

    #ax.set_ylim([0, 0.20])
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
    fsz = 17
    plt.legend(labels=plt_labels, title='Patch Mask Ratio',
                title_fontsize=fsz+1, fontsize=fsz, fancybox=True)
    plt.title('Training Percentile: t={}'.format(models[i]))
    plt.xlabel(r"Median $LL_{\rm Ulmo}$")
    plt.ylabel("Average RMSE (K)")

    plotting.set_fontsize(ax, 19)
    ax.legend()
                         
    #fig.tight_layout()
    #fig.subplots_adjust(top=0.92)
    #fig.subplots_adjust(wspace=0.2)
    #fig.suptitle('RMSE vs LL', fontsize=16)
    
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

    # Example
    if flg_fig & (2 ** 3):
        fig_reconstruct()

    # VIIRS vs LLC with LL
    if flg_fig & (2 ** 4):
        fig_viirs_rmse()

    # Check valid 2
    if flg_fig & (2 ** 5):
        fig_chk_valid()

    # VIIRS patches
    if flg_fig & (2 ** 6):
        fig_patch_rmse(
            '/home/xavier/Projects/Oceanography/SST/VIIRS/Enki/Recon/VIIRS_100clear_patches_t10_p10.npz',
            outfile='fig_viirs_patches_t10_p10.png',
            tp=(10,10))


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        flg_fig += 2 ** 0  # patches
        #flg_fig += 2 ** 1  # cutouts
        #flg_fig += 2 ** 2  # LLC (Enki vs inpainting)
        #flg_fig += 2 ** 3  # Reconstruction example
        #flg_fig += 2 ** 4  # VIIRS LL
        #flg_fig += 2 ** 5  # Check valid 2
        #flg_fig += 2 ** 6  # VIIRS patches
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)