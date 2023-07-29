""" Figures for Enki paper """
import os, sys
import numpy as np
import scipy
from pkg_resources import resource_filename

import xarray

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
                              'Enki_LLC_valid_nonoise.parquet')
valid_img_file = os.path.join(enki_path, 'PreProc',
                              'Enki_LLC_valid_nonoise_preproc.h5')

smper = r'$m_\%$'
stper = r'$t_\%$'

def fig_reconstruct(outfile:str='fig_reconstruct.png', t:int=20, 
                    p:int=30, patch_sz:int=4):

    # Load
    tbl_file, orig_file, recon_file, mask_file = enki_utils.set_files(
        'LLC2_nonoise', 20, 30)

    print(f"Orig: {orig_file}")
    print(f"Recon: {recon_file}")
    print(f"Mask: {mask_file}")

    tbl = ulmo_io.load_main_table(tbl_file)
    bias = enki_utils.load_bias((t,p))

    # Pick one
    #LL = 1. # Pretty nice
    LL = 2. # Pretty nice
    imin = np.argmin(np.abs(tbl.LL - LL))
    cutout = tbl.iloc[imin]


    f_orig = h5py.File(orig_file, 'r')
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

    print(f'Residual max: {max_res:.3f}, RMSE max: {max_rmse:.3f}, Average RMSE: {np.mean(results["std_diff"]):.3f}')


def fig_patches(outfile:str, patch_file:str, model:str='std'):
    print(f"Using: {patch_file}")
    lsz = 16.

    fig = plt.figure(figsize=(12,7))
    plt.clf()
    gs = gridspec.GridSpec(4,8)

    # Spatial
    ax0 = plt.subplot(gs[:,0:4])
    fig_patch_ij_binned_stats('std_diff', 'median',
                              patch_file, in_ax=ax0)
    ax0.set_title('(a)', fontsize=lsz, color='k', loc='left')

    # RMSE
    ax1 = fig_patch_rmse(patch_file, in_ax=gs, model=model)
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
    models = [10,20,35,50,75]
    ax0 = plt.subplot(gs[0])
    rmse = figs_rmse_vs_LL(ax=ax0, models=models)

    lsz = 16.
    ax0.set_title('(a)', fontsize=lsz, color='k')

    # RMSE
    ax1 = plt.subplot(gs[1])
    fig_rmse_models(ax=ax1, rmse=rmse, models=models)
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
                   other_patch_files:list=None,
                   lbls:list=None, show_model:bool=True,
                   tp:tuple=None, model:str='std'):
    """ Binned stats for patches

    Args:
        patch_file (str): Path to patch file
        another_patch_file (str, optional): Path to another patch file. Defaults to None.
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
    x_edge, eval_stats, stat, x_lbl, lbl, popt, tbl = enki_anly_rms.anly_patches(
        patch_file, model=model)

    # Others
    if other_patch_files is not None:
        items = []
        for other_patch_file in other_patch_files:
            items.append(enki_anly_rms.anly_patches(
                other_patch_file, model=model))

    # Figure
    ax_hist = None
    if in_ax is None:
        fig = plt.figure(figsize=(8, 8))
        plt.clf()
        ax = plt.gca()
    elif isinstance(in_ax, plt.Axes):
        ax = in_ax
    else:
        ax = plt.subplot(in_ax[1:,4:])
        ax_hist = plt.subplot(in_ax[0,4:], sharex=ax)

    # Patches
    plt_x = (x_edge[:-1]+x_edge[1:])/2
    ax.plot(plt_x, eval_stats, 'o', color='b', 
            label='Patches' if lbls is None else lbls[0])

    markers = ['*', '^', 's',  'v']
    colors = ['g', 'orange', 'purple', 'cyan']
    if other_patch_files is not None:
        for kk in range(len(other_patch_files)):
            x_edge2, eval_stats2, _, _, _, _, _ = items[kk]
            plt_x2 = (x_edge2[:-1]+x_edge2[1:])/2
            ax.plot(plt_x2, eval_stats2, markers[kk], 
                    color=colors[kk], label=lbls[kk+1])

    # Write
    df = pandas.DataFrame()
    df['log10_sigT'] = plt_x
    df['log10_RMSE'] = eval_stats
    df.to_csv('fig_patch_rmse.csv', index=False)
    print(f'Wrote: fig_patch_rmse.csv')

    # PMC Model
    #consts = (0.01, 8.)
    consts = popt
    xval = np.linspace(lims[0], lims[1], 1000)
    if model == 'std':
        yval = enki_anly_rms.two_param_model(10**xval, floor=consts[0], scale=consts[1])
    elif model == 'denom':
        yval = enki_anly_rms.denom_model(10**xval, floor=consts[0], scale=consts[1])
        
    if show_model:
        ax.plot(xval, np.log10(yval), 'r:', 
            label=r'$\log_{10}( \, (\sigma_T + '+f'{consts[0]:0.3f})/{consts[1]:0.1f}'+r')$')


    # Histogram
    if ax_hist is not None:
        keep = tbl.stdT > 0.
        ax_hist.hist(np.log10(tbl.stdT.values[keep]), bins=100, density=True, color='b')#, alpha=0.5)
        plt.setp(ax_hist.get_xticklabels(), visible=False)


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

    return ax_hist if ax_hist is not None else ax

    

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
    chkpt_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki', 'models', f'mae_t{t}_399.pth')
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
        'Enki_LLC_valid_nonoise.parquet')
    local_mae_valid_nonoise_file = os.path.join(
        enki_path, 'PreProc', 
        'Enki_LLC_valid_nonoise_preproc.h5')
    local_orig_file = local_mae_valid_nonoise_file
    inpaint_file = os.path.join(
        ogcm_path, 'LLC', 'Enki', 
        'Recon', f'Enki_LLC2_nonoise_biharmonic_t{t}_p{p}.h5')
    recon_file = enki_utils.img_filename(t,p, local=True, dataset='LLC2_nonoise')
    mask_file = enki_utils.mask_filename(t,p, local=True, dataset='LLC2_nonoise')

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


    # Calculate
    rms_enki = rms_images(f_orig, f_recon, f_mask, nimgs=nimgs)
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

    ax2.plot([1e-3, 1], [1e-3,1], 'k--')
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


def figs_rmse_vs_LL(outfile='rmse_t10only.png', ax=None, models=None):
    """_summary_

    Args:
        outfile (str, optional): _description_. Defaults to 'rmse_t10only.png'.
        ax (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    # Setup
    if models is None:
        models = [10,20,35,50,75]
    enki_file = os.path.join(os.getenv('OS_OGCM'), 
        'LLC', 'Enki', 'Tables', 'Enki_LLC_valid_nonoise.parquet')

    # load rmse
    rmse = enki_anly_rms.create_llc_table(models=models,
        data_filepath=enki_file)
        
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        plt.clf()
        gs = gridspec.GridSpec(1,1)
        ax = plt.subplot(gs[0])
    
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

def fig_rmse_models(outfile='fig_rmse_models.png', ax=None, rmse=None, models=None):
                         
    # load rmse
    if rmse is None:
        rmse = enki_anly_rms.create_llc_table()
    
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        plt.clf()
        gs = gridspec.GridSpec(1,1)
        ax = plt.subplot(gs[0])
    
    if models is None:
        models = [10,20,35,50,75]
    masks = [10,20,30,40,50]
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    plt_labels = []
        
    
    # plot
    for i in range(len(models)):
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
                   nbatch:int=10, show_inpaint:bool=True,
                   show_llc_quad:bool=False):
                         
    # load rmse
    llc = ulmo_io.load_main_table(os.path.join(
        os.getenv('OS_OGCM'), 'LLC', 'Enki', 
        'Tables', 'Enki_LLC_valid_nonoise.parquet'))
    
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
    #embed(header='fig_: 719')

    # VIIRS plot
    x = rmse_viirs['median_LL']
    y = rmse_viirs[f'rms_t{t}_p{p}']
    y_inpaint = rmse_viirs[f'rms_inpaint_t{t}_p{p}']
    plt.scatter(x, y, marker='s', color='k', label=f'VIIRS Enki ({stper}={t}, {smper}={p})')
    if show_inpaint:
        plt.scatter(x, y_inpaint, marker='o', color='b', label='VIIRS Biharmonic')

    # LLC analysis
    x_llc, y_llc = [], []
    for start, end in zip(starts, ends):
        in_llc = llc.LL.between(start, end)
        x_llc.append(np.median(llc[in_llc].LL))
        y_llc.append(np.median(llc[in_llc][f'RMS_t{t}_p{p}']))
        
    # LLC
    plt.scatter(x_llc, y_llc, marker='*', color='r', label='LLC Enki')

    # LLC quad
    if show_llc_quad:
        y_llc_quad = np.sqrt(np.array(y_llc)**2 + 0.03**2)
        plt.scatter(x_llc, y_llc_quad, marker='o', color='r', label='LLC Enki + VIIRS noise')
        
    fsz = 17
    plt.legend(title_fontsize=fsz+1, fontsize=fsz, 
               fancybox=True)
    #plt.title('Training Percentile: t={}'.format(models[i]))
    plt.xlabel(r"Median $LL_{\rm Ulmo}$")
    plt.ylabel("Average RMSE (K)")

    plotting.set_fontsize(ax, 19)
    ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
    #plt.title(f't={t}, p={p}')
                         
    #if show_inpaint:                         
    #    ax.set_yscale('log')
    
    if in_ax is None:
        plt.savefig(outfile, dpi=300)
        plt.close()
        print(f'Wrote: {outfile}')
    return

def fig_chk_valid(outfile='fig_chk_valid.png'):

    # load rmse
    rmse1 = enki_anly_rms.create_llc_table()
    rmse2 = enki_anly_rms.create_llc_table(
        models=[10], data_filepath=os.path.join(
        os.getenv('OS_OGCM'), 
        'LLC', 'Enki', 'Tables', 'Enki_LLC_valid_nonoise.parquet'))
    rmse3 = enki_anly_rms.create_llc_table(
        models=[10], data_filepath=os.path.join(
        os.getenv('OS_OGCM'), 
        'LLC', 'Enki', 'Tables', 'Enki_LLC_valid_noise.parquet'))
    
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
        plt_labels.append(smper+f'={p}')
        for ss, rmse in enumerate([rmse1, rmse2, rmse3]):
            x = rmse['median_LL']
            y = rmse['rms_t{t}_p{p}'.format(t=models[i], p=p)]

            # Marker
            if ss == 0:
                mrkr = 's'
            elif ss == 1:
                mrkr = 'o'
            elif ss == 2:
                mrkr = '*'
            
            # Labels
            if p == 10:
                if ss == 0:
                    lbl = 'Original'
                elif ss == 1:
                    lbl = 'Offset by 0.25deg'
                elif ss == 2:
                    lbl = 'Offset + noise'
            else:
                lbl = None
            plt.scatter(x, y, marker=mrkr, color=c, label=lbl)

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


def fig_llc_many_inpainting(outfile='fig_llc_many_inpainting.png', 
                   t:int=20, p:int=30, in_ax=None, 
                   nbatch:int=10, show_inpaint:bool=True,
                   show_llc_quad:bool=False):
                         
    # load rmse
    llc = ulmo_io.load_main_table(os.path.join(
        os.getenv('OS_OGCM'), 'LLC', 'Enki', 
        'Tables', 'Enki_LLC_valid_nonoise.parquet'))
    
    fig = plt.figure(figsize=(10, 10))
    plt.clf()
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])

    # Calculations
    models = [t]
    masks = [p]
    rmse_std = enki_anly_rms.create_llc_table(table=llc, models=models, masks=masks)
    rmse_biharm = enki_anly_rms.create_llc_table(table=llc, method='biharmonic',
                                                 models=models, masks=masks)
    rmse_nearest = enki_anly_rms.create_llc_table(table=llc, method='grid_nearest',
                                                 models=models, masks=masks)
    rmse_linear = enki_anly_rms.create_llc_table(table=llc, method='grid_linear',
                                                 models=models, masks=masks)
    rmse_cubic = enki_anly_rms.create_llc_table(table=llc, method='grid_cubic',
                                                 models=models, masks=masks)

    #embed(header='827 of figs')

    # Plot me
    for tbl, lbl in zip([rmse_std, rmse_biharm, rmse_nearest, rmse_linear, rmse_cubic], 
                        ['Enki', 'biharm', 'nearest', 'linear', 'cubic']):
        x_llc = tbl['median_LL']
        y_llc = tbl[f'rms_t{t}_p{p}']
        if lbl == 'Enki':
            marker = '*'
        else:
            marker = 'o'
        plt.scatter(x_llc, y_llc, marker=marker, label=lbl)

        
    fsz = 17
    plt.legend(title_fontsize=fsz+1, fontsize=fsz, 
               fancybox=True)
    #plt.title('Training Percentile: t={}'.format(models[i]))
    plt.xlabel(r"Median $LL_{\rm Ulmo}$")
    plt.ylabel("Average RMSE (K)")

    plotting.set_fontsize(ax, 19)
    ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
                         
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f'Wrote: {outfile}')


def fig_dineof(outfile='fig_dineof.png'): 
                         
    # Calculate RMSE
    # rmse
    orig_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki', 'DINEOF',
                             'Enki_LLC_orig.nc')
    ds_orig = xarray.open_dataset(orig_file)
    orig_imgs = np.asarray(ds_orig.variables['SST'])

    def simple_rmse(recon_imgs, mask_imgs):
        calc = (recon_imgs - orig_imgs)*mask_imgs
        calc = calc**2
        nmask = np.sum(mask_imgs, axis=(1,2))
        calc = np.sum(calc, axis=(1,2)) / nmask
        rmse = np.sqrt(calc)
        return np.mean(rmse)

    def rmse_DINEOF(p):
        # open files
        dineof_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki', 'DINEOF',
            f'Enki_LLC_DINEOF_p{p}.nc')
        print(f'Working on: {dineof_file}')
        ds_recon = xarray.open_dataset(dineof_file)
        mask_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki', 'Recon',
            f'mae_mask_t75_p{p}.h5')
        f_ma = h5py.File(mask_file, 'r')
        
        # extract data
        recon_imgs = np.asarray(ds_recon.variables['sst_filled'])
        mask_imgs = f_ma['valid'][:180,0,...]
        #for i in range(180):
        #    mask_imgs.append(f_ma['valid'][i,0,...])

        return simple_rmse(recon_imgs, mask_imgs)
            
    def rmse_enki(p):
        enki_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Enki', 'DINEOF',
            f'Enki_LLC_DINEOF_enki_p{p}.nc')
        mask_file = enki_file.replace('enki', 'mask')
        # Load
        f_enki = h5py.File(enki_file, 'r')
        recon_imgs = np.asarray(f_enki['valid'][:,0,...])
        f_mask = h5py.File(mask_file, 'r')
        mask_imgs = np.asarray(f_mask['valid'][:,0,...])

        return simple_rmse(recon_imgs, mask_imgs)

    rmses = []
    enki_rmses = []
    ps = [10, 20, 30, 40, 50]
    for p in ps:
        rmses.append(rmse_DINEOF(p))
        enki_rmses.append(rmse_enki(p))
    
    # Plot
    fig = plt.figure(figsize=(10, 10))
    plt.clf()
    gs = gridspec.GridSpec(1,1)
    ax = plt.subplot(gs[0])

    plt.plot(ps, rmses, 'o', label='DINEOF')
    plt.plot(ps, enki_rmses, 's', color='k', label='Enki')

    fsz = 17
    plt.legend(title_fontsize=fsz+1, fontsize=fsz, 
               fancybox=True, loc='lower left')
    #plt.title('Training Percentile: t={}'.format(models[i]))
    plt.xlabel(smper)
    plt.ylabel("Average RMSE (K)")
    ax.set_xlim(0., 60)
    ax.set_ylim(0., 0.3)

    plotting.set_fontsize(ax, 19)
    ax.grid(color='gray', linestyle='dashed', linewidth = 0.5)
                         
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f'Wrote: {outfile}')

#### ########################## #########################
def main(flg_fig):
    if flg_fig == 'all':
        flg_fig = np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg_fig = int(flg_fig)

    # Patches
    if flg_fig & (2 ** 0):
        #fig_patches('fig_patches_t10_p20.png',
        #            'enki_patches_t10_p20.npz')
        fig_patches('fig_patches_t20_p30.png',
                    'enki_patches_t20_p30.npz')
        #fig_patches('fig_patches_t10_p20_denom.png',
        #            'mae_patches_t10_p20.npz',
        #            model='denom')

    # Cutouts
    if flg_fig & (2 ** 1):
        fig_cutouts('fig_cutouts.png')

    # LLC (Enki vs inpainting)
    if flg_fig & (2 ** 2):
        #fig_llc_inpainting('fig_llcinpainting_t10_p10.png', 10, 10, 
        #                   single=True)#, debug=True)
        fig_llc_inpainting('fig_llcinpainting_t20_p30.png', 
                           20, 30, 
                           single=True)#, debug=True)

    # Example
    if flg_fig & (2 ** 3):
        fig_reconstruct()

    # VIIRS vs LLC with LL
    if flg_fig & (2 ** 4):
        #fig_viirs_rmse()
        fig_viirs_rmse(outfile='fig_viirs_rmse_t20p40.png',
                       t=20, p=40, show_inpaint=False)
        #fig_viirs_rmse(outfile='fig_viirs_rmse_t20p50.png',
        #               t=20, p=50)
        #fig_viirs_rmse(outfile='fig_viirs_rmse_noinpaint.png',
        #               show_inpaint=False)
        #fig_viirs_rmse(outfile='fig_viirs_rmse_quad.png',
        #               show_inpaint=False, show_llc_quad=True)

    # Check valid 2, with and without noise
    if flg_fig & (2 ** 5):
        fig_chk_valid()

    # More patch figures
    if flg_fig & (2 ** 6):
        fig_patch_rmse(
                '/backup/Oceanography/OGCM/LLC/Enki/Recon/enki_patches_t10_p10.npz',
            other_patch_files=[
                '/backup/Oceanography/OGCM/LLC/Enki/Recon/enki_noise_patches_t10_p10.npz',
                '/backup/Oceanography/OGCM/LLC/Enki/Recon/enki_noise_patches_noiseless_t10_p10.npz',
                '/home/xavier/Projects/Oceanography/SST/VIIRS/Enki/Recon/VIIRS_100clear_patches_t10_p10.npz',
                #'/backup/Oceanography/OGCM/LLC/Enki/Recon/enki_noise02_patches_t10_p10.npz',
            ],
            lbls=[
                  'LLC without Noise', 
                  'LLC with Noise=0.04K', 
                  'LLC Noise+Noiseless', 
                'VIIRS', 
                  #'LLC2 Noise 0.02K',
                  ],
            outfile='fig_viirs_llc_patches_t10_p10.png',
            tp=(10,10),
            show_model=False)

        #fig_patch_rmse(
        #    '/home/xavier/Projects/Oceanography/SST/VIIRS/Enki/Recon/VIIRS_100clear_patches_t10_p10.npz',
        #    outfile='fig_viirs_patches_t10_p10.png',
        #    tp=(10,10))

    # Lots of LLC2 with inpainting
    if flg_fig & (2 ** 7):
        fig_llc_many_inpainting()

    # DINEOF
    if flg_fig & (2 ** 8):
        fig_dineof()


# Command line execution
if __name__ == '__main__':

    if len(sys.argv) == 1:
        flg_fig = 0
        #flg_fig += 2 ** 0  # patches
        #flg_fig += 2 ** 1  # cutouts
        #flg_fig += 2 ** 2  # LLC RMSE (Enki vs inpainting)
        #flg_fig += 2 ** 3  # Reconstruction example
        #flg_fig += 2 ** 4  # VIIRS RMSE vs LL (Figure 5)
        #flg_fig += 2 ** 5  # Check valid 2
        flg_fig += 2 ** 6  # More patch figures
        #flg_fig += 2 ** 7  # Compare Enki against many inpainting
        #flg_fig += 2 ** 8  # DINEOF
    else:
        flg_fig = sys.argv[1]

    main(flg_fig)
