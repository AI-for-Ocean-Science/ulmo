""" Code related to fronto genesis calculations"""

import numpy as np
import sys, os

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import seaborn as sns

from ulmo.plotting import plotting
from ulmo.llc import io as llc_io
from ulmo.llc import kinematics
from ulmo import io as ulmo_io

from IPython import embed

# Local
sys.path.append(os.path.abspath("../Analysis/py"))
import brazil_utils

def brazil_cutouts(outfile='brazil_kin_cutouts.npz', debug=False):
    """ Generate the Kin cutouts of F_S, w_z, Divb2 for DT ~ 1K cutouts
    in the Brazil-Malvanis confluence
    """
    # Load LLC
    tbl_test_noise_file = 's3://llc/Tables/LLC_uniform144_r0.5.parquet'
    llc_table = ulmo_io.load_main_table(tbl_test_noise_file)

    evals_bz, idx_R1, idx_R2 = brazil_utils.grab_brazil_cutouts(llc_table,
                                                             dDT=0.25) # Higher dDT for stats

    # R1 first
    R1_F_s, R1_W, R1_divb, R1_divT = [], [], [], []
    for kk, iR1 in enumerate(idx_R1):
        if debug and kk > 1:
            continue
        print(f'R1: {kk} of {idx_R1.size}')
        cutout = evals_bz.iloc[iR1]
        # Load  -- These are done local
        U, V, SST, Salt, W = llc_io.grab_velocity(
            cutout, add_SST=True, add_Salt=True, 
            add_W=True)
        # Calculate F_s
        F_s, divb = kinematics.calc_F_s(U, V, SST, Salt, add_gradb=True)
        divT = kinematics.calc_gradT(SST)
        # Store
        R1_F_s.append(F_s)
        R1_W.append(W)
        R1_divb.append(divb)
        R1_divT.append(divT)

    # R2 first
    R2_F_s, R2_W, R2_divb, R2_divT = [], [], [], []
    for kk, iR2 in enumerate(idx_R2):
        if debug and kk > 1:
            continue
        print(f'R2: {kk} of {idx_R2.size}')
        cutout = evals_bz.iloc[iR2]
        # Load 
        U, V, SST, Salt, W = llc_io.grab_velocity(
            cutout, add_SST=True, add_Salt=True,
            add_W=True)
        # Calculate
        F_s, divb = kinematics.calc_F_s(U, V, SST, Salt, add_gradb=True)
        divT = kinematics.calc_gradT(SST)
        # 
        R2_F_s.append(F_s)
        R2_W.append(W)
        R2_divb.append(divb)
        R2_divT.append(divT)

    # Output
    np.savez(outfile, 
             R1_F_s=np.stack(R1_F_s),
             R1_W=np.stack(R1_W),
             R1_divb=np.stack(R1_divb),
             R1_divT=np.stack(R1_divT),
             R2_F_s=np.stack(R2_F_s),
             R2_W=np.stack(R2_W),
             R2_divb=np.stack(R2_divb),
             R2_divT=np.stack(R2_divT),
             )
    print(f"Wrote: {outfile}")

def explore_stat_thresh(stat, outroot='_thresh.png', debug=False,
    brazil_front_file = '../Analysis/brazil_kin_cutouts.npz'):

    outfile = stat+outroot

    # Load
    brazil_front_dict = np.load(brazil_front_file)
    if stat == 'F_s':
        R1_stat = brazil_front_dict['R1_F_s']
        R2_stat = brazil_front_dict['R2_F_s']
        mnmx = [2e-5, 1e-2]
    elif stat == 'divb':
        R1_stat = brazil_front_dict['R1_divb']
        R2_stat = brazil_front_dict['R2_divb']
        mnmx = [1e-14, 1e-12]
    elif stat == 'divT':
        R1_stat = brazil_front_dict['R1_divT']
        R2_stat = brazil_front_dict['R2_divT']
        mnmx = [1e-2, 1]

    N_R1 = R1_stat.shape[0]
    N_R2 = R2_stat.shape[0]

    # Thesholds
    threshes = 10**np.linspace(np.log10(mnmx[0]),
                                 np.log10(mnmx[1]), 50)
                                
    # Prep
    nlim = [1, 3, 10, 30]
    lss = [':', '--', '-', '-.']
    lim_dict = {}
    for ilim in nlim:
        lim_dict[f'R1_{ilim}'] = []
        lim_dict[f'R2_{ilim}'] = []

    # Loop
    for thresh in threshes:
        # R1
        gd_R1 = R1_stat > thresh
        ngd_R1 = np.sum(gd_R1, axis=(1,2))
        # R2
        gd_R2 = R2_stat > thresh
        ngd_R2 = np.sum(gd_R2, axis=(1,2))

        # Loop
        for ilim in nlim:
            # R1
            nR1 = np.sum(ngd_R1 >= ilim)
            lim_dict[f'R1_{ilim}'].append(nR1) 
            # R2
            nR2 = np.sum(ngd_R2 >= ilim)
            lim_dict[f'R2_{ilim}'].append(nR2) 

    # Plot me
    
    plt.clf()
    ax = plt.gca()
    for kk, ilim in enumerate(nlim):
        # R1
        ax.plot(threshes, np.array(lim_dict[f'R1_{ilim}'])/N_R1,
                label=f'R1_{ilim}', ls=lss[kk], color='blue')
        # R2
        if ilim == nlim[0]:
            R2_lbl = f'R2_{ilim}'
        else:
            R2_lbl = None
        ax.plot(threshes, np.array(lim_dict[f'R2_{ilim}'])/N_R2,
                label=R2_lbl, ls=lss[kk], color='red')
    # 
    ax.set_xscale('log')
    ax.legend(fontsize=15.)
    ax.set_xlabel(f"{stat} Threshold")
    ax.set_ylabel("Fraction of Images")

    #
    plt.savefig(outfile, dpi=200)
    print(f"Wrote: {outfile}")
    #plt.show()

def fig_brazil_front_stats(stat:str, outroot='fig_brazil_stats',
    brazil_front_file = '../Analysis/brazil_kin_cutouts.npz',
    xlog=False):

    outfile = f'{outroot}_{stat}.png'

    # Load
    brazil_front_dict = np.load(brazil_front_file)

    # Stat
    if stat == 'F_s':
        R1_stat = brazil_front_dict['R1_F_s']
        R2_stat = brazil_front_dict['R2_F_s']
        xlbl = r'$F_s$'
    elif stat == 'divb':
        R1_stat = brazil_front_dict['R1_divb']
        R2_stat = brazil_front_dict['R2_divb']
        xlbl = r'$|\nabla b|^2$'
    elif stat == 'divT':
        R1_stat = brazil_front_dict['R1_divT']
        R2_stat = brazil_front_dict['R2_divT']
        xlbl = r'$|\nabla T|^2$'

    # Scaling
    if xlog:
        bins = np.linspace(np.log10(max(R1_stat.min(),1e-5)), 
                           np.log10(R1_stat.max()), 100)
    else:
        bins = np.linspace(R1_stat.min(), R1_stat.max(), 100)
    lscale = (xlog, True)


    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(1,2)


    # R1
    ax_R1 = plt.subplot(gs[0])
    _ = sns.histplot(x=R1_stat.flatten(), bins=bins, log_scale=lscale, ax=ax_R1)
    ax_R1.set_title('R1')

    ax_R2 = plt.subplot(gs[1])
    _ = sns.histplot(x=R2_stat.flatten(), bins=bins, log_scale=lscale, ax=ax_R2,
                     color='orange')
    ax_R2.set_title('R2')

    # Axes
    for ax in [ax_R1, ax_R2]:
        plotting.set_fontsize(ax, 14.)
        ax.set_xlabel(xlbl)
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f'Wrote: {outfile}')

def fig_brazil_divT_cdf(outfile='fig_brazil_divT_cdf.png', 
                        xlog=False, 
                        viirs_brazil_front_file='../Analysis/viirs_brazil_kin_cutouts.npz'):

    # Load
    brazil_front_file = '../Analysis/brazil_kin_cutouts.npz'
    brazil_front_dict = np.load(brazil_front_file)
    viirs_brazil_front_dict = np.load(viirs_brazil_front_file)

    # Load up
    R1_stat = brazil_front_dict['R1_divT']
    R2_stat = brazil_front_dict['R2_divT']
    viirs_R1_stat = viirs_brazil_front_dict['R1_divT']
    viirs_R2_stat = viirs_brazil_front_dict['R2_divT']
    xlbl = r'$|\nabla T|^2$'

    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(1,1)

    # CDFs
    ax = plt.subplot(gs[0])
    # LLC
    _ = sns.ecdfplot(x=R1_stat.flatten(), ax=ax, log_scale=(True,False),
                     label='R1 LLC')
    _ = sns.ecdfplot(x=R2_stat.flatten(), ax=ax, log_scale=(True,False),
                     label='R2 LLC', color='r')
    _ = sns.ecdfplot(x=viirs_R1_stat.flatten(), ax=ax, log_scale=(True,False),
                     label='R1 VIIRS', ls='--', color='b')
    _ = sns.ecdfplot(x=viirs_R2_stat.flatten(), ax=ax, log_scale=(True,False),
                     label='R2 VIIRS', ls='--', color='r')

    # Legend
    ax.legend(loc='upper left')
    ax.set_xlim(1e-8, 1.)

    # Axes
    for ax in [ax]:
        plotting.set_fontsize(ax, 14.)
        ax.set_xlabel(xlbl)
    plt.tight_layout(pad=0.2,h_pad=0.,w_pad=0.1)
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f'Wrote: {outfile}')

def main(flg):

    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    if flg & (2**0):
        brazil_cutouts()

    # F_s
    #fig_brazil_front_stats('F_s')
    # divb
    #fig_brazil_front_stats('divb')
    # divT
    #fig_brazil_front_stats('divT')

    # Thresholds
    #explore_stat_thresh('F_s')
    #explore_stat_thresh('divb')
    #explore_stat_thresh('divT')


    # VIIRS
    #fig_brazil_front_stats('divT', 
    #                       outroot='fig_viirs_brazil_stats',
    #    brazil_front_file = '../Analysis/viirs_brazil_kin_cutouts.npz',
    #    xlog=True)
    #explore_stat_thresh('divT', outroot='_viirs_thresh.png',
    #    brazil_front_file = '../Analysis/viirs_brazil_kin_cutouts.npz')

    # VIIRS vs. LLC
    #fig_brazil_divT_cdf()
    if flg & (2**9):
        fig_brazil_divT_cdf(
            outfile='fig_brazil_divT_cdf_smooth25.png',
            viirs_brazil_front_file='../Analysis/viirs_brazil_kin_cutouts_smooth25.npz')

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        flg += 2 ** 0  # 1 -- cutouts
        #flg += 2 ** 1  # 2 -- Extract + Kin
        #flg += 2 ** 2  # 4 -- Evaluate
    else:
        flg = sys.argv[1]

    main(flg)