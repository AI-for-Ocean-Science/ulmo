""" Code related to fronto genesis calculations"""
import numpy as np

from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import seaborn as sns

from ulmo.plotting import plotting
from ulmo.llc import io as llc_io
from ulmo import io as ulmo_io
from ulmo.llc import kinematics

from IPython import embed

def grab_brazil_cutouts(DT=2.05, dDT=0.05):
    # Load LLC
    #tbl_test_noise_file = 's3://llc/Tables/test_noise_modis2012.parquet'
    tbl_test_noise_file = 's3://llc/Tables/LLC_uniform144_r0.5.parquet'
    llc_table = ulmo_io.load_main_table(tbl_test_noise_file)

    # Add in DT
    if 'DT' not in llc_table.keys():
        llc_table['DT'] = llc_table.T90 - llc_table.T10

    # Brazil
    in_brazil = ((np.abs(llc_table.lon.values + 57.5) < 10.)  & 
        (np.abs(llc_table.lat.values + 43.0) < 10))
    in_DT = np.abs(llc_table.DT - DT) < dDT
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

    # Grab em
    idx_R1 = np.where(in_R1)[0]
    idx_R2 = np.where(in_R2)[0]

    print(f"We have {idx_R1.size} cutouts in R1 and {idx_R2.size} in R2")

    return evals_bz, idx_R1, idx_R2


def brazil_pdfs(outfile='brazil_kin_cutouts.npz', debug=False):
    """ Generate the Kin cutouts of F_S, w_z, Divb2 for DT ~ 1K cutouts
    in the Brazil-Malvanis confluence
    """
    evals_bz, idx_R1, idx_R2 = grab_brazil_cutouts(dDT=0.25) # Higher dDT for stats

    # R1 first
    R1_F_s, R1_W, R1_divb = [], [], []
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
        # Store
        R1_F_s.append(F_s)
        R1_W.append(W)
        R1_divb.append(divb)

    # R2 first
    R2_F_s, R2_W, R2_divb = [], [], []
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
        # 
        R2_F_s.append(F_s)
        R2_W.append(W)
        R2_divb.append(divb)

    # Output
    np.savez(outfile, 
             R1_F_s=np.stack(R1_F_s),
             R1_W=np.stack(R1_W),
             R1_divb=np.stack(R1_divb),
             R2_F_s=np.stack(R2_F_s),
             R2_W=np.stack(R2_W),
             R2_divb=np.stack(R2_divb),
             )
    print(f"Wrote: {outfile}")

def explore_stat_thresh(stat, outroot='_thresh.png', debug=False):

    outfile = stat+outroot

    # Load
    brazil_front_dict = np.load('../Analysis/brazil_kin_cutouts.npz')
    if stat == 'F_s':
        R1_stat = brazil_front_dict['R1_F_s']
        R2_stat = brazil_front_dict['R2_F_s']
        xlbl = r'$F_s$'
        mnmx = [2e-5, 1e-2]
    elif stat == 'divb':
        R1_stat = brazil_front_dict['R1_divb']
        R2_stat = brazil_front_dict['R2_divb']
        xlbl = r'$|\nabla b|^2$'
        mnmx = [1e-3, 2e-1]

    N_R1 = R1_stat.shape[0]

    # Thesholds
    threshes = 10**np.linspace(np.log10(mnmx[0]),
                                 np.log10(mnmx[1]), 20)
                                
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
        ax.plot(threshes, np.array(lim_dict[f'R2_{ilim}'])/N_R1,
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

def fig_brazil_front_stats(stat:str, outroot='fig_brazil_stats'):

    outfile = f'{outroot}_{stat}.png'
    brazil_front_dict = np.load('../Analysis/brazil_kin_cutouts.npz')

    # Load
    if stat == 'F_s':
        R1_stat = brazil_front_dict['R1_F_s']
        R2_stat = brazil_front_dict['R2_F_s']
        xlbl = r'$F_s$'
    elif stat == 'divb':
        R1_stat = brazil_front_dict['R1_divb']
        R2_stat = brazil_front_dict['R2_divb']
        xlbl = r'$|\nabla b|^2$'

    bins = np.linspace(R1_stat.min(), R1_stat.max(), 100)


    fig = plt.figure(figsize=(12,8))
    gs = gridspec.GridSpec(1,2)

    # R1
    ax_R1 = plt.subplot(gs[0])
    _ = sns.histplot(x=R1_stat.flatten(), bins=bins, log_scale=(False,True), ax=ax_R1)
    ax_R1.set_title('R1')

    ax_R2 = plt.subplot(gs[1])
    _ = sns.histplot(x=R2_stat.flatten(), bins=bins, log_scale=(False,True), ax=ax_R2,
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

# Command line execution
if __name__ == '__main__':
    #grab_brazil_cutouts()
    #brazil_pdfs()#debug=True)

    # F_s
    #fig_brazil_front_stats('F_s')
    # divb
    #fig_brazil_front_stats('divb')

    # Thresholds
    #explore_stat_thresh('F_s')
    explore_stat_thresh('divb')
