""" Code related to fronto genesis calculations"""
import numpy as np

from matplotlib import pyplot as plt

from ulmo.llc import io as llc_io
from ulmo import io as ulmo_io
from ulmo.llc import kinematics

from IPython import embed

def grab_brazil_cutouts(DT=2.05, dDT=0.05):
    # Load LLC
    tbl_test_noise_file = 's3://llc/Tables/test_noise_modis2012.parquet'
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

    return evals_bz, idx_R1, idx_R2


def brazil_pdfs(outfile='F_S_pdfs.npz', debug=False):
    """ Generate the PDFs of F_S for DT ~ 1K cutouts
    in the Brazil-Malvanis confluence
    """
    evals_bz, idx_R1, idx_R2 = grab_brazil_cutouts(dDT=0.25) # Higher dDT for stats

    # R1 first
    R1_F_s = []
    for kk, iR1 in enumerate(idx_R1):
        if debug and kk > 1:
            continue
        print(f'R1: {kk} of {idx_R1.size}')
        cutout = evals_bz.iloc[iR1]
        # Load 
        U, V, SST, Salt = llc_io.grab_velocity(
            cutout, add_SST=True, add_Salt=True)
        # Calculate
        F_s = kinematics.calc_F_s(U.data, V.data, 
                SST.data, Salt.data)
        R1_F_s.append(F_s)

    # R2 first
    R2_F_s = []
    for kk, iR2 in enumerate(idx_R2):
        if debug and kk > 1:
            continue
        print(f'R2: {kk} of {idx_R2.size}')
        cutout = evals_bz.iloc[iR2]
        # Load 
        U, V, SST, Salt = llc_io.grab_velocity(
            cutout, add_SST=True, add_Salt=True)
        # Calculate
        F_s = kinematics.calc_F_s(U.data, V.data, 
                SST.data, Salt.data)
        R2_F_s.append(F_s)

    # Output
    np.savez(outfile, R1_F_s=np.stack(R1_F_s),
             R2_F_s=np.stack(R2_F_s))
    print(f"Wrote: {outfile}")

def explore_F_S_thresh(outfile='F_S_thresh.png', debug=False):
    # Load
    fs_dict = np.load('F_S_pdfs.npz')

    R1_F_s = fs_dict['R1_F_s']
    N_R1 = R1_F_s.shape[0]
    R2_F_s = fs_dict['R2_F_s']

    # F_s_thresh
    mnmx = [2e-5, 1e-2]
    F_S_threshes = 10**np.linspace(np.log10(mnmx[0]),
                                 np.log10(mnmx[1]), 20)
                                
    # Prep
    nlim = [1, 3, 10, 30]
    lss = [':', '--', '-', '-.']
    lim_dict = {}
    for ilim in nlim:
        lim_dict[f'R1_{ilim}'] = []
        lim_dict[f'R2_{ilim}'] = []

    # Loop
    for F_S_thresh in F_S_threshes:
        # R1
        gd_R1 = R1_F_s > F_S_thresh
        ngd_R1 = np.sum(gd_R1, axis=(1,2))
        # R2
        gd_R2 = R2_F_s > F_S_thresh
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
        ax.plot(F_S_threshes, np.array(lim_dict[f'R1_{ilim}'])/N_R1,
                label=f'R1_{ilim}', ls=lss[kk], color='blue')
        # R2
        if ilim == nlim[0]:
            R2_lbl = f'R2_{ilim}'
        else:
            R2_lbl = None
        ax.plot(F_S_threshes, np.array(lim_dict[f'R2_{ilim}'])/N_R1,
                label=R2_lbl, ls=lss[kk], color='red')
    # 
    ax.set_xscale('log')
    ax.legend(fontsize=15.)
    ax.set_xlabel("F_S Threshold")
    ax.set_ylabel("Fraction of Images")

    #
    plt.savefig(outfile, dpi=200)
    print(f"Wrote: {outfile}")
    #plt.show()


# Command line execution
if __name__ == '__main__':
    #grab_brazil_cutouts()
    #brazil_pdfs()#debug=True)
    explore_F_S_thresh()