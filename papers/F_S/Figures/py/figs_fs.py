""" Figures related to the SSL paper but not quite """
import os
import numpy as np

import h5py

import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib import pyplot as plt
import seaborn as sns

from ulmo.utils import table as table_utils
from ulmo.nenya import figures

from IPython import embed

vx_dt = {'DT15': 1., 'DT1': 0.75, 'DT2': 1.5}

def load_tbl(survey:str, DT:str='DT1'):
    if survey == 'viirs':
        # VIIRS
        tbl_file = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Nenya', 'Tables', 
                                f'VIIRS_Nenya_{DT}.parquet')
    elif survey == 'viirs_on_viirs':
        tbl_file = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Nenya', 'Tables', 
                                f'VIIRS_Nenya_98clear_v1_{DT}.parquet')
    elif survey == 'llc_on_viirs':
        tbl_file = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Nenya', 'Tables', 
                                f'VIIRS_Nenya_LLC_{DT}.parquet')
    elif survey == 'viirs+llc_on_viirs':
        tbl_file = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Nenya', 'Tables', 
                                f'VIIRS_Nenya_VIIRS_LLC_{DT}.parquet')
    elif survey == 'llc':
        tbl_file = f'/data/Projects/Oceanography/AI/OOD/SST/LLC/Tables/LLC_A_Nenya_{DT}.parquet'
    elif survey == 'llc_on_llc':
        tbl_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Nenya', 'Tables', 
                                f'LLC_A_Nenya_v1_{DT}.parquet')
    elif survey == 'viirs_on_llc':
        tbl_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Nenya', 'Tables', 
                                f'LLC_A_Nenya_VIIRS_{DT}.parquet')
    elif survey == 'viirs+llc_on_llc':
        tbl_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Nenya', 'Tables', 
                                f'LLC_A_Nenya_VIIRS_LLC_{DT}.parquet')
    elif survey == 'modis':
        tbl_file = os.path.join(os.getenv('OS_AI'),
                                f'/data/Projects/Oceanography/AI/OOD/SST/LLC/Tables/LLC_A_Nenya_{DT}.parquet')
    #
    tbl = table_utils.load(tbl_file)
    return tbl

def counts_FS(outfile:str, table:str, subset:str, Ncut:int,
              cmap:str=None):
    # Load
    tbl = load_tbl(table, DT=subset)

    # F_S cut
    good = tbl.FS_Npos > Ncut
    cut_tbl = tbl[good]

    #embed(header='52 of figs')

    # Plot
    figures.umap_density(cut_tbl, ['US0', 'US1'], outfile=outfile, show_cbar=True,
                         normalize=False, cmap=cmap)


def fig_compare_T(outfile:str, nrand=50000, FS_thresh:float=1e-14):


    # Load files
    FS_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'F_S', 'PreProc',
                       'LLC_FS_preproc_Fs.h5')
    TT_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'F_S', 'PreProc',
                       'LLC_FS_preproc_T_SST.h5')
    SST_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'F_S', 'PreProc',
                       'LLC_FS_preproc.h5')

    f_SST = h5py.File(SST_file, 'r')                
    f_FS = h5py.File(FS_file, 'r')
    f_TT = h5py.File(TT_file, 'r')
    nimg = f_FS['valid'].shape[0]

    # Select a random set of images
    irand = np.random.choice(np.arange(nimg), size=nrand, replace=False)
    r_FS = f_FS['valid'][np.sort(irand), 0, ...]
    r_TT = f_TT['valid'][np.sort(irand), 0, ...]

    gd_FS = r_FS > FS_thresh

    ax = sns.histplot(y=r_FS[gd_FS], x=r_TT[gd_FS], log_scale=(True,True),
                      cbar=True)

    x0, y0 = 1e-3, 1e-14
    x1, y1 = x0 * 1e2, y0 * 1e2
    ax.plot([x0,x1], [y0,y1], ls='--', color='r')

    ax.set_xlabel(r'$T_{\rm SST}$')
    ax.set_ylabel(r'$F_S$')

    ax.set_xlim(1e-7, 1)


    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Wrote: {outfile}")
    

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Check Nenya with Multi
    if flg & (2**0):

        metrics = ['DT', 'stdDT', 'LL', 'clouds', 'abslat', 'log10counts']

        # VIIRS
        #tbl = load_tbl('viirs')
        #outfile='fig_nenya_viirs_multi_umap_DT1.png'

        #tbl = load_tbl('viirs_on_viirs')
        #outfile='fig_nenya_viirs2_multi_umap_DT1.png'

        # LLC
        #tbl = load_tbl('llc')
        #outfile='fig_nenya_llcA_multi_umap_DT1.png'
        #metrics = ['DT', 'stdDT', 'abslat', 'log10counts']

        #subsets =  ['DT15', 'DT1', 'DT2']
        subsets =  ['DT2']
        for subset in subsets:
            #tbl = load_tbl('viirs_on_llc', DT=subset)
            #outfile= f'fig_nenya_llc_viirs_multi_umap_{subset}.png'
            #outfile= f'fig_nenya_llc_viirs+llc_multi_umap_{subset}.png'
            table = 'llc_on_llc'
            tbl = load_tbl(table, DT=subset)
            outfile= f'fig_nenya_llc2_multi_umap_{subset}.png'
            metrics = ['DT', 'log10FS_pos_sum', 'meanT', 
                       'log10FS_Npos', 'abslat', 'log10counts']

            # LLC with VIIRS UMAP (DT1)
            if subset == 'DT1':
                binx=np.linspace(2,10.5,30)
                biny=np.linspace(3.5,11.5,30)
            elif subset == 'DT15':
                if table == 'llc_on_llc':
                    binx=np.linspace(-4, 10, 30)
                    biny=np.linspace(5,13,30)
                else:
                    binx=np.linspace(1.5,11.5,30)
                    biny=np.linspace(3.5,11.5,30)
            elif subset == 'DT2':
                if table == 'llc_on_llc':
                    binx=np.linspace(-4,5.,30)
                    biny=np.linspace(1,14.,30)
                else:
                    binx=np.linspace(0.,9.5,30)
                    biny=np.linspace(1,10.,30)
                

            # Plot
            # MODIS UMAP
            #binx=np.linspace(-1,10.5,30)
            #biny=np.linspace(-3.5,4.5,30)

            # VIIRS UMAP
            #binx=np.linspace(1,13,30)
            #biny=np.linspace(3.5,11.5,30)
            
            figures.umap_multi_metric(
                tbl, binx, biny,
                metrics=metrics,
                outfile=outfile)

    # Galleries
    if flg & (2**1):
        '''
        # VIIRS with MODIS
        viirs = load_tbl('viirs')
        figures.umap_gallery(viirs, 'fig_nenya_viirs_gallery_DT1.png',
                             local=os.path.join(os.getenv('OS_SST'), 'VIIRS'),
                             in_vmnx=[-0.75, 0.75])

        
        # VIIRS with VIIRS
        subsets =  ['DT1', 'DT15']
        for subset in subsets:
            viirs = load_tbl('viirs_on_viirs', DT=subset)
            figures.umap_gallery(
                viirs, f'fig_nenya_viirs2_gallery_{subset}.png',
                local=os.path.join(os.getenv('OS_SST'), 'VIIRS'),
                in_vmnx=[-vx_dt[subset], vx_dt[subset]])

        # VIIRS with VIIRS+LLC
        subsets =  ['DT15']
        for subset in subsets:
            viirs = load_tbl('viirs_on_viirs+llc', DT=subset)
            figures.umap_gallery(
                viirs, f'fig_nenya_viirs_viirs+llc_gallery_{subset}.png',
                local=os.path.join(os.getenv('OS_SST'), 'VIIRS'),
                in_vmnx=[-vx_dt[subset], vx_dt[subset]])
        # VIIRS with VIIRS
        subsets =  ['DT15', 'DT1']
        for subset in subsets:
            viirs = load_tbl('llc_on_viirs', DT=subset)
            figures.umap_gallery(
                viirs, f'fig_nenya_llc_on_viirs_gallery_{subset}.png',
                local=os.path.join(os.getenv('OS_SST'), 'VIIRS'),
                in_vmnx=[-vx_dt[subset], vx_dt[subset]])
        '''

        # LLC with LLC
        subsets =  ['DT2'] #'DT15', 'DT1']#, 'DT2']
        for subset in subsets:
            tbl = load_tbl('llc_on_llc', DT=subset)
            figures.umap_gallery(
                tbl, f'fig_nenya_llc2_gallery_{subset}.png',
                local=os.path.join(os.getenv('OS_OGCM'), 'LLC', 'F_S'),
                in_vmnx=[-vx_dt[subset], vx_dt[subset]])
        '''
        # LLC with VIIRS
        subsets =  ['DT15', 'DT1', 'DT2']
        for subset in subsets:
            tbl = load_tbl('viirs_on_llc', DT=subset)
            figures.umap_gallery(
                tbl, 
                f'fig_nenya_llc_viirs_gallery_{subset}.png',
                local=os.path.join(os.getenv('OS_OGCM'), 'LLC', 'F_S'),
                in_vmnx=[-vx_dt[subset], vx_dt[subset]])

        # LLC with VIIRS
        subsets =  ['DT15']
        for subset in subsets:
            tbl = load_tbl('viirs+llc_on_llc', DT=subset)
            figures.umap_gallery(
                tbl, 
                f'fig_nenya_llc_viirs+llc_gallery_{subset}.png',
                local=os.path.join(os.getenv('OS_OGCM'), 'LLC', 'F_S'),
                in_vmnx=[-vx_dt[subset], vx_dt[subset]])
        '''

    # Show counts of FS
    if flg & (2**2):
        subset = 'DT2'
        counts_FS(f'FS_counts_LLC_LLC_{subset}.png', 'llc_on_llc', subset, 10,
                  cmap='Blues')

    # Tendency vs Tendency
    if flg & (2**3):
        fig_compare_T('fig_compare_T.png', nrand=50000)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Checking Nenya via multi figs
        #flg += 2 ** 1  # 2 -- Galleries
        #flg += 2 ** 2  # 4 -- F_S counts
        #flg += 2 ** 3  # 8 -- Tendency vs Tendency
    else:
        flg = sys.argv[1]

    main(flg)