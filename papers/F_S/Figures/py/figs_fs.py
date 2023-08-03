""" Figures related to the SSL paper but not quite """
import os
import numpy as np

from ulmo.utils import table as table_utils
from ulmo.nenya import figures

from IPython import embed


def load_tbl(survey:str, DT:str='DT1'):
    if survey == 'viirs':
        # VIIRS
        tbl_file = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Nenya', 'Tables', 
                                f'VIIRS_Nenya_{DT}.parquet')
    elif survey == 'viirs_on_viirs':
        tbl_file = os.path.join(os.getenv('OS_SST'), 'VIIRS', 'Nenya', 'Tables', 
                                f'VIIRS_Nenya_98clear_v1_{DT}.parquet')
    elif survey == 'llc':
        tbl_file = f'/data/Projects/Oceanography/AI/OOD/SST/LLC/Tables/LLC_A_Nenya_{DT}.parquet'
    elif survey == 'llc_on_llc':
        tbl_file = os.path.join(os.getenv('OS_OGCM'), 'LLC', 'Nenya', 'Tables', 
                                f'LLC_A_Nenya_v1_{DT}.parquet')
    elif survey == 'modis':
        tbl_file = os.path.join(os.getenv('OS_AI'),
                                f'/data/Projects/Oceanography/AI/OOD/SST/LLC/Tables/LLC_A_Nenya_{DT}.parquet')
    #
    tbl = table_utils.load(tbl_file)
    return tbl

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

        tbl = load_tbl('viirs_on_viirs')
        outfile='fig_nenya_viirs2_multi_umap_DT1.png'

        # LLC
        #tbl = load_tbl('llc')
        #outfile='fig_nenya_llcA_multi_umap_DT1.png'
        #metrics = ['DT', 'stdDT', 'abslat', 'log10counts']

        # Plot
        # MODIS
        #binx=np.linspace(-1,10.5,30)
        #biny=np.linspace(-3.5,4.5,30)

        # VIIRS
        binx=np.linspace(1,13,30)
        biny=np.linspace(3.5,11.5,30)
        
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
        '''

        # VIIRS with VIIRS
        viirs = load_tbl('viirs_on_viirs')
        figures.umap_gallery(viirs, 'fig_nenya_viirs2_gallery_DT1.png',
                             local=os.path.join(os.getenv('OS_SST'), 'VIIRS'),
                             in_vmnx=[-0.75, 0.75])

        '''
        # LLC with LLC
        viirs = load_tbl('llc_on_llc')
        figures.umap_gallery(viirs, 'fig_nenya_llc2_gallery_DT1.png',
                             local=os.path.join(os.getenv('OS_OGCM'), 'LLC'),
                             in_vmnx=[-0.75, 0.75])
        '''


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        flg += 2 ** 0  # 1 -- Checking Nenya via multi figs
        #flg += 2 ** 1  # 2 -- Galleries
    else:
        flg = sys.argv[1]

    main(flg)