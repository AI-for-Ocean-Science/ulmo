""" Figures related to the SSL paper but not quite """
import os
import numpy as np
import scipy

from ulmo.utils import table as table_utils
from ulmo.ssl import figures


def load_viirs():
    # VIIRS
    viirs_file = '/data/Projects/Oceanography/AI/OOD/SST/VIIRS/Tables/VIIRS_Nenya_DT1.parquet'
    viirs = table_utils.load(viirs_file)
    return viirs

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Check Nenya with Multi
    if flg & (2**0):


        viirs = load_viirs()
        # Plot
        binx=np.linspace(-1,10.5,30)
        biny=np.linspace(-3.5,4.5,30)
        metrics = ['LL', 'DT', 'stdDT', 'clouds', 'abslat', 'log10counts']
        figures.umap_multi_metric(
            viirs, binx, biny,
            metrics=metrics,
            outfile='fig_nenya_viirs_multi_umap_DT1.png')

    # Galleries
    if flg & (2**1):
        viirs = load_viirs()
        figures.umap_gallery(viirs, 'fig_nenya_viirs_gallery_DT1.png',
                             local=os.path.join(os.getenv('SST_OOD'), 'VIIRS'),
                             in_vmnx=[-0.75, 0.75])


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Checking Nenya via multi figs
        flg += 2 ** 1  # 2 -- Galleries
    else:
        flg = sys.argv[1]

    main(flg)