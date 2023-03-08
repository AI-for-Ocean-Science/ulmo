""" Figures related to the SSL paper but not quite """
import numpy as np
import scipy

from ulmo.utils import table as table_utils
from ulmo.ssl import figures



def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Check Nenya on VIIRS
    if flg & (2**0):
        viirs_file = '/data/Projects/Oceanography/AI/OOD/SST/VIIRS/Tables/VIIRS_Nenya_DT1.parquet'
        viirs = table_utils.load(viirs_file)

        # Plot
        binx=np.linspace(-1,10.5,30)
        biny=np.linspace(-3.5,4.5,30)
        metrics = ['LL', 'DT', 'stdDT', 'clouds', 'abslat', 'log10counts']
        figures.fig_umap_multi_metric(
            viirs, binx, biny,
            metrics=metrics,
            outfile='fig_nenya_viirs_multi_umap_DT1.png')

    if flg & (2**1):
        #u_extract_F_S('', debug=True, dlocal=True)  # debug
        u_extract_kin(full_fileA)


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        flg += 2 ** 0  # 1 -- Checking Nenya via multi figs
        #flg += 2 ** 1  # 2 -- Extract
    else:
        flg = sys.argv[1]

    main(flg)