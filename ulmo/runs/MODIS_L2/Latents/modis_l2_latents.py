""" Script to evaluate Latents for MODIS L2 """

import os
import numpy as np
from urllib.parse import urlparse

from ulmo import io as ulmo_io
from ulmo.analysis import evaluate as ulmo_evaluate 

from IPython import embed

tbl_file = 's3://modis-l2/Tables/MODIS_L2_std.parquet'


def u_evaluate(clobber_local=False):
    
    # Load
    modisl2_table = ulmo_io.load_main_table(tbl_file)

    # Evaluate
    ulmo_evaluate.eval_from_main(modisl2_table)

    # Write 
    ulmo_io.write_main_table(modisl2_table, tbl_file)

def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

        # MMT/MMIRS
    if flg & (2**0):
        u_evaluate()()


# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Evaluate
    else:
        flg = sys.argv[1]

    main(flg)