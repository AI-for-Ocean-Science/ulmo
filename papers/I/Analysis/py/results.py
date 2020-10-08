""" Result methods"""

import glob, os
import pandas

from ulmo import defs

def load_log_prob(pproc):

    # Load up the tables
    table_files = glob.glob(os.path.join(defs.eval_path, 'R2010_on*{}_log_prob.csv'.format(pproc)))

    # Cut down?
    # table_files = table_files[0:2]

    evals_tbl = pandas.DataFrame()
    for table_file in table_files:
        print("Loading: {}".format(table_file))
        df = pandas.read_csv(table_file)
        evals_tbl = pandas.concat([evals_tbl, df])

    print('NEED TO ADD IN 2010!!!')

    # Return
    return evals_tbl
