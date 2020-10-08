""" Result methods"""

import glob, os
import pandas
import datetime

from ulmo import defs

from IPython import embed

def load_log_prob(pproc):

    # Load up the tables
    table_files = glob.glob(os.path.join(defs.eval_path, 'R2010_on*{}_log_prob.csv'.format(pproc)))

    # Cut down?
    # table_files = table_files[0:2]

    ioff = 10
    evals_tbl = pandas.DataFrame()
    for table_file in table_files:
        print("Loading: {}".format(table_file))
        df = pandas.read_csv(table_file)
        # Dates
        dtimes = [datetime.datetime(int(ifile[1+ioff:5+ioff]),
                                    int(ifile[5+ioff:7+ioff]),
                                    int(ifile[7 + ioff:9+ioff]),
                                    int(ifile[10+ioff:12+ioff]),
                                    int(ifile[12+ioff:14+ioff]))
                  for ifile in df['filename'].values]
        #except:
        #    embed(header='32 of results')
        df['date'] = dtimes
        evals_tbl = pandas.concat([evals_tbl, df])

    print('NEED TO ADD IN 2010!!!')

    # Return
    return evals_tbl
