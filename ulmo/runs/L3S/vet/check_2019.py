import pandas as pd
import ulmo
import os
from ulmo.analysis import evaluate as ulmo_evaluate 
import datetime
from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils

def check_files():
    l3s_viirs_tbl_file = 's3://sst-l3s/Tables/SST_L3S_VIIRS.parquet'

    l3s_table = ulmo.io.load_main_table(l3s_viirs_tbl_file)

    files_2019 = l3s_table[l3s_table['VIIRS_datetime'].dt.year == 2019]['VIIRS_UID']

    for file in files_2019:
        # Check if the file exists
        if os.path.exists(file):
            # Read the file
            with open(file, 'r'):
                data = ulmo.io.load_nc(file)
                
                # Check if the file has data
                has_data = False
                if data:
                    has_data = True

                print(f"File: {file}\nHas Data: {has_data}\n")
        else:
            print(f"File does not exist: {file}\n")

check_files()






