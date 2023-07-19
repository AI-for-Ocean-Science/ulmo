"""
l3s_viirs_tbl_file = 's3://sst-l3s/Tables/SST_L3S_VIIRS.parquet'
l3s_table = ulmo.io.load_main_table(l3s_viirs_tbl_file)
dates_2019 = l3s_table[l3s_table['VIIRS_datetime'].dt.year == 2019]['VIIRS_datetime'].dt.date.unique()

corrupted_dates = []  # List to store corrupted dates

for date in dates_2019:
    # Select rows for the current date
    files_date = l3s_table[l3s_table['VIIRS_datetime'].dt.date == date]

    # Check for missing or NaN values in each column for the current date
    if files_date.isnull().values.any():
        corrupted_dates.append(date)

# Print the corrupted dates
for date in corrupted_dates:
    print(f"Corrupted file on {date}")

print(corrupted_dates)



# so there's no missing values in the general table, I need to be able to open the hard drive files
# and see if the temperature data is missing

import xarray
import shutil

dates_2019 = l3s_table[l3s_table['VIIRS_datetime'].dt.year == 2019]['VIIRS_datetime'].dt.date.unique()

corrupted_dates = []  # List to store corrupted dates

for date in dates_2019:
    # Select rows for the current date
    files_date = l3s_table[l3s_table['VIIRS_datetime'].dt.date == date]

    for index, row in files_date.iterrows():
        file = row['ex_filename']

        try:
            # Copy the file to your desktop
            destination_file = os.path.expanduser("~/Desktop/" + os.path.basename(file))
            shutil.copy(file, destination_file)

            # Open the copied .nc file using xarray
            ds = xarray.open_dataset(destination_file)

            # Check if 'sea_surface_temperature' variable exists
            if 'sea_surface_temperature' in ds:
                sst = ds['sea_surface_temperature']

                # Calculate the mean temperature
                if numpy.isnan(sst).any():
                    print(f"Missing temperature data in {file}")
                else:
                    mean_temp = sst.mean().item()
                    print(f"Mean sea_surface_temperature for {file}: {mean_temp:.2f}°C")
            else:
                print(f"'sea_surface_temperature' variable not found in {file}")

        except Exception as e:
            corrupted_dates.append(date)
            print(f"Error processing file on {date}: {file} - {str(e)}")

        finally:
            # Close the file
            #ds.close()

            # Delete the copied file from your desktop
            os.remove(destination_file)

# Print the corrupted dates
for date in corrupted_dates:
    print(f"Corrupted file on {date}")

print(corrupted_dates)


files_date = l3s_table[l3s_table['VIIRS_datetime'].dt.date == pd.Timestamp('2019-02-11')]
file = files_date['ex_filename']
destination_file = os.path.expanduser("~/Desktop/" + os.path.basename(file.values[0]))
shutil.copy(str(file), str(destination_file))
"""

""" module for vet of the SST L3S dataset
"""
import os
import numpy as np

import time
import h5py
import numpy as np
import argparse

import pandas
import datetime
#from datetime import datetime, timedelta

from matplotlib import pyplot as plt
import seaborn as sns

from ulmo.analysis import evaluate as ulmo_evaluate 
from ulmo import io as ulmo_io
from ulmo.utils import catalog as cat_utils

from ulmo.sst_l3s import extract


from IPython import embed

l3s_viirs_tbl_file = 's3://sst-l3s/Tables/SST_L3S_VIIRS.parquet'
viirs_tbl_file = 's3://viirs/Tables/VIIRS_all_98clear_std.parquet'

def init_l3s_tbl():
    # Load VIIRS table
    viirs = ulmo_io.load_main_table(viirs_tbl_file)

    # Copy
    l3s = pandas.DataFrame()

    # Save VIIRS info
    viirs_keys = ['row', 'col', 'UID', 'LL', 'pp_file', 
                'pp_idx', 'T90', 'T10', 'DT', 'pp_type',
                'Tmin', 'Tmax',
                'clear_fraction', 'datetime', 'filename',
                'ex_filename', 'lat', 'lon']
    # Generate the L3S table
    for key in viirs_keys:
        l3s[f'VIIRS_{key}'] = viirs[key]

    # Add L3S data
    l3s['row'] = ((90 - l3s['VIIRS_lat']) * (9000 / 180)).astype(int)
    l3s['col'] = ((l3s['VIIRS_lon'] + 180) * (18000 / 360)).astype(int)
    l3s['lat'] = l3s['VIIRS_lat']
    l3s['lon'] = l3s['VIIRS_lon']

    l3s['viirs_date'] = l3s['VIIRS_datetime'].dt.date
    desired_time = datetime.time(1, 30, 0)
    l3s['combined_datetime'] = pandas.to_datetime(l3s['viirs_date'].astype(str) + ' ' + str(desired_time))
    l3s['datetime'] = (l3s['combined_datetime'] + pandas.to_timedelta(l3s['VIIRS_lon'] * 4, unit='minutes')).dt.round('S')

    l3s['viirs_date'] = pandas.to_datetime(l3s['VIIRS_datetime'].dt.date.astype(str) + ' ' + str(desired_time))
    l3s['viirs_date_minus1'] = pandas.to_datetime((l3s['VIIRS_datetime'].dt.date - datetime.timedelta(days=1)).astype(str) + ' ' + str(desired_time))
    l3s['viirs_date_plus1'] = pandas.to_datetime((l3s['VIIRS_datetime'].dt.date + datetime.timedelta(days=1)).astype(str) + ' ' + str(desired_time))

    time_threshold = pandas.Timedelta(hours=12)

    if (l3s['datetime'] - l3s['viirs_date_minus1']).apply(lambda x: x < time_threshold).any():
        l3s['date'] = l3s['viirs_date_minus1'].dt.date
    elif (l3s['datetime'] - l3s['viirs_date']).apply(lambda x: x < time_threshold).any():
        l3s['date'] = l3s['viirs_date'].dt.date
    elif (l3s['datetime'] - l3s['viirs_date_plus1']).apply(lambda x: x < time_threshold).any():
        l3s['date'] = l3s['viirs_date_plus1'].dt.date
    else:
        l3s['date'] = l3s['viirs_date'].dt.date

    l3s['ex_filename'] = (
        '/Volumes/Aqua-1/Hackathon/daily/l3s_fields/' +
        l3s['VIIRS_datetime'].dt.strftime('%Y/%j/%Y%m%d') +
        '120000-STAR-L3S_GHRSST-SSTsubskin-LEO_Daily-ACSPO_V2.80-v02.0-fv01.0.nc')
    l3s = l3s.drop(['viirs_date', 'combined_datetime', 'viirs_date_minus1', 'viirs_date_plus1', 'date'], axis=1)

    # Check the table -- it should complain about missing required keys
    cat_utils.vet_main_table(l3s, cut_prefix='VIIRS_')

    # Write
    ulmo_io.write_main_table(l3s, l3s_viirs_tbl_file)

# EXTRACTION
def l3s_viirs_extract(tbl_file:str, 
                      year:int,
                      root_file=None, 
                      preproc_root='l3s_viirs', 
                      debug=False):
    """ Perform the extraction for the L3S dataset

    Args:
        tbl_file (str): table file (s3)
        year (int): Year to analyze
        root_file (_type_, optional): 
            Output filename. Defaults to None.
        preproc_root (str, optional): 
            Defines the options for pre-processing. 
            Defaults to 'l3s_viirs'.
        debug (bool, optional): If True, perform
            a limited extraction as a test. Defaults to False.
    """

    # Giddy up (will take a bit of memory!)
    l3s_table = ulmo_io.load_main_table(tbl_file)
    # Check indices
    l3s_table.reset_index(drop=True, inplace=True)
    assert np.all(np.arange(len(l3s_table)) == l3s_table.index)

    # New table file``
    new_tbl_file = tbl_file.replace('.parquet', f'_{year}.parquet')

    if debug:
        # Cut down to the first month
        gd_date = l3s_table.datetime <= datetime.datetime(2012,2,2)
        l3s_table = l3s_table[gd_date]
        debug_local = True

    # Cut on year
    gd_date = (l3s_table.VIIRS_datetime >= datetime.datetime(year,1,1)) & (
        l3s_table.VIIRS_datetime < datetime.datetime(year+1,1,1)) 
    l3s_table_year = l3s_table[gd_date]
    l3s_table_year.reset_index(drop=True, inplace=True)
    print(f"Running on year={year} with {len(l3s_table_year)} rows")

    if debug:
        root_file = 'L3S_VIIRS144_test_preproc.h5'
    else:
        if root_file is None:
            root_file = f'L3S_VIIRS144_{year}_preproc.h5'
            debug_local = False

    # Setup
    pp_local_file = 'PreProc/'+root_file
    pp_s3_file = 's3://sst-l3s/PreProc/'+root_file
    if not os.path.isdir('PreProc'):
        os.mkdir('PreProc')

    print(f"Outputting to: {pp_s3_file}")

    # Run it
    if debug_local:
        pp_s3_file = 's3://sst-l3s/PreProc/tst.h5'

    # Do it
    #embed(header='210 of llc viirs')
    extract.preproc_for_analysis(l3s_table_year, 
                                 pp_local_file,
                                 preproc_root=preproc_root,
                                 s3_file=pp_s3_file,
                                 override_RAM=True)
    # Vet
    assert cat_utils.vet_main_table(l3s_table_year, cut_prefix=['VIIRS_'])

    # Final write
    if debug:
        ulmo_io.write_main_table(l3s_table_year, 'tmp.parquet', to_s3=False)
    else:
        ulmo_io.write_main_table(l3s_table_year, new_tbl_file)
    print("You should probably remove the PreProc/ folder")
    
def l3s_ulmo(year:int, debug=False, model='viirs-98'):
    """Evaluate the L3S data using Ulmo

    Args:
        debug (bool, optional): [description]. Defaults to False.
        model (str, optional): [description]. Defaults to 'viirs-98'.
    """
    new_tbl_file = l3s_viirs_tbl_file.replace('.parquet', f'_{year}.parquet')

    # Load Table
    l3s_tbl = ulmo_io.load_main_table(new_tbl_file)

    # Evaluate
    print("Starting evaluating..")
    viirs_tbl = ulmo_evaluate.eval_from_main(l3s_tbl, model=model,
                                             local=True)

    # Write 
    assert cat_utils.vet_main_table(l3s_tbl, cut_prefix=['VIIRS_'])
    ulmo_io.write_main_table(l3s_tbl, new_tbl_file)
    print("Done evaluating..")


def main(flg):
    if flg== 'all':
        flg= np.sum(np.array([2 ** ii for ii in range(25)]))
    else:
        flg= int(flg)

    # Extract the L3S images
    if flg & (2**1):
        #l3s_viirs_extract(l3s_viirs_tbl_file, debug=False)

        # Try 2012
        l3s_viirs_extract(l3s_viirs_tbl_file, 2019, debug=False)

    # Run Ulmo
    if flg & (2**2):

        # Try 2012
        l3s_ulmo(2019, debug=False)

# Command line execution
if __name__ == '__main__':
    import sys

    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # 1 -- Init L3S table
    else:
        flg = sys.argv[1]

    main(flg)

# Generate the table
# python -u sst_l3s_vet.py 1

# Extract
# python -u sst_l3s_vet.py 2

# Ulmo
# python -u sst_l3s_vet.py 4
