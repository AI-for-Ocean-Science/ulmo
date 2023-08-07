''' Helper routines for Kinematic analysis'''

import numpy as np

import pandas

def grab_brazil_cutouts(llc_table:pandas.DataFrame, DT=2.05, dDT=0.05):

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

    print(f"We have {idx_R1.size} cutouts in R1 and {idx_R2.size} in R2")

    return evals_bz, idx_R1, idx_R2
