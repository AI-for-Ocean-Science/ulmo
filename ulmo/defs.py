""" Definitons for OOD analysis """

import os
import numpy as np

import pandas

# MODIS L2
if os.getenv('SST_OOD') is not None:
    modis_extract_path = os.path.join(os.getenv("SST_OOD"), 'MODIS_L2', 'Extractions')
    modis_model_path = os.path.join(os.getenv("SST_OOD"), 'MODIS_L2', 'Models')
    modis_eval_path = os.path.join(os.getenv("SST_OOD"), 'MODIS_L2', 'Evaluations')

# Main Table definitions
mtbl_dmodel = {
    'field_size': dict(dtype=(int, np.integer),
                help='Size of the cutout side (pixels)'),
    'lat': dict(dtype=(float,np.floating),
                help='Latitude of lower-left corner of the cutout (deg)'),
    'lon': dict(dtype=(float,np.floating),
                help='Longitude of lower-left corner of the cutout (deg)'),
    'col': dict(dtype=(int, np.integer),
                help='Column of lower-left corner of the cutout'),
    'row': dict(dtype=(int, np.integer),
                help='Row of lower-left corner of the cutout'),
    'filename': dict(dtype=str,
                help='Filename of the original file from which the cutout was extracted'),
    'ex_filename': dict(dtype=str,
                help='Filename of the extraction file holding the cutouts'),
    'datetime': dict(dtype=pandas.Timestamp,
                help='Row of lower-left corner of the cutout'),
    'LL': dict(dtype=(float,np.floating),
                help='Log-likelihood of the cutout from Ulmo'),
    'clear_fraction': dict(dtype=float,
                help='Fraction of the cutout clear from clouds'),
    'mean_temperature': dict(dtype=(float,np.floating),
                help='Average SST of the cutout'),
    'Tmin': dict(dtype=(float,np.floating),
                help='Minimum T of the cutout'),
    'Tmax': dict(dtype=(float,np.floating),
                help='Maximum T of the cutout'),
    'T10': dict(dtype=(float,np.floating),
                help='10th percentile of T of the cutout'),
    'T90': dict(dtype=(float,np.floating),
                help='90th percentile of T of the cutout'),
    'pp_root': dict(dtype=str,
                help='Describes the pre-processing steps applied'),
    'pp_file': dict(dtype=str,
                help='Filename of the pre-processed file holding the cutout'),
    'pp_idx': dict(dtype=(int,np.integer), 
                help='Index describing position of the cutout in the pp_file'),
    'pp_type': dict(dtype=(int, np.integer), allowed=(-1, 0,1), 
                    valid=0, train=1, init=-1,
                    help='-1: illdefined, 0: valid, 1: train'),
                    # In Ulmo, we use 1 for the subset of training and 0 for the rest
                    # In SSL, we use 1 for train, 0 for validation and -1 for the rest [but not always]
    'zonal_slope': dict(dtype=(float,np.floating),
                help='Power-law spectra slope in the zonal direction'),
    'zonal_slope_err': dict(dtype=(float,np.floating),
                help='Error in Power-law spectra slope in the zonal direction'),
    'merid_slope': dict(dtype=(float,np.floating),
                help='Power-law spectra slope in the meridianal direction'),
    'merid_slope_err': dict(dtype=(float,np.floating),
                help='Error in Power-law spectra slope in the meridianal direction'),
    'U0': dict(dtype=(float,np.floating),
                help='UMAP 0th coefficient'),
    'U1': dict(dtype=(float,np.floating),
                help='UMAP 1st coefficient'),
    'US0': dict(dtype=(float,np.floating),
                help='UMAP 0th coefficient for DT subset'),
    'US1': dict(dtype=(float,np.floating),
                help='UMAP 1st coefficient for DT subset'),
    'UT1_0': dict(dtype=(float,np.floating),
                help='UMAP 0th coefficient for DT=1 UMAP model'),
    'UT1_1': dict(dtype=(float,np.floating),
                help='UMAP 1st coefficient for DT=1 UMAP model'),
    'U1': dict(dtype=(float,np.floating),
                help='UMAP 1st coefficient'),
    'U3_0': dict(dtype=(float,np.floating),
                help='UMAP 0th coefficient for ndim=3'),
    'U3_1': dict(dtype=(float,np.floating),
                help='UMAP 1st coefficient for ndim=3'),
    'U3_2': dict(dtype=(float,np.floating),
                help='UMAP 2nd coefficient for ndim=3'),
    'UID': dict(dtype=(int, np.integer),
                help='Unique identifier generated for each cutout'),
    # KINEMATICS
    'gradb_Npos': dict(dtype=(int, np.integer),
                help='Number of pixels exceeding the |grad b|^2 threshold'),
    'FS_Npos': dict(dtype=(int, np.integer),
                help='Number of pixels exceeding the F_S threshold'),
    'FS_pos_sum': dict(dtype=(float,np.floating),
                help='Sum of F_s for pixels with F_S>0'),
    # REQUIRED
    'required': ('lat', 'lon', 'row', 'col', 'datetime')
    }
