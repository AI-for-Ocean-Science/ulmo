""" Definitons for OOD analysis """

import os

import pandas

# MODIS L2
if os.getenv('SST_OOD') is not None:
    modis_extract_path = os.path.join(os.getenv("SST_OOD"), 'MODIS_L2', 'Extractions')
    modis_model_path = os.path.join(os.getenv("SST_OOD"), 'MODIS_L2', 'Models')
    modis_eval_path = os.path.join(os.getenv("SST_OOD"), 'MODIS_L2', 'Evaluations')

# Main Table definitions
mtbl_dmodel = {
    'field_size': dict(dtype=int,
                help='Size of the cutout side (pixels)'),
    'lat': dict(dtype=float,
                help='Latitude of lower-left corner of the cutout (deg)'),
    'lon': dict(dtype=float,
                help='Longitude of lower-left corner of the cutout (deg)'),
    'col': dict(dtype=int,
                help='Column of lower-left corner of the cutout'),
    'row': dict(dtype=int,
                help='Row of lower-left corner of the cutout'),
    'filename': dict(dtype=str,
                help='Filename of the original file from which the cutout was extracted'),
    'datetime': dict(dtype=pandas.Timestamp,
                help='Row of lower-left corner of the cutout'),
    'LL': dict(dtype=float,
                help='Log-likelihood of the cutout from Ulmo'),
    'clear_fraction': dict(dtype=float,
                help='Fraction of the cutout clear from clouds'),
    'mean_temperature': dict(dtype=float,
                help='Average SST of the cutout'),
    'Tmin': dict(dtype=float,
                help='Minimum T of the cutout'),
    'Tmax': dict(dtype=float,
                help='Maximum T of the cutout'),
    'T10': dict(dtype=float,
                help='10th percentile of T of the cutout'),
    'T90': dict(dtype=float,
                help='90th percentile of T of the cutout'),
    'pp_root': dict(dtype=str,
                help='Describes the pre-processing steps applied'),
    'pp_file': dict(dtype=str,
                help='Filename of the pre-processed file holding the cutout'),
    'pp_idx': dict(dtype=int, 
                help='Index describing position of the cutout in the pp_file'),
    'pp_type': dict(dtype=int, allowed=(-1, 0,1), 
                    valid=0, train=1, init=-1,
                    help='-1: illdefined, 0: valid, 1: test'),
    'UID': dict(dtype=int, 
                help='Unique identifier generated for each cutout'),
    }