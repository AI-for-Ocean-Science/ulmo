""" Run UMAP on subsets of the data (DT) """
import os
from pkg_resources import resource_filename

import ssl_paper_analy

opt_path_CF = os.path.join(resource_filename('ulmo', 'runs'),
                        'SSL', 'MODIS', 'v2', 
                        'experiments', 'modis_model_v2', 'opts_cloud_free.json')
opt_path_96 = os.path.join(resource_filename('ulmo', 'runs'),
                        'SSL', 'MODIS', 'v3', 
                        'opts_96clear_ssl.json')

def run_subset(subset, remove=True, CF=False):                    
    if subset == 'DT0':
        DT_cut = (0.25, 0.25)
    elif subset == 'DT10':
        DT_cut = (1.0, 0.05)
    elif subset == 'DT1':
        DT_cut = (0.75, 0.25)
    elif subset == 'DT15':
        DT_cut = (1.25, 0.25)
    elif subset == 'DT2':
        DT_cut = (2.0, 0.5)
    elif subset == 'DT4':
        DT_cut = (3.25, 0.75)
    elif subset == 'DT5':
        DT_cut = (5.0, -1)
    elif subset == 'all':
        DT_cut = None

    # Prep
    if CF:
        base1 = 'cloud_free'
        opt_path = opt_path_CF
    else:
        base1 = '96clear'
        opt_path = opt_path_96
    outfile = os.path.join(
        os.getenv('SST_OOD'), 
        f'MODIS_L2/Tables/MODIS_SSL_{base1}_{subset}.parquet')
    umap_savefile = os.path.join(
        os.getenv('SST_OOD'), 
        f'MODIS_L2/UMAP/MODIS_SSL_{base1}_{subset}_UMAP.pkl')

    # Run
    ssl_paper_analy.umap_subset(opt_path, outfile, 
                                DT_cut=DT_cut, debug=False,
                                umap_savefile=umap_savefile,
                                remove=remove, CF=CF)

# All
run_subset('all', remove=False)

# DT cuts
# DT0  0 - 0.5 :: 656783 cutouts
#run_subset('DT0', remove=False)

# DT1  0.5 - 1 :: 3579807 cutouts
#run_subset('DT1', remove=False)

# DT15 1 - 1.5
#run_subset('DT15', remove=False)

# DT2  1.5 - 2.5
#run_subset('DT2', remove=False)

# DT4  2.5 - 4 # > 200000
#run_subset('DT4', remove=False)

# DT5  >5  # 16000 cutouts
#run_subset('DT5', remove=False) 


# CF
# IF WE REDO, SET CF=True