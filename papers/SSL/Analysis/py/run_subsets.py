""" Run UMAP on subsets of the data (DT) """
import os
from pkg_resources import resource_filename

import ssl_paper_analy

opt_path = os.path.join(resource_filename('ulmo', 'runs'),
                        'SSL', 'MODIS', 'v2', 
                        'experiments', 'modis_model_v2', 'opts_cloud_free.json')

def run_subset(subset, remove=True):                    
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

    outfile = os.path.join(
        os.getenv('SST_OOD'), 
        f'MODIS_L2/Tables/MODIS_SSL_cloud_free_{subset}.parquet')
    umap_savefile = os.path.join(
        os.getenv('SST_OOD'), 
        f'MODIS_L2/UMAP/MODIS_SSL_cloud_free_{subset}_UMAP.pkl')
    ssl_paper_analy.umap_subset(opt_path, outfile, 
                                DT_cut=DT_cut, debug=False,
                                umap_savefile=umap_savefile,
                                remove=remove)

# DT cuts
# DT0  0 - 0.5 :: 430000 cutouts
#run_subset('DT0', remove=False)

# DT1  0.5 - 1 :: 2000000 cutouts
#run_subset('DT1', remove=False)

# DT15 1 - 1.5
#run_subset('DT15', remove=False)

# DT2  1.5 - 2.5
#run_subset('DT2', remove=False)

# DT4  2.5 - 4
#run_subset('DT4', remove=False)

# DT5  >5  # 3000 cutouts
#run_subset('DT5', remove=True) 

# DT1  1 :: 2000000 cutouts
run_subset('DT10', remove=False)