""" Run SSL Analysis specific to the paper """
import os
from pkg_resources import resource_filename

import ssl_paper_analy

# DT=2
if False:
    opt_path = os.path.join(resource_filename('ulmo', 'runs'),
                            'SSL', 'MODIS', 'v2', 
                            'experiments', 'modis_model_v2', 'opts_cloud_free.json')
    outfile = os.path.join(os.getenv('SST_OOD'),
                                        'MODIS_L2/Tables/MODIS_SSL_cloud_free_DT2.parquet')
    ssl_paper_analy.umap_subset(opt_path, outfile, DT_cut=(2., 0.2), debug=False) # 180,000 images

# DT=1
if True:
    opt_path = os.path.join(resource_filename('ulmo', 'runs'),
                            'SSL', 'MODIS', 'v2', 
                            'experiments', 'modis_model_v2', 'opts_cloud_free.json')
    outfile = os.path.join(os.getenv('SST_OOD'),
                                        'MODIS_L2/Tables/MODIS_SSL_cloud_free_DT1.parquet')
    umap_savefile = os.path.join(os.getenv('SST_OOD'),
                                        'MODIS_L2/UMAP/MODIS_SSL_cloud_free_DT1_UMAP.pkl')
    ssl_paper_analy.umap_subset(opt_path, outfile, DT_cut=(1., 0.05), debug=False,
                                umap_savefile=umap_savefile) # 262,000 images