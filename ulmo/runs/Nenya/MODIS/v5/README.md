# Dataset:

MODIS + 96% clear

# SSL model

Same as v4

Copied v4 table to v5

## cp MODIS_SSL_v4.parquet MODIS_Nenya_v5.parquet

# UMAP

Rerun of UMAP (pickle problems!!)

## python nenya_modis_v5.py --func_flag umap --debug --local