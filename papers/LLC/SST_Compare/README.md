# Overview of SST_Compare folder structure
---

The ulmo/papers/LLC/SST_Compare folder contains the python scripts and Jupyter notebooks to re-produce the analyses undertaken in 
Gallmeier, Prochaska, Cornillon, Menemenlis, and Kelm, 
[submitted](https://gmd.copernicus.org/preprints/gmd-2023-39/).

## Dryad
Contains data of VIIRS SST data and LLC4320 model outputs. Find more information in its own README.md file. 

## Analysis
Contains primarily Jupyter notebooks of several of our exploratory efforts, as well as .npy (and .mat) files to reproduce several results in the notebooks. Find useful helper functions in the 'py' module. 
* load_table() function in py/sst_compare_utils.py

## Figures
Contains two Jupyter notebooks, py module and imgs folder. 
* 'Paper_Figures.ipynb' showcases what figures 'py/figs_llc_viirs.py' can generate. 
* Figures produced by 'py/figs_llc_viirs.py' are stored the imgs folder. These images are also found in our paper. 

