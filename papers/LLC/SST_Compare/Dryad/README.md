# VIIRS data and Ulmo model for comparison to the LLC4320 ECCO ocean general circulation model
---

Give a brief summary of dataset contents, contextualized in experimental procedures and results.

## Abstract

This archive contains images and tables for the analyses undertaken for the 
purpose of assesing the LLC 4320 ocean model output of sea surface temperature (SST). Results of our investigations are reported in full in 
Gallmeier, Prochaska, Cornillon, Menemenlis, and Kelm, 
[submitted](https://gmd.copernicus.org/preprints/gmd-2023-39/).

In brief, we compared VIIRS remote sensing data for SST on scales of ~100 km x 100 km
against model outputs from the state-of-the-art ECCO LLC4320 ocean general circulation
model.  Using a machine learning metric named [Ulmo](https://ui.adsabs.harvard.edu/abs/2021RemS...13..744P/abstract), we demonstrate the LLC4320
model performs well across most of the global ocean.  We highlight notable departures
in the gulf stream, on the Equatorial Pacific, and in the Antarctic Cicumpolar Current.

All of the data provided here were sourced from public archives:
  1) the JPL Physical Oceanography Distributed Active Archive Center (PO.DAAC, https://podaac.jpl.nasa.gov)
  2) MITgcm.org (using xmitgcm)
or are products of our own analysis.


## Description of the data and file structure

There are two main datasets: (1) cutouts and tables from the VIIRS analysis
and (2) cutouts and tables for the LLC analysis (matched to the VIIRS data).

The cutout images are provided in hdf5 files and there are up to four dataset
groups (the first 2 are always present): 
 (i) **valid** -- the images with shape (n_images, 1, 64, 64)
(ii) **valid_metadata** -- a string representation of primary metadata
(iii) **train** -- the training images for Ulmo with shape (n_images, 1, 64, 64)
(iv) **train_metadata** -- a string representation of primary metadata for the training images
describing each cutout (n_images, 18 columns).  
The column names are held in .attrs['columns']: 

  * filename 
  * row 
  * col 
  * lat 
  * lon 
  * clear_fraction
  * field_size 
  * datetime 
  * ex_filename 
  * pp_file 
  * pp_root
  * pp_idx 
  * pp_type 
  * mean_temperature 
  * Tmin 
  * Tmax
  * T90 
  * T10

[Here is the datamodel](https://github.com/AI-for-Ocean-Science/ulmo/blob/main/ulmo/defs.py) for these variables. 

In addition, we provide files of healpix evaluations of various
quantities presented in the paper figures.  These are stored as
numpy files (without a .np extension!) and ASCII .csv tables.
Following is a list of the contents of the dataset:

1) head_viirs.csv - The VIIRS dataset was divided into two parts, an early period, 2012-2015, referred to as 'head' and a later period, 2018-2020, referred to as 'tail'. The purpose of this was to estimate the uncertainty expected in the median value of LL obtained from each HEALPix cell. This file is, yup, you guessed it, associated with the first period and contains the following fields. 
    1. sigma - the standard deviation of the LL values for all cutouts falling in this HEALPix cell,
    2.  mean - the mean LL of cutouts in the cell,
    3.  median - the mean LL of cutouts in the cell,
    4.  N - the number of cutouts in the cell,
    5.  idx - the HEALPix cell number.

2) tail_viirs.csv - as for head_viirs.cs but for 2018-2020.
3) all_viirs.csv - as above but for the entire period, 2012-2020.
4) all_llc.csv - table of values for LLC4320 HEALPix cells.
5) hp_lons_V98 - the longitude values for each HEALPix cell in the same order as in the .csv files.
6) hp_lats_V98 - the latitude values for each HEALPix cell.
7) flow.pt - This is a pytorch file holding the normalizing flow model of Ulmo used for the project.
8) autoencoder.pt - This is a pytorch file holding the autoencoder model of Ulmo used for the project.
9) VIIRS_2013_98clear_192x192_preproc_viirs_std_train_scaler.pkl -- This is a Python pickled file of the scalar of Ulmo used for the project.
10) model.json -- JSON file describing the Ulmo model used.

## Code/Software

All code related to this project may be found on 
[GitHub](https://github.com/AI-for-Ocean-Science/ulmo)
and one may cite [this doi](https://doi.org/10.5281/zenodo.7685510).
