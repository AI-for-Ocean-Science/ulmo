# Assessing the LLC with Ulmo
---

Give a brief summary of dataset contents, contextualized in experimental procedures and results.

## Abstract

This archive contains images and tables for the analyses undertaken for the 
purpose of assesing the LLC 4320 ocean model output of sea surface temperature (SST). Results of our investigations are reported in full in 
Gallmeier, Prochaska, Cornillon, Menemenlis, and Kelm, 
[submitted](https://gmd.copernicus.org/preprints/gmd-2023-39/).

## Description of the data and file structure

There are two main datasets: (1) cutouts and tables from the VIIRS analysis
and (2) cutouts and tables for the LLC analysis (matched to the VIIRS data).

The cutout images are provided in hdf5 files and there are two dataset
groups: (i) **valid** -- the images with shape (n_images, 1, 64, 64)
and (ii) **valid_metadata** -- a string representation of primary metadata
describing each cutout (n_images, 18 columns).  The column names
are held in .attrs['columns'].

In addition, we provide files of healpix evaluations of various
quantities presented in the paper figures.  These are stored as
numpy files (without a .np extension!) and ASCII .csv tables.


## Code/Software

All code related to this project may be found on 
[GitHub](https://github.com/AI-for-Ocean-Science/ulmo)
and one may cite [this doi](https://doi.org/10.5281/zenodo.7685510).
