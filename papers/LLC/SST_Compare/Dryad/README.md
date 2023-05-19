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
Following is a list of the contents of the hp folder:

1) head_viirs.csv - The VIIRS dataset was divided into two parts, an early period, 2012-2015, referred to as 'head' and a later period, 2018-2020, referred to as 'tail'. The purpose of this was to estimate the uncertainty expected in the median value of LL obtained from each HEALPix cell. This file is, yup, you guessed it, associated with the first period and contains the following fields. 
    1. sigma - the standard deviation of the LL values for all cutouts falling in this HEALPix cell,
    2.  mean - the mean LL of cutouts in the cell,
    3.  median - the mean LL of cutouts in the cell,
    4.  N - the number of cutouts in the cell,
    5.  idx - the HEALPix cell number.

2) tail_viirs.cs - as for head_viirs.cs but for 2018-2020.
3) all viirs.csv - as above but for the entire period, 2012-2020.
4) all llo.csv - table of values for LLC4320 HEALPix cells.
5) 

## Code/Software

All code related to this project may be found on 
[GitHub](https://github.com/AI-for-Ocean-Science/ulmo)
and one may cite [this doi](https://doi.org/10.5281/zenodo.7685510).
