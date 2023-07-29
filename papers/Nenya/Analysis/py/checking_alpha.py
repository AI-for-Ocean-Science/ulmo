""" Bits and pieces for checking alpha """

# imports

import numpy as np

from ulmo import io as ulmo_io
from ulmo.plotting import plotting
from ulmo.utils import image_utils
from ulmo.analysis import fft


def gen_image(U0, U1, outfile):
    # Load
    tbl_file = '/data/Projects/Oceanography/AI/OOD/SST/MODIS_L2/Tables/MODIS_SSL_96clear_v4_DT1.parquet'
    modis_tbl = ulmo_io.load_main_table(tbl_file)

    # Find it
    i_large = np.argmin(np.abs((modis_tbl.US0-U0)**2 + (modis_tbl.US1-U1)**2))
    large_cutout = modis_tbl.iloc[i_large]
    large_img = image_utils.grab_image(large_cutout)

    # Plot it
    plotting.show_image(large_img, show=True)

    # Save it
    np.save(outfile, large_img)
    print(f'Wrote: {outfile}')

    
def gen_images_for_peter():
    # Large
    U0, U1 = 1, -1.8
    outfile = 'large_cutout.npy'
    gen_image(U0, U1, outfile)

    # Small
    U0, U1 = 3, 2.
    outfile = 'small_cutout.npy'
    gen_image(U0, U1, outfile)


if __name__ == '__main__':
    gen_images_for_peter()