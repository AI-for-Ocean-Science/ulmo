""" Methods to read/write our PAE models """
import os

from ulmo.ood import ood

def load_modis_l2(flavor='std', datadir=None, local=False):
    """Load up the MODIS L2 model (Prochaska, Cornillon & Reiman 2021)

    Args:
        flavor (str, optional): Type of model trained. Defaults to 'std'.
        datadir (str, optional): Path to data for the model. Defaults to None.
        local (bool, optional): If True, use local storage. Defaults to False.

    Returns:
        ulmo.ood.ProbablisticAutoencoder: the PAE
    """

    # TODO -- Allow for s3

    #
    if flavor == 'loggrad':
        if datadir is None:
            datadir = './Models/R2019_2010_128x128_loggrad'
        filepath = 'PreProc/MODIS_R2019_2010_95clear_128x128_preproc_loggrad.h5'
    elif flavor == 'std':
        if local:
            if datadir is None:
                datadir = os.path.join(os.environ('SST_OOD'),
                                       'MODIS_L2', 'Models',
                                       'R2019_2010_128x128_std')
        filepath = 'PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5'
    pae = ood.ProbabilisticAutoencoder.from_json(
        os.path.join(datadir , 'model.json'), 
        datadir=datadir,
        filepath=filepath,
        logdir=datadir)
    # Giddy up                                            
    pae.load_autoencoder()
    pae.load_flow()

    # Return
    return pae