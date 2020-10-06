""" Utiltity functions for models"""
import os

from ulmo.ood import ood

model_path = os.path.join(os.getenv("SST_OOD"), 'Models')

def load(mtype):
    # Load up the model
    if mtype == 'standard':
        datadir = os.path.join(model_path, 'R2019_2010_128x128_std')
        filepath = 'PreProc/MODIS_R2019_2010_95clear_128x128_preproc_std.h5'
        pae = ood.ProbabilisticAutoencoder.from_json(datadir + '/model.json',
                                                 datadir=datadir,
                                                 filepath=filepath,
                                                 logdir=datadir)
    else:
        raise IOError("Not ready for mtype={}".format(mtype))
    pae.load_autoencoder()
    pae.load_flow()

    # Return
    return pae
