import os
from pkg_resources import resource_filename
import shutil

from ulmo.models import DCAE, ConditionalFlow
from ulmo.ood import ood

#dpath = '/home/jovyan/ulmo/ulmo/ssh/'
dpath = '/home/jovyan/Oceanography/SSH/Training/'
datadir= os.path.join(dpath, 'SSH_std')
model_file = os.path.join(resource_filename('ulmo', 'ssh'), 'ssh_pae_model_std.json')
preproc_file = os.path.join(dpath, 'PreProc', 'SSH_100clear_32x32_train.h5')

# Do it
pae = ood.ProbabilisticAutoencoder.from_json(model_file, 
                                             filepath=preproc_file,
                                             datadir=datadir, logdir=datadir)

pae.load_autoencoder()

pae.train_flow(n_epochs=10, batch_size=64, lr=2.5e-4, summary_interval=50, eval_interval=2000)
