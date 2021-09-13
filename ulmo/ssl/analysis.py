""" Analysis methods for self-supervised learning 
"""
import numpy as np
import pickle

import pandas
from matplotlib import pyplot as plt

import umap

from ulmo import io as ulmo_io
from ulmo.plotting import plotting

from IPython import embed

def latents_umap(latents:np.ndarray, train:np.ndarray, 
         valid:np.ndarray, valid_tbl:pandas.DataFrame,
         fig_root='', transformer_file=None):
    """ Run a UMAP on input latent vectors.
    A subset are used to train the UMAP and then
    one applies it to the valid set.

    The UMAP U0, U1 coefficients are written to an input table.

    Args:
        latents (np.ndarray): Total set of latent vectors (training+valid)
            Shape should be (nvectors, size of latent space)
        train (np.ndarray): indices for training
        valid (np.ndarray): indices for applying the UMAP
        valid_tbl (pandas.DataFrame): [description]
        fig_root (str, optional): [description]. Defaults to ''.
        transformer_file (str, optional): Write the UMAP fit to this file
    """

    # UMAP me
    print("Running UMAP..")
    reducer_umap = umap.UMAP()
    latents_mapping = reducer_umap.fit(latents[train])
    if transformer_file is not None:
        pickle.dump(latents_mapping, open(transformer_file, "wb" ) )
        tmp = pickle.load(ulmo_io.open(transformer_file, "rb" ) )
    print("Done")

    # Apply to embedding
    print("Applying to the valid images")
    train_embedding = latents_mapping.transform(latents[train])
    valid_embedding = latents_mapping.transform(latents[valid])
    print("Done")

    # Quick figures
    if len(fig_root) > 0:
        print("Generating plots")
        num_samples = train.size
        point_size = 20.0 / np.sqrt(num_samples)
        dpi = 100
        width, height = 800, 800

        plt.figure(figsize=(width//dpi, height//dpi))
        plt.scatter(latents_mapping.embedding_[:, 0], 
            latents_mapping.embedding_[:, 1], s=point_size)
        plt.savefig(fig_root+'_train_UMAP.png')

        # New plot
        num_samplesv = latents[valid].shape[0]
        point_sizev = 1.0 / np.sqrt(num_samplesv)
        plt.figure(figsize=(width//dpi, height//dpi))
        ax = plt.gca()
        img = ax.scatter(valid_embedding[:, 0], 
                valid_embedding[:, 1], s=point_sizev,
            c=valid_tbl.LL, cmap='jet', vmin=-1000)
        cb = plt.colorbar(img, pad=0.)
        cb.set_label('LL', fontsize=20.)
        #
        ax.set_xlabel(r'$U_0$')
        ax.set_ylabel(r'$U_1$')
        plotting.set_fontsize(ax, 15.)
        #
        plt.savefig(fig_root+'_valid_UMAP.png', dpi=300)

    # Return
    return train_embedding, valid_embedding, latents_mapping