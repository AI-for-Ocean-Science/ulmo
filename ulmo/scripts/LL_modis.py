""" Script to calculate LL for a field in a MODIS image"""

from IPython import embed

def parser(options=None):
    import argparse
    # Parse
    parser = argparse.ArgumentParser(description='LL for a MODIS image')
    parser.add_argument("file", type=str, help="MODIS filename")
    parser.add_argument("row", type=int, help="Row for the field")
    parser.add_argument("col", type=int, help="Column for the field")
    parser.add_argument("-s","--show", default=False, action="store_true", help="Show pre-processed image?")
    #parser.add_argument("-g", "--galaxy_options", type=str, help="Options for fg/host building (photom,cigale)")
    parser.add_argument("--model_env", type=str, default='SST_OOD_MODELDIR',
                        help="Environmental variable pointing to model directory")

    if options is None:
        pargs = parser.parse_args()
    else:
        pargs = parser.parse_args(options)
    return pargs


def main(pargs):
    """ Run
    """
    import numpy as np
    import os
    import warnings
    from matplotlib import pyplot as plt
    import seaborn as sns

    import glob
    import pickle
    import torch
    from tqdm.auto import tqdm

    from ulmo import io as ulmo_io
    from ulmo.preproc import utils as pp_utils
    from ulmo.plotting import plotting
    from ulmo.models import autoencoders, ConditionalFlow
    from ulmo import ood


    # Load the image
    sst, qual, latitude, longitude = ulmo_io.load_nc(pargs.file, verbose=False)

    # Generate the masks
    masks = pp_utils.build_mask(sst, qual)

    # Grab the field and mask
    field_size = (128, 128)
    row, col = pargs.row, pargs.col

    field = sst[row:row + field_size[0], col:col + field_size[1]]
    mask = masks[row:row + field_size[0], col:col + field_size[1]]

    print("This {} field has {:0.1f}% cloud coverage".format(field_size, 100*mask.sum()/field.size))

    # Pre-process
    pp_field, mu = pp_utils.preproc_field(field, mask)
    print("Preprocessing done!")

    # Show?
    if pargs.show:
        pal, cm = plotting.load_palette()
        plt.clf()
        ax = plt.gca()
        sns.heatmap(pp_field, ax=ax, xticklabels=[], yticklabels=[], cmap=cm, vmin=-5, vmax=5)
        plt.show()

    # Load model
    model_path = os.getenv(pargs.model_env)
    print("Loading model in {}".format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dcae = autoencoders.DCAE.from_file(os.path.join(model_path, 'autoencoder.pt'),
                                       image_shape=(1, 64, 64),
                                       latent_dim=512)
    flow = ConditionalFlow(
        dim=512,
        context_dim=None,
        transform_type='autoregressive',
        n_layers=10,
        hidden_units=256,
        n_blocks=2,
        dropout=0.2,
        use_batch_norm=False,
        tails='linear',
        tail_bound=10,
        n_bins=5,
        min_bin_height=1e-3,
        min_bin_width=1e-3,
        min_derivative=1e-3,
        unconditional_transform=False,
        encoder=None)
    flow.load_state_dict(torch.load(os.path.join(model_path, 'flow.pt'), map_location=device))
    pae = ood.ProbabilisticAutoencoder(dcae, flow, 'tmp/', device=device, skip_mkdir=True)
    print("Model loaded!")

    pae.autoencoder.eval()
    pae.flow.eval()

    # Latent
    pp_field.resize(1, 1, 64, 64)
    dset = torch.utils.data.TensorDataset(torch.from_numpy(pp_field).float())
    loader = torch.utils.data.DataLoader(
        dset, batch_size=1, shuffle=False,
        drop_last=False, num_workers=16)
    with torch.no_grad():
        latents = [pae.autoencoder.encode(data[0].to(device)).detach().cpu().numpy()
                   for data in tqdm(loader, total=len(loader), unit='batch', desc='Computing latents')]
    print("Latents generated!")

    # Scaler
    scaler_path = glob.glob(os.path.join(model_path, '*scaler.pkl'))[0]
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    latents = scaler.transform(np.concatenate(latents))
    #latents = np.concatenate(latents)

    # Debug
    #print("Debug: {}".format(np.sum(latents)))

    # LL
    dset = torch.utils.data.TensorDataset(torch.from_numpy(latents).float())
    loader = torch.utils.data.DataLoader(
        dset, batch_size=1024, shuffle=False,
        drop_last=False, num_workers=16)

    with torch.no_grad():
        log_prob = [pae.flow.log_prob(data[0].to(pae.device)).detach().cpu().numpy()
                        for data in tqdm(loader, total=len(loader), unit='batch', desc='Computing log probs')]
    print("Log probabilities generated!")

    print("The LL for the field is: {}".format(float(log_prob[0])))
