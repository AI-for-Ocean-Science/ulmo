import os
from importlib_metadata import Deprecated
from pkg_resources import resource_filename
import shutil

from ulmo.models import DCAE, ConditionalFlow
from ulmo.ood import ood
from ulmo.models import io as mod_io

def train():
    raise Deprecated("Moved to ssh_run")

def show_log_probs():
    pae_ssh = mod_io.load_ssh()
    # Plot
    pae_ssh.plot_log_probs(save_figure=True)

# Command line execution
if __name__ == '__main__':
    # Plot log probs
    show_log_probs()