import os

def data_path(filename):
    data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'files')
    return os.path.join(data_dir, filename)
