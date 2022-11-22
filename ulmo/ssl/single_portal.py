""" Code for running the OS Portal for a single image"""

from ulmo.webpage_dynamic import os_portal


class OSSinglePortal(os_portal.OSPortal):

    def __init__(self, sngl_image, opt):

        # Get UMAP values

        # Launch
        os_portal.OSPortal.__init__(self, data_dict)