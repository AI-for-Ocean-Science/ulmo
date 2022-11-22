""" Code for running the OS Portal for a single image"""
import numpy as np
import os

import h5py

from ulmo import io as ulmo_io
from ulmo.webpage_dynamic import os_portal

from IPython import embed

class OSSinglePortal(os_portal.OSPortal):

    def __init__(self, sngl_image, opt, Nimages=1000):

        # Get UMAP values
        table_file = os.path.join(
            os.getenv('SST_OOD'), 
            'MODIS_L2', 'Tables',
            'MODIS_SSL_96clear_v4_DT15.parquet')

        self.umap_tbl = ulmo_io.load_main_table(
            table_file)

        #self.img_U0 = embedding[0,0]
        #self.img_U1 = embedding[0,1]
        self.img_U0 = 2.2
        self.img_U1 = 2.5

        # Load images
        self.N = Nimages
        self.load_data()

        # Init
        data_dict = {}
        data_dict['images'] = self.images

        # Metrics
        metrics= [["U0", "US0"],
        ["U1", "US1"],
        ["LL", "LL"],
        ["DT", "DT"],
        ["lon", "lon"],
        ["lat", "lat"],
        ["avgT", "mean_temperature"]]
        metric_dict = dict(obj_ID=np.arange(len(self.cut_tbl)))
        for metric in metrics:
            if metric[0] == 'DT':
                metric_dict[metric[0]] = (self.cut_tbl.T90-self.cut_tbl.T10).values
            elif metric[1] == 'min_slope':
                metric_dict[metric[0]] = np.minimum(self.cut_tbl.merid_slope.values,
                                                    self.cut_tbl.zonal_slope.values)
            else:
                metric_dict[metric[0]] = self.cut_tbl[metric[1]].values
        data_dict['metrics'] = metric_dict

        # Launch
        os_portal.OSPortal.__init__(self, data_dict)

    def load_data(self):
        # Find the closest
        dist = (self.img_U0-self.umap_tbl.US0.values)**2 + (
            self.img_U1-self.umap_tbl.US1.values)**2
        srt_dist = np.argsort(dist)

        items = srt_dist[:self.N]
        self.cut_tbl = self.umap_tbl.iloc[items]

        # Open files
        file_dict = {}
        uni_ppfiles = np.unique(self.cut_tbl.pp_file.values)
        for ppfile in uni_ppfiles:
            base = os.path.basename(ppfile)
            ifile = os.path.join(os.getenv('SST_OOD'), 
                                 'MODIS_L2', 'PreProc', base)
            file_dict[base] = h5py.File(ifile, 'r')
            
        # Grab em
        images = []
        for kk, row in self.cut_tbl.iterrows():
            ppfile = row.pp_file
            pp_idx = row.pp_idx
            # Grab
            base = os.path.basename(ppfile)
            key = 'valid' if row.ulmo_pp_type == 0 else 'train'
            img = file_dict[base][key][pp_idx, 0, ...]
            images.append(img)
        # Save
        self.images = np.array(images)

# TESTING
if __name__ == "__main__":
    #tmp = OSSinglePortal(None, None)

    # Odd work-around
    def get_session(doc):
        sess = OSSinglePortal(None, None)
        return sess(doc)

    # Do me!
    server = os_portal.Server({'/': get_session}, num_procs=1)
    server.start()
    print('Opening Bokeh application for OS data on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
        