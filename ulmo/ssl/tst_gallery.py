""" Code for running the OS Portal for a single image"""
import numpy as np
import os

import h5py

from bokeh.transform import transform
from bokeh.layouts import column, gridplot, row
from bokeh.models import LinearColorMapper, ColorBar, ColumnDataSource, Range1d, CustomJS, Div, \
    CDSView, BasicTicker

from ulmo import io as ulmo_io
from ulmo.utils import catalog 
from ulmo.webpage_dynamic import os_portal

from IPython import embed

class OSSinglePortal(os_portal.OSPortal):

    def __init__(self, sngl_image, opt, Nmax=20000, Nclose=1000):

        # Get UMAP values
        table_file = os.path.join(
            os.getenv('SST_OOD'), 
            'MODIS_L2', 'Tables',
            'MODIS_SSL_96clear_v4_DT15.parquet')

        umap_tbl = ulmo_io.load_main_table(
            table_file)
        # Cut down
        keep = np.array([False]*len(umap_tbl))
        keep[np.arange(min(Nmax,
                       len(umap_tbl)))] = True 
        self.umap_tbl = umap_tbl[keep].copy()

        #self.img_U0 = embedding[0,0]
        #self.img_U1 = embedding[0,1]
        self.img_U0 = 2.2
        self.img_U1 = 2.5

        # Load images
        self.N = len(self.umap_tbl)
        self.load_images()

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
        metric_dict = dict(obj_ID=np.arange(len(self.umap_tbl)))
        for metric in metrics:
            if metric[0] == 'DT':
                metric_dict[metric[0]] = (self.umap_tbl.T90-self.umap_tbl.T10).values
            elif metric[1] == 'min_slope':
                metric_dict[metric[0]] = np.minimum(self.umap_tbl.merid_slope.values,
                                                    self.umap_tbl.zonal_slope.values)
            else:
                metric_dict[metric[0]] = self.umap_tbl[metric[1]].values
        data_dict['metrics'] = metric_dict

        # Launch
        os_portal.OSPortal.__init__(self, data_dict)

        # Specific to single portal
        self.select_on_source = True

        # Select objects
        dist = (self.img_U0-self.umap_source.data['xs'])**2 + (
            self.img_U1-self.umap_source.data['ys'])**2
        srt_dist = np.argsort(dist)

        self.Nclose = min(Nclose, len(srt_dist))
        all_items = np.array(self.umap_source.data['names'])[srt_dist[:self.Nclose]]
        

        # In view
        rows = catalog.match_ids(all_items, 
                                 np.array(self.umap_source_view.data['names']),
                                 require_in_match=False)
        in_view = rows > 0                                

        self.umap_source_view.selected.indices = rows[in_view].tolist()
        #self.umap_source_callback(None, None, rows[in_view].tolist())
        self.search_galaxy_source.data = dict(
            xs=[self.img_U0], ys=[self.img_U1])
        self.update_color(None)

        # Gallery
        self.reset_gallery_index()
        self.stacks_callback()
        self.plot_gallery()  # Resets color map

    def __call__(self, doc):
        doc.add_root(column(row(self.main_title_div ), 
                            row(column(self.info_div), 
                                column(self.gallery_figure,
                                    row(self.prev_set, self.next_set), 
                                    ), 
                            )
        ))
        doc.title = 'OS Gallery'

    def plot_gallery(self):
        """ Plot the gallery of images
        """
        self.spectrum_stacks = []
        for i in range(self.nrow*self.ncol):
            spec_stack = self.gallery_figures[i].image(
                'image', 'x', 'y', 'dw', 'dh', 
                source=self.stacks_sources[i],
                color_mapper=self.snap_color_mapper)
            self.spectrum_stacks.append(spec_stack)
            if self.verbose:
                print(f"Updated gallery {i}")

    def plot_snapshot(self, init=False):
        """ Plot the current image with color mapper and bar
        """
        self.snap_color_mapper = LinearColorMapper(palette="Turbo256", 
                                         low=-1., #self.data_source.data['min'][0],
                                         high=1.) #self.data_source.data['max'][0])
        self.data_image = self.data_figure.image(
            'image', 'x', 'y', 'dw', 'dh', source=self.data_source,
            color_mapper=self.snap_color_mapper)
        # Add color bar
        self.color_bar = ColorBar(color_mapper=self.snap_color_mapper, 
                             ticker= BasicTicker(), location=(0,0))
        if init:
            self.data_figure.add_layout(self.color_bar, 'right')
        else:
            self.data_figure.right[0].color_mapper.low = self.snap_color_mapper.low
            self.data_figure.right[0].color_mapper.high = self.snap_color_mapper.high

    def load_images(self):
        print("Loading images")

        # Open files
        file_dict = {}
        uni_ppfiles = np.unique(self.umap_tbl.pp_file.values)
        for ppfile in uni_ppfiles:
            base = os.path.basename(ppfile)
            ifile = os.path.join(os.getenv('SST_OOD'), 
                                 'MODIS_L2', 'PreProc', base)
            file_dict[base] = h5py.File(ifile, 'r')
            
        # Grab em
        images = []
        for kk, row in self.umap_tbl.iterrows():
            ppfile = row.pp_file
            pp_idx = row.pp_idx
            # Grab
            base = os.path.basename(ppfile)
            key = 'valid' if row.ulmo_pp_type == 0 else 'train'
            img = file_dict[base][key][pp_idx, 0, ...]
            images.append(img)
        # Save
        self.images = np.array(images)
        print("Done")

# TESTING
if __name__ == "__main__":
    #tmp = OSSinglePortal(None, None)

    # Odd work-around
    def get_session(doc):
        sess = OSSinglePortal(None, None, Nmax=2000, Nclose=50)
        #sess = OSSinglePortal(None, None)
        return sess(doc)

    # Do me!
    server = os_portal.Server({'/': get_session}, num_procs=1)
    server.start()
    print('Opening Bokeh application for OS data on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
        