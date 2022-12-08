""" Code for running the OS Portal for a single image"""
import numpy as np
import os

import h5py

from bokeh.transform import transform

from bokeh.layouts import column, gridplot, row
from bokeh.models import ColumnDataSource, Plot
from bokeh.palettes import Viridis256, Plasma256, Inferno256, Magma256, all_palettes
from bokeh.models import LinearColorMapper, ColorBar, ColumnDataSource, Range1d, CustomJS, Div, \
    CDSView, BasicTicker
from bokeh.models.widgets import TextInput, DataTable, TableColumn, NumberFormatter
from bokeh.models.widgets import Select, Button, Dropdown
from bokeh.plotting import figure
from bokeh.models.annotations import Title
from bokeh.colors import RGB
from bokeh.transform import transform
from bokeh.events import DoubleTap, PanEnd, Reset

from bokeh.server.server import Server  # THIS NEEDS TO STAY!
from bokeh.io import curdoc, show

# For the geography figure
#from bokeh.tile_providers import get_provider, Vendors
from bokeh.transform import linear_cmap


from ulmo import io as ulmo_io
from ulmo.utils import catalog 
#from ulmo.webpage_dynamic import os_portal
from ulmo.webpage_dynamic import utils as portal_utils

from IPython import embed

class OSSinglePortal(object):

    def __init__(self, sngl_image, opt, 
                 Nmax=20000, Nclose=1000):

        self.debug=False
        self.imsize = (64, 64) # Need to consider altering this

        # Load UMAP table
        table_file = os.path.join(
            os.getenv('SST_OOD'), 
            'MODIS_L2', 'Tables',
            'MODIS_SSL_96clear_v4_DT15.parquet')
        self.umap_tbl = ulmo_io.load_main_table(
            table_file)
        # Grab h5 pointers
        self.open_files() # Held in self.file_dict

        # Fake the image for now
        self.img_Us = 2.2, 2.5
        self.find_closest(self.img_Us)
        #self.primary_image = sngl_image
        self.primary_image = self.load_images([self.closest])[0]
        
        #embed(header='37 of single_portal.py')

        # Setup focus Us
        #self.reset_focus()

        # Load images
        self.N = len(self.umap_tbl)
        #self.load_images()

        # Init data
        data_dict = {}
        #data_dict['images'] = self.images

        # Metrics
        metrics= [
            ["U0", "US0"], 
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
        #os_portal.OSPortal.__init__(self, data_dict)

        '''
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
        '''
        self.init_bits_and_pieces()

        # Show
        show(column(self.primary_figure))

    def init_bits_and_pieces(self):
        """ Allow for some customization """
        #self.init_title_text_tables()
        #self.generate_buttons()
        self.generate_sources()
        self.generate_figures()
        self.generate_plots()
        #self.register_callbacks()

    def generate_sources(self):

        # Primary image
        xsize, ysize = self.imsize
        im = self.primary_image
        self.primary_source = ColumnDataSource(
            data = {'image':[im], 'x':[0], 'y':[0], 
                    'dw':[xsize], 'dh':[ysize],
                    'min': [np.min(im)], 
                    'max': [np.max(im)]}
        )


    def generate_figures(self):

        self.umap_plot_width = 800
        column_width = 500

        '''
        self.umap_figure = figure(tools='lasso_select,tap,box_zoom,pan,save,reset',
                                  width=self.umap_plot_width,
                                  height=600,
                                  toolbar_location="above", output_backend='webgl', )  # x_range=(-10, 10),
        self.umap_colorbar = ColorBar(color_mapper=self.color_mapper, location=(0, 0), 
                                      major_label_text_font_size='15pt', 
                                      label_standoff=13)
        self.umap_figure.add_layout(self.umap_colorbar, 'right')
        self.umap_figure_axes()
        '''

        # Primary figure
        self.primary_figure = figure(
            tools="box_zoom,save,reset", 
            width=column_width,
            height=column_width,
            toolbar_location="above", 
            output_backend='webgl',
            x_range=(0,self.imsize[0]), 
            y_range=(0,self.imsize[1]))
        portal_utils.remove_ticks_and_labels(self.primary_figure)
                                  #x_range=(0,96), y_range=(0,96))
        '''
        # Gallery
        self.init_gallery_figure()


        # TODO: make this the index
        t = Title()
        t.text = 'TMP'
        self.data_figure.title = t


        for i in range(len(self.gallery_figures)):
            self.remove_ticks_and_labels(self.gallery_figures[i])
            t = Title()
            t.text = ' '
            self.gallery_figures[i].title = t

        # Table
        self.selected_objects_table = DataTable(
            source=self.selected_objects_source,
            columns=self.selected_objects_columns,
            width=column_width,
            height=200,
            scroll_to_selection=False)

        # Geography figure
        tooltips = [("ID", "@obj_ID"), ("Lat","@lat"), ("Lon", "@lon")]
        self.geo_figure = figure(tools='box_zoom,pan,save,reset',
                                  width=self.umap_plot_width,
                                  height=600,
                                  toolbar_location="above", 
                                  x_axis_type="mercator", 
                                  y_axis_type="mercator", 
                                  x_axis_label = 'Longitude', 
                                  y_axis_label = 'Latitude', 
                                  tooltips=tooltips,
                                  output_backend='webgl', )  # x_range=(-10, 10),
        '''

    def generate_plots(self):
        """Generate/init plots
        """

        '''
        # Main scatter plot
        self.umap_scatter = self.umap_figure.scatter(
            'xs', 'ys', source=self.umap_source_view,
            color=transform('color_data', self.color_mapper),
            nonselection_fill_color = self.nonselection_fill_color, 
            nonselection_line_color = 'moccasin',
            nonselection_alpha = 0.2,
            nonselection_line_alpha = 0,
            alpha=0.7,
            line_color=None, #'black',
            size='radius',
            view=self.umap_view)
        '''

        # Snapshot
        self.plot_primary(init=True)

        '''
        # Gallery
        self.plot_gallery()
        #self.gallery_figure = gridplot(self.gallery_figures, 
        #                               ncols=self.ncol)

        # Search circle
        self.umap_search_galaxy = self.umap_figure.circle(
            'xs', 'ys', source=self.search_galaxy_source, alpha=0.5,
            color='tomato', size=self.R_DOT*4, line_color="black", line_width=2)

        LINE_ARGS = dict(color="#3A5785", line_color=None)

        # Geography plot
        # Add map tile
        chosentile = get_provider(Vendors.STAMEN_TONER)
        self.geo_figure.add_tile(chosentile)
        self.geo_color_mapper = linear_cmap(
            field_name = self.dropdown_dict['metric'],
            palette = Plasma256, 
            low = self.color_mapper.low,
            high = self.color_mapper.high)
        self.geo_figure.circle(
            x = 'mercator_x', 
            y = 'mercator_y', 
            color = self.geo_color_mapper, 
            source=self.geo_source, 
            size=5, fill_alpha = 0.7)
        '''

    def plot_primary(self, init=False):
        """ Plot the primary image with color mapper and bar
        """
        # Color map
        self.prim_color_mapper = LinearColorMapper(
            palette="Turbo256", 
            low=self.primary_source.data['min'][0],
            high=self.primary_source.data['max'][0])
        # Data
        self.data_image = self.primary_figure.image(
            'image', 'x', 'y', 'dw', 'dh', 
            source=self.primary_source,
            color_mapper=self.prim_color_mapper)
        # Add color bar
        self.color_bar = ColorBar(
            color_mapper=self.prim_color_mapper, 
            ticker= BasicTicker(), location=(0,0))
        # Finish
        if init:
            self.primary_figure.add_layout(
                self.color_bar, 'right')
        else:
            self.primary_figure.right[0].color_mapper.low = self.prim_color_mapper.low
            self.primary_figure.right[0].color_mapper.high = self.prim_color_mapper.high


    def __call__(self, doc):
        doc.add_root(column(
            row(self.umap_figure, 
                column(self.data_figure,
                       row(self.x_text, self.y_text),
                       self.select_metric,
                       row(self.main_low, self.main_high),
                       self.selected_objects_table,
                       self.print_table,
                )),
            self.gallery_figure,
            row(self.prev_set, self.next_set), 
            self.geo_figure)
                     ) 
        doc.title = 'OS Web Portal'

    def open_files(self):
        # Open files
        self.file_dict = {}
        uni_ppfiles = np.unique(self.umap_tbl.pp_file.values)
        for ppfile in uni_ppfiles:
            base = os.path.basename(ppfile)
            ifile = os.path.join(os.getenv('SST_OOD'), 
                                 'MODIS_L2', 'PreProc', base)
            self.file_dict[base] = h5py.File(ifile, 'r')

    def load_images(self, tbl_idx):
        print("Loading images")
        # Grab em
        images = []
        for kk in tbl_idx: #, row in self.umap_tbl.iterrows():
            row = self.umap_tbl.iloc[kk]
            #
            ppfile = row.pp_file
            pp_idx = row.pp_idx
            # Grab
            base = os.path.basename(ppfile)
            key = 'valid' if row.ulmo_pp_type == 0 else 'train'
            img = self.file_dict[base][key][pp_idx, 0, ...]
            images.append(img)
        # Save
        return np.array(images)

    def reset_focus(self):
        self.focus_Us = (self.img_Us[0], self.img_Us[1])

    def find_closest(self, Us):
        dist = (self.img_Us[0]-self.umap_tbl.US0)**2 + (
            self.img_Us[1]-self.umap_tbl.US1)**2
        self.closest = np.argmin(dist)
        self.umap_closest = self.umap_tbl.iloc[self.closest]

    def set_primary_by_U(self, Us):
        # Find closest
        self.find_closest(Us)
        # Set image
        self.primary_image = self.load_images(
            [self.closest])[0]


        



# TESTING
if __name__ == "__main__":
    #tmp = OSSinglePortal(None, None)

    '''
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
    '''

    obj = OSSinglePortal(None, None)
        