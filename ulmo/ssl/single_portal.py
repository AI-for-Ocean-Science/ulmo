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
        self.verbose = False

        # Load UMAP table
        table_file = os.path.join(
            os.getenv('SST_OOD'), 
            'MODIS_L2', 'Tables',
            'MODIS_SSL_96clear_v4_DT15.parquet')
        self.umap_tbl = ulmo_io.load_main_table(
            table_file)
        self.umap_tbl['DT'] = self.umap_tbl.T90 - self.umap_tbl.T10
        self.umap_data = np.array([self.umap_tbl.US0.values, 
                                   self.umap_tbl.US1.values,]).T

        # Grab h5 pointers
        self.open_files() # Held in self.file_dict

        #embed(header='37 of single_portal.py')

        # Setup focus Us
        #self.reset_focus()

        # Primary Figure
        self.primary_column_width = 500

        # UMAP FIGURE
        rev_Plasma256 = Plasma256[::-1]
        self.DECIMATE_NUMBER = 5000
        self.UMAP_XYLIM_DELTA = 0.5
        self.R_DOT = 6#10
        self.high_colormap_factor = 0.1
        self.umap_color_mapper = LinearColorMapper(
            palette=rev_Plasma256, low=0, high=1, 
            nan_color=RGB(220, 220, 220, a = 0.1))
        self.xkey, self.ykey = 'U0', 'U1'
        self.nonselection_fill_color = transform(
            'color_data', self.umap_color_mapper)
        self.umap_plot_width = 800

        # Gallery Figure
        self.nrow, self.ncol = 2, 5
        self.gallery_index = 0

        # ########################################
        # Fake the image for now
        self.img_Us = 2.2, 2.5
        self.set_primary_by_U(self.img_Us)
        # Match
        self.match_radius = 0.5
        self.set_matched(self.prim_Us, 
                         self.match_radius)
        # Load images
        self.set_gallery_images()

        # ########################################
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

        self.metric_dict = dict(obj_ID=np.arange(len(self.umap_tbl)))
        for metric in metrics:
            if metric[0] == 'DT':
                self.metric_dict[metric[0]] = (self.umap_tbl.T90-self.umap_tbl.T10).values
            elif metric[1] == 'min_slope':
                self.metric_dict[metric[0]] = np.minimum(self.umap_tbl.merid_slope.values,
                                                    self.umap_tbl.zonal_slope.values)
            else:
                self.metric_dict[metric[0]] = self.umap_tbl[metric[1]].values
        self.obj_ids = self.metric_dict['obj_ID']
        #data_dict['metrics'] = metric_dict

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

        '''
        # Show
        show(column(row(
            column(self.primary_figure,
                row(self.DT_text, self.PCB_low, self.PCB_high),
                   )))
        )
        '''

    def __call__(self, doc):
        doc.add_root(column(
            row(column(
                self.primary_figure,
                row(self.DT_text, self.PCB_low, self.PCB_high),
                   ),
                self.umap_figure,
                ),
            self.gallery_figure,
            ))
        doc.title = 'SSL Portal'

    def init_bits_and_pieces(self):
        """ Allow for some customization """
        self.init_title_text_tables()
        #self.generate_buttons()
        self.generate_sources()
        self.generate_figures()
        self.generate_plots()
        self.register_callbacks()

    def init_title_text_tables(self):
        # Label primary figure
        self.DT_text = Div(text=f'DT: {self.prim_DT:.2f}K',
                           styles={'font-size': '199%', 
                                   'color': 'black'}, 
                           width=100)

        # Color bar for primary figure
        self.PCB_low = TextInput(title='PCB Low:', max_width=100)
        self.PCB_high = TextInput(title='PCB High:', max_width=100)

        # Color bar for UMAP figure
        self.UCB_low = TextInput(title='PCB Low:', max_width=100)
        self.UCB_high = TextInput(title='UCB High:', max_width=100)

    def generate_sources(self):
        # Primry figure
        self.set_primary_source()

        # UMAP scatter
        self.update_umap_color(None)
        # Unpack for convenience
        #metric = self.metric_dict[self.dropdown_dict['metric']]
        metric = self.metric_dict['LL']
        self.U_xlim = (np.min(self.umap_source.data['xs']) 
                       - self.UMAP_XYLIM_DELTA, 
                       np.max(self.umap_source.data['xs']) 
                       + self.UMAP_XYLIM_DELTA)
        self.U_ylim = (np.min(self.umap_source.data['ys']) 
                       - self.UMAP_XYLIM_DELTA, 
                       np.max(self.umap_source.data['ys']) 
                       + self.UMAP_XYLIM_DELTA)

        points = portal_utils.get_decimated_region_points(
            self.U_xlim[0], self.U_xlim[1], 
            self.U_ylim[0], self.U_ylim[1],
            self.umap_source.data, self.DECIMATE_NUMBER)

        self.umap_source_view = ColumnDataSource(
            #data=dict(xs=self.umap_data[embedding][points, 0],
            #          ys=self.umap_data[embedding][points, 1],
            data=dict(xs=self.umap_data[points, 0],
                      ys=self.umap_data[points, 1],
                      color_data=metric[points],
                      names=list(points),
                      radius=[self.R_DOT] * len(points)),
                    )
        self.points = np.array(points)
        # Selected
        self.umap_view = CDSView()#source=self.umap_source_view)


    def set_primary_source(self):
        # Primary image
        xsize, ysize = self.imsize
        im = self.primary_image
        self.primary_source = ColumnDataSource(
            data = {'image':[im], 'x':[0], 'y':[0], 
                    'dw':[xsize], 'dh':[ysize],
                    'min': [np.min(im)], 
                    'max': [np.max(im)]}
        )

    def update_umap_color(self, event):
        """ Update the color bar and set source for UMAP

        Args:
            event (bokey event): 
        """
        # Update?
        if event is not None:
            self.dropdown_dict['metric'] = event.item
            self.umap_figure_axes()

        # UPDATE LATER
        #metric_key = self.dropdown_dict['metric']
        metric_key = 'LL'
        metric = self.metric_dict[metric_key]

        self.set_colormap(metric, metric_key)
        # Set limits
        self.UCB_low.value = str(self.umap_color_mapper.low)
        self.UCB_high.value = str(self.umap_color_mapper.high)

        self.umap_source = ColumnDataSource(
            #data=dict(xs=self.umap_data[embedding][:, 0],
            #            ys=self.umap_data[embedding][:, 1],
            data=dict(xs=self.umap_data[:, 0],
                        ys=self.umap_data[:, 1],
                        color_data=metric,
                        radius=[self.R_DOT] * len(metric),
                        names=list(np.arange(len(metric))),
                    ))
        # Init?
        if event is None:
            return

        background_objects = self.umap_source_view.data['names']
        self.selected_objects, indices = self.get_selected_from_match(background_objects)

        self.get_new_view_keep_selected(background_objects, 
                                        indices)
        #self.select_score_table.value = self.select_score.value

    def set_colormap(self, metric:np.ndarray, 
                     metric_key:str):
        """Set the color map for the given metric

        Args:
            metric (np.ndarray): Metric values of interest
            metric_key (str): Metric of interest, e.g. LL
        """
        mx = np.nanmax(metric)
        mn = np.nanmin(metric)
        if mn == mx:
            high = mx + 1
            low = mn - 1
        else:
            high = mx + (mx - mn)*self.high_colormap_factor
            low = mn
            # set max of colormap to Nth largets val, to deal with outliers
            nth = 100
            if len(metric)>nth:
                nmx = np.sort(metric)[-nth]
                if nmx*1.2 < mx:
                    high = nmx
        
        # THIS ISN'T THE BEST IDEA
        if metric_key == 'LL':
            low = max(low, -1000.)

        self.umap_color_mapper.high = high
        self.umap_color_mapper.low = low

        return

    def get_new_view_keep_selected(self, background_objects, 
                                   selected_objects_idx, 
                                   custom_sd = None):

        if custom_sd is None:
            # UPDATE THIS
            #metric = self.metric_dict[self.dropdown_dict['metric']]
            metric = self.metric_dict['LL'] #self.dropdown_dict['metric']]
        else:
            metric = custom_sd

        new_objects = np.array(background_objects)

        #self.umap_source_view = ColumnDataSource(
        #         data=dict(xs=self.umap_data[new_objects, 0],
        #                   ys=self.umap_data[new_objects, 1],
        #                   color_data=metric[new_objects],
        #                   names=list(new_objects),
        #                   radius=[self.R_DOT] * len(new_objects),
        #                 ))
        tmp_view = ColumnDataSource(
                 data=dict(xs=self.umap_data[new_objects, 0],
                           ys=self.umap_data[new_objects, 1],
                           color_data=metric[new_objects],
                           names=list(new_objects),
                          radius=[self.R_DOT] * len(new_objects),
                         ))
        self.points = np.array(new_objects)
        #self.umap_scatter.data_source.data = dict(tmp_view.data)
        #self.umap_scatter.data_source = self.umap_source_view
        self.umap_source_view.data = dict(tmp_view.data)
        #self.umap_source_view.selected = tmp_view.selected
        if len(selected_objects_idx) > 0:
            self.umap_source_view.selected.indices = selected_objects_idx
        else:
            self.umap_source_view.selected.indices = [-1]

        # Set indices etc.
        '''
        # Bring back for table
        new_dict = dict(index=list(selected_objects))
        for key in self.metric_dict.keys():
            new_dict[key] = [self.metric_dict[key][s] for s in selected_objects]
        self.selected_objects.data = new_dict
        self.update_table.value = str(np.random.rand())
        '''
        #print(self.umap_data[new_objects[selected_objects_idx]])

        #elif len(selected_objects_) > 0:
        #    self.selected_objects = ColumnDataSource(
        #        data=dict(index=[], score=[], order=[], 
        #                  info_id=[], object_id=[]))
        #    self.update_table.value = str(np.random.rand())
        #    self.internal_reset.value = str(np.random.rand())
        #else:
        #    self.update_table.value = str(np.random.rand())

        # Update indices?

        '''
        # Update circle
        index = self.select_object.value

        if (index in set(background_objects)) :
            pass
        else:
            if len(selected_objects) > 0:
                if (index in set(selected_objects)):
                    pass
                else:
                    index = str(selected_objects[0])
                    self.select_object.value = index
            else:
                index = str(background_objects[0])
                self.select_object.value = index

        self.update_search_circle(index)
        '''                                
        return


    def generate_figures(self):


        # Primary figure
        self.primary_figure = figure(
            tools="box_zoom,save,reset", 
            width=self.primary_column_width,
            height=self.primary_column_width,
            toolbar_location="above", 
            output_backend='webgl',
            x_range=(0,self.imsize[0]), 
            y_range=(0,self.imsize[1]))
        portal_utils.remove_ticks_and_labels(self.primary_figure)

        # UMAP figure
        self.umap_figure = figure(tools='box_zoom,pan,save,reset',
                                  width=self.umap_plot_width,
                                  height=600,
                                  toolbar_location="above", 
                                  output_backend='webgl', )  # x_range=(-10, 10),
        self.umap_colorbar = ColorBar(
            color_mapper=self.umap_color_mapper, 
            location=(0, 0), 
            major_label_text_font_size='15pt', 
            label_standoff=13)
        self.umap_figure.add_layout(
            self.umap_colorbar, 'right')
        self.umap_figure_axes()
                                  #x_range=(0,96), y_range=(0,96))
        # Gallery
        self.init_gallery_figure()


        '''
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

    def umap_figure_axes(self):
        """ Set the x-y axes
        """

        #embedding_name = self.dropdown_dict['embedding']
        embedding_name = f"{self.xkey}, {self.ykey}"
        # DROPDOWN
        #metric_name = self.dropdown_dict['metric']
        metric_name = 'LL'
        if metric_name == 'No color':
            self.umap_figure.title.text  = '{}'.format(embedding_name)
        else:
            self.umap_figure.title.text  = '{} - Colored by {}'.format(embedding_name , metric_name)
        self.umap_figure.title.text_font_size = '17pt'

        # Labels
        self.umap_figure.xaxis.axis_label = self.xkey
        self.umap_figure.yaxis.axis_label = self.ykey

        self.umap_figure.xaxis.major_tick_line_color = None  # turn off x-axis major ticks

        self.umap_figure.xaxis.major_tick_line_color = 'black'  # turn off x-axis major ticks
        self.umap_figure.xaxis.minor_tick_line_color = 'black'  # turn off x-axis minor ticks

        self.umap_figure.yaxis.minor_tick_line_color = 'black'  # turn off y-axis major ticks
        self.umap_figure.yaxis.major_tick_line_color = 'black'  # turn off y-axis minor ticks

        self.umap_figure.xaxis.major_label_text_font_size = "15pt"
        self.umap_figure.yaxis.major_label_text_font_size = "15pt"
        self.umap_figure.xaxis.axis_label_text_font_size = "15pt"
        self.umap_figure.yaxis.axis_label_text_font_size = "15pt"

    def init_gallery_figure(self):
        self.gallery_width = self.primary_column_width + self.umap_plot_width
        # Gallery figure 
        title_height = 20
        buffer = 10*self.ncol
        self.collage_im_width = int((self.gallery_width-buffer)/self.ncol)
        self.gallery_figures = []
        for kk in range(self.nrow*self.ncol):
            sfig = figure(tools="box_zoom,save,reset", 
                          width=self.collage_im_width, 
                          height=self.collage_im_width+title_height, 
                          toolbar_location="above", #output_backend='webgl', 
                          x_range=(0,self.imsize[0]), y_range=(0,self.imsize[1]))
            self.gallery_figures.append(sfig)

        self.gallery_figure = gridplot(self.gallery_figures, 
                                       ncols=self.ncol)

        for i in range(len(self.gallery_figures)):
            portal_utils.remove_ticks_and_labels(
                self.gallery_figures[i])
            t = Title()
            t.text = ' '
            self.gallery_figures[i].title = t

    def generate_plots(self):
        """Generate/init plots
        """

        # Primary
        self.plot_primary(init=True)

        # Main scatter plot
        self.umap_scatter = self.umap_figure.scatter(
            'xs', 'ys', source=self.umap_source_view,
            color=transform('color_data', 
                            self.umap_color_mapper),
            nonselection_fill_color = self.nonselection_fill_color, 
            nonselection_line_color = 'moccasin',
            nonselection_alpha = 0.2,
            nonselection_line_alpha = 0,
            alpha=0.7,
            line_color=None, #'black',
            size='radius',
            view=self.umap_view)

        # Test
        background_objects = self.umap_source_view.data['names']
        self.selected_objects, indices = self.get_selected_from_match(background_objects)
        self.get_new_view_keep_selected(background_objects, 
                                        indices)

        # Set selected
        #self.selected_objects, indices = self.get_selected_from_match( self.umap_source_view.data['names'])
        #self.umap_source_view.selected.indices = indices

        # Gallery
        self.plot_gallery()

        '''
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
            low = self.umap_color_mapper.low,
            high = self.umap_color_mapper.high)
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
        # Text
        if init:
            self.DT_text.text = f'DT: {self.prim_DT:.2f}K'
        # Color bar
        if init:
            self.PCB_low.value = f'{np.percentile(self.primary_image,10):.1f}'
            self.PCB_high.value = f'{np.percentile(self.primary_image,90):.1f}'
        self.prim_color_mapper = LinearColorMapper(
            palette="Turbo256", 
            low=float(self.PCB_low.value),
            high=float(self.PCB_high.value))
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

    def plot_gallery(self):
        """ Plot the gallery of images
        """
        #self.spectrum_stacks = []
        for i in range(self.nrow*self.ncol):
            spec_stack = self.gallery_figures[i].image(
                'image', 'x', 'y', 'dw', 'dh', 
                source=self.gallery_data[i],
                color_mapper=self.prim_color_mapper)
            #self.spectrum_stacks.append(spec_stack)
            if self.verbose:
                print(f"Updated gallery {i}")


    def register_callbacks(self):
        # Primary
        self.PCB_low.on_change('value', self.PCB_low_callback)
        self.PCB_high.on_change('value', self.PCB_high_callback)

        # UMAP 
        #self.UCB_low.on_change('value', self.UCB_low_callback)
        #self.UCB_high.on_change('value', self.UCB_high_callback)
        #self.umap_figure.on_event(PanEnd, self.reset_gallery_index)
        self.umap_figure.on_event(
            PanEnd, self.update_umap_filter_event())
        self.umap_source_view.selected.on_change(
            'indices', self.umap_source_callback)     

        self.umap_figure.on_event(
            Reset, self.update_umap_filter_event(reset=True))

    def PCB_low_callback(self, attr, old, new):
        """Fuss with the low value of the main color bar

        Args:
            attr ([type]): [description]
            old ([type]): [description]
            new (str): New value
        """
        print("PCB Low callback")
        self.plot_primary()

    def PCB_high_callback(self, attr, old, new):
        """Fuss with the high value of the main color bar

        Args:
            attr ([type]): [description]
            old ([type]): [description]
            new (str): New value
        """
        self.plot_primary()

    def umap_source_callback(self, attr, old, new):
        print("In umap_source_callback")
        #print(self.umap_source_view.selected.indices)
        pass

    def update_umap_filter_event(self, reset=False):
        """ Callback for UMAP

        Returns:
            callback: 
        """
        def callback(event):
            print('update_umap_filter_event')

            ux = self.umap_source_view.data['xs']
            uy = self.umap_source_view.data['ys']

            if reset:
                self.umap_figure.x_range.start = self.U_xlim[0]
                self.umap_figure.x_range.end = self.U_xlim[1]
                self.umap_figure.y_range.start = self.U_ylim[0]
                self.umap_figure.y_range.end = self.U_ylim[1]

            px_start = self.umap_figure.x_range.start
            px_end = self.umap_figure.x_range.end
            py_start = self.umap_figure.y_range.start
            py_end = self.umap_figure.y_range.end

            if ( (px_start > np.min(ux) ) or
                 (px_end   < np.max(ux) ) or
                 (py_start > np.min(uy) ) or
                 (py_end   < np.max(uy) ) or reset  ):

                background_objects = portal_utils.get_decimated_region_points(
                    self.umap_figure.x_range.start,
                    self.umap_figure.x_range.end,
                    self.umap_figure.y_range.start,
                    self.umap_figure.y_range.end,
                    self.umap_source.data,
                    self.DECIMATE_NUMBER)

                print("umap selected:")
                self.selected_objects, indices = self.get_selected_from_match(background_objects)
                self.get_new_view_keep_selected(background_objects, 
                                                indices)
            else:
                print('Pan event did not require doing anything')
                pass

        return callback


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
            if kk < 0:
                images.append(self.get_im_empty())
                continue
            #
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
        return images

    def get_im_empty(self):
        """ Grab an empty image
        """
        return np.zeros((self.imsize[0], self.imsize[1])).astype(np.uint8)

    def set_gallery_images(self):
        self.gallery_images = []
        ngallery = self.nrow*self.ncol
        nmatch = len(self.match_idx) - self.gallery_index
        #
        if nmatch >= ngallery:
            load_idx = self.match_idx[self.gallery_index:self.gallery_index+ngallery]
        else:
            load_idx = self.match_idx[self.gallery_index:] + [-1]*(ngallery-nmatch)
        # Load
        self.gallery_images = self.load_images(load_idx)
        # Gallery data
        self.gallery_data = []
        for im in self.gallery_images:
            data_source = ColumnDataSource(
                data = {'image':[im], 'x':[0], 'y':[0], 
                    'dw':[self.imsize[0]], 'dh':[self.imsize[1]],
                    'min': [np.min(im)], 'max': [np.max(im)]}
            )
            self.gallery_data.append(data_source)

    def reset_focus(self):
        self.focus_Us = (self.img_Us[0], self.img_Us[1])

    def find_closest(self, Us):
        dist = (Us[0]-self.umap_tbl.US0)**2 + (
            Us[1]-self.umap_tbl.US1)**2
        self.closest = np.argmin(dist)
        self.umap_closest = self.umap_tbl.iloc[self.closest]

    def set_primary_by_U(self, Us):
        # Find closest
        self.find_closest(Us)
        # Set image
        self.primary_image = self.load_images(
            [self.closest])[0]
        # Could change self.imsize here
        self.set_primary_source()
        # DT
        self.prim_DT = self.umap_closest.DT
        self.prim_Us = self.umap_closest.US0, self.umap_closest.US1
        
    def set_matched(self, Us, radius):
        dist = (Us[0]-self.umap_tbl.US0)**2 + (
            Us[1]-self.umap_tbl.US1)**2
        self.match_idx = np.where(dist < radius)[0].tolist()

    def get_selected_from_match(self, objects_in_view):
        """ Return true indices of selected objects
        in the umap_source_view

        Args:
            objects_in_view (_type_): _description_

        Returns:
            np.ndarray, list: _description_
        """
        if len(self.match_idx) == 0:
            return
        rows = catalog.match_ids(
            np.array(self.match_idx),
            np.array(objects_in_view), #self.umap_source_view.data['names']),
            require_in_match=False)
        sel_in_view = rows > 0                                
        indices = rows[sel_in_view]
        selected = np.array(self.match_idx)[sel_in_view]
        # Return
        return selected, rows[sel_in_view].tolist()



# TESTING
if __name__ == "__main__":
    #tmp = OSSinglePortal(None, None)

    # Odd work-around
    def get_session(doc):
        sess = OSSinglePortal(None, None)
        #sess = OSSinglePortal(None, None)
        return sess(doc)

    # Do me!
    server = Server({'/': get_session}, num_procs=1)
    server.start()
    print('Opening Bokeh application for OS data on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()

    '''
    obj = OSSinglePortal(None, None)
    '''
        