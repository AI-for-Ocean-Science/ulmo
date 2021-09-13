""" Bokeh portal for Ocean Sciences.  Based on Itamar Reiss' code 
and further modified by Kate Storrey-Fisher"""
from bokeh.models.widgets.tables import StringFormatter
import numpy as np
import os

import h5py
import pandas

from bokeh.layouts import column, gridplot, row
from bokeh.models import ColumnDataSource, Slider
from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature
from bokeh.palettes import Viridis256, Plasma256, Inferno256, Magma256, all_palettes
from bokeh.models import LinearColorMapper, ColorBar, ColumnDataSource, Range1d, CustomJS, Div, \
    CDSView, BasicTicker, \
    IndexFilter, BooleanFilter, Span, Label, BoxZoomTool, TapTool
from bokeh.models.widgets import TextInput, RadioButtonGroup, DataTable, TableColumn, AutocompleteInput, NumberFormatter
from bokeh.models.widgets import Select, Button, Dropdown
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from bokeh.models.annotations import Title
from bokeh.colors import RGB
from bokeh.transform import transform
from bokeh.events import DoubleTap, PinchEnd, PanEnd, Reset

# For the geography figure
from bokeh.tile_providers import get_provider, Vendors
from bokeh.transform import linear_cmap


from IPython import embed

def mercator_coord(lat, lon):
    """Function to switch from lat/long to mercator coordinates

    Args:
        lat (float or np.ndarray): [description]
        lon (float or np.ndarray): [description]

    Returns:
        tuple: x, y values in mercator
    """
    r_major = 6378137.000
    x = r_major * np.radians(lon)
    scale = x/lon
    y = 180.0/np.pi * np.log(np.tan(np.pi/4.0 + 
        lat * (np.pi/180.0)/2.0)) * scale
    return x, y

class os_web(object):
    """ Primary class for the web site """
    def __init__(self, data_dict, config=None, verbose=False): #images, obj_ids, metric_dict, umap_data):

        # Slurp
        self.verbose = verbose
        self.images = data_dict['images']
        self.obj_links = np.arange(self.images.shape[0])

        self.metric_dict = data_dict['metrics']
        self.obj_ids = data_dict['metrics']['obj_ID']

        # Check for geo
        if 'lat' in self.metric_dict.keys() and 'lon' in self.metric_dict.keys():
            self.geo = True
        else:
            print("Geography plotting disabled.  Add lon,lat to your metrics")
            self.geo = False

        # Main data
        if 'xy_values' in data_dict.keys():
            self.xkey, self.ykey = [item.strip() for item in 
                          data_dict['xy_values'].split(',')]
        else:
            self.xkey, self.ykey = 'U0', 'U1'
        self.set_umap_data()

        #
        if self.images.ndim == 3:
            self.nchannel = 1
        else:
            self.nchannel = self.images.shape[-1]
        #
        self.N = len(self.images)
        self.imsize = [self.images[0].shape[0], self.images[0].shape[1]]
        self.reverse_obj_links = self.gen_reverse_obj_links()

        rev_Plasma256 = Plasma256[::-1]
        self.color_mapper = LinearColorMapper(palette=rev_Plasma256, low=0, high=1, 
                                              nan_color=RGB(220, 220, 220, a = 0.1))
        self.high_colormap_factor = 0.1
        self.R_DOT = 6#10
        self.DECIMATE_NUMBER = 5000
        self.UMAP_XYLIM_DELTA = 0.5
        self.umap_on_load = 1 #index of initial umap to load
        self.nof_stacks = 1
        self.n_anomalies = 51
        self.stack_by = 'x'
        self.stacks_colors = ['#fde725', '#90d743', '#35b779', '#21908d', '#31688e', '#443983', '#440154'][::-1]
        self.nrow, self.ncol = 2, 5
        self.first_im_index = 0
        self.zero_gallery = 0

        # TODO -- ELIMINATE THIS
        self.selected_objects = ColumnDataSource(
            data=dict(index=[], score=[], order=[], info_id=[], object_id=[]))

        self.dropdown_dict = {}
        #self.dropdown_dict['embedding'] = 'UMAP'
        self.dropdown_dict['metric'] = 'LL'


        self.init_title_text_tables()
        self.generate_buttons()
        self.generate_sources()
        self.generate_figures()
        self.generate_plots()
        self.register_callbacks()


    def __call__(self, doc):
        doc.add_root(column(row(self.main_title_div ), 
                            row(column(self.info_div), 
                                column(self.umap_figure, self.gallery_figure,
                                    row(self.prev_set, self.next_set), 
                                       self.geo_figure), 
                                column(
                                    row(self.x_text, self.y_text),
                                    self.select_metric,
                                    row(self.main_low, self.main_high),
                                    self.search_object, 
                                    self.data_figure, 
                                    self.selected_objects_table,
                                    self.print_table),
                            )
                                   #row(self.prev_button, self.next_button)),
                            )
                     )
        doc.title = 'OS Web Portal'

    def init_title_text_tables(self):

        self.main_title_div = Div(text='<center>OS Image Visualization Tool</center>', style={'font-size': '299%', 'color': 'black'}, sizing_mode="stretch_width")
        info_text = """
        <p><b>What is this?</b></p>
        <p>This is an interactive tool for visualizing the results of pattern analysis of Ocean Science imagery</p>
        
        <p><b>Cool! How do I use it?</b></p>
        <ul>
            <li>Use the lasso to select a region of sources; these will appear in the bottom squares. 
            <li>The selected objects will also appear in the table on the right; sort them by clicking the column names. Click a row to see the image in the large viewbox.
            <li>Use the zoom tool to see more objects in an area; only a subset are shown (and will be selected) on the main plot.
            <li>Type a object ID in the text box and hit enter to jump to that image on the plot and in the viewbox.
            <li>Double click anywhere on the plot to reset it.
        </ul>
        <p>Find the code for this project on <a href="https://github.com/kstoreyf/anomalies-GAN-HSC">github</a>. Happy object-finding!</p>
        <p><b>Author:</b> X</p>
        <p><i>Adapted from the <a href="https://toast-docs.readthedocs.io/en/latest/">SDSS galaxy portal</a> by Itamar Reis and Kate Storey-Fisher</i></p>
        """
        self.info_div = Div(text=info_text, style={'font-size': '119%', 'color': 'black'})#, sizing_mode="stretch_width")

        self.selected_objects_columns = []
        for key in self.metric_dict.keys():
            # Format
            if key == 'obj_ID':
                formatter=StringFormatter()
            else:
                formatter=NumberFormatter(format='0.00')
            self.selected_objects_columns.append(
                TableColumn(field=key, title=key, 
                            formatter=formatter))

        self.select_object = TextInput(title='Select Object Index:', 
                                       value=str(self.first_im_index))

        self.update_table = Select(title="Inactive", value="",
                                         options=[])
        self.search_object = TextInput(title='Select Object ID:')
        self.main_low = TextInput(title='Main CB Low:', max_width=100)
        self.main_high = TextInput(title='Main CB High:', max_width=100)

        # x,y
        self.x_text = TextInput(title='x:', max_width=100, value=self.xkey)
        self.y_text = TextInput(title='y:', max_width=100, value=self.ykey)

    def gen_reverse_obj_links(self):
        """ makes a dictionary of info_ids to indices

        Returns:
            dict: 
        """
        gl = self.obj_links
        return {str(int(gl[v])): str(v) for v in range(len(gl))}

    def generate_buttons(self):
        """Setup buttons and Dropdown objects
        """
        self.select_score_table = Select(title="Inactive", value="",
                                         options=[])
        self.internal_reset = Select(title="Inactive", value="",
                                         options=[])
        self.print_table = Button(label="Print Table", button_type="default")
        self.next_set = Button(label="Next Set", button_type="default")
        self.prev_set = Button(label="Previous Set", button_type="default")
        # 
        select_metric_menu = []
        for key in self.metric_dict.keys():
            select_metric_menu.append((key,key))
        self.select_metric = Dropdown(label="Color by:", 
                                      button_type="danger", 
                                      menu=select_metric_menu)#, value='Anomaly Score')
        self.dropdown_dict['metric'] = 'LL'

    def generate_figures(self):

        umap_plot_width = 800
        column_width = 500
        self.umap_figure = figure(tools='lasso_select,tap,box_zoom,pan,save,reset',
                                  plot_width=umap_plot_width,
                                  plot_height=600,
                                  toolbar_location="above", output_backend='webgl', )  # x_range=(-10, 10),
        self.umap_colorbar = ColorBar(color_mapper=self.color_mapper, location=(0, 0), 
                                      major_label_text_font_size='15pt', 
                                      label_standoff=13)
        self.umap_figure.add_layout(self.umap_colorbar, 'right')
        self.umap_figure_axes()


        # Snapshot figure
        self.data_figure = figure(tools="box_zoom,save,reset", 
                                  plot_width=column_width,
                                  plot_height=column_width,
                                  toolbar_location="above", 
                                  output_backend='webgl',
                                  x_range=(0,self.imsize[0]), 
                                  y_range=(0,self.imsize[1]))
                                  #x_range=(0,96), y_range=(0,96))

        # Gallery figure 
        title_height = 20
        buffer = 10*self.ncol
        collage_im_width = int((umap_plot_width-buffer)/self.ncol)
        self.gallery_figures = []
        for _ in range(self.nrow*self.ncol):
            sfig = figure(tools="box_zoom,save,reset", 
                          plot_width=collage_im_width, 
                          plot_height=collage_im_width+title_height, 
                          toolbar_location="above", output_backend='webgl', 
                          x_range=(0,self.imsize[0]), y_range=(0,self.imsize[1]))
            self.gallery_figures.append(sfig)

        self.gallery_figure = gridplot(self.gallery_figures, ncols=self.ncol)

        # TODO: make this the index
        t = Title()
        t.text = 'TMP'
        self.data_figure.title = t

        self.remove_ticks_and_labels(self.data_figure)

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
                                  plot_width=umap_plot_width,
                                  plot_height=600,
                                  toolbar_location="above", 
                                  x_axis_type="mercator", 
                                  y_axis_type="mercator", 
                                  x_axis_label = 'Longitude', 
                                  y_axis_label = 'Latitude', 
                                  tooltips=tooltips,
                                  output_backend='webgl', )  # x_range=(-10, 10),

    def generate_plots(self):
        """Generate/init plots
        """

        print("gen plots")
        # Main scatter plot
        self.umap_scatter = self.umap_figure.scatter(
            'xs', 'ys', source=self.umap_source_view,
            color=transform('color_data', self.color_mapper),
            nonselection_fill_color = 'moccasin',
            nonselection_line_color = 'moccasin',
            nonselection_alpha = 1,
            nonselection_line_alpha = 0,
            alpha=0.5,
            line_color=None,
            size='radius',
            view=self.umap_view)

        # Snapshot
        self.plot_snapshot(init=True)

        # Gallery
        self.plot_gallery()
        self.gallery_figure = gridplot(self.gallery_figures, 
                                       ncols=self.ncol)

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


    def generate_sources(self):
        """Load up the various data sources into Bokeh objects
        """

        self.update_color(None)
        # Unpack for convenience
        metric = self.metric_dict[self.dropdown_dict['metric']]
        #embedding = self.dropdown_dict['embedding']

        self.xlim = (np.min(self.umap_source.data['xs']) - self.UMAP_XYLIM_DELTA, np.max(self.umap_source.data['xs']) + self.UMAP_XYLIM_DELTA)
        self.ylim = (np.min(self.umap_source.data['ys']) - self.UMAP_XYLIM_DELTA, np.max(self.umap_source.data['ys']) + self.UMAP_XYLIM_DELTA)

        '''
        self.xlim_all = {}
        self.ylim_all = {}

        for umap in list(self.umap_data.keys()):
            temp_umap_source = ColumnDataSource(
                data=dict(xs=self.umap_data[umap][:, 0],
                          ys=self.umap_data[umap][:, 1],
                          color_data=metric[:],
                          names=list(np.arange(len(metric))),
                          radius=[self.R_DOT] * len(metric)),
                          )
            rxs, rys, _ = get_relevant_objects_coords(temp_umap_source.data)
            temp_xlim = (np.min(rxs) - self.UMAP_XYLIM_DELTA, np.max(rxs) + self.UMAP_XYLIM_DELTA)
            temp_ylim = (np.min(rys) - self.UMAP_XYLIM_DELTA, np.max(rys) + self.UMAP_XYLIM_DELTA)
            self.xlim_all[umap] = temp_xlim
            self.ylim_all[umap] = temp_ylim
        '''

        points = get_decimated_region_points(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1],
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
        self.umap_view = CDSView(source=self.umap_source_view)

        im = self.process_image(
            self.images[self.first_im_index])
        xsize, ysize = self.imsize
        self.data_source = ColumnDataSource(
            data = {'image':[im], 'x':[0], 'y':[0], 
                    'dw':[xsize], 'dh':[ysize],
                    'min': [np.min(im)], 'max': [np.max(im)]}
        )

        # Collage of empty images
        xsize, ysize = self.imsize
        self.stacks_sources = []
        count = 0
        ncollage = self.nrow*self.ncol
        im_empty = self.get_im_empty()
        while count < ncollage:
            source = ColumnDataSource(
                data = {'image':[im_empty], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
            )
            self.stacks_sources.append(source)
            count += 1
        self.stacks_source = ColumnDataSource(
            data = {'image':[im], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
        )

        # Generate the data table for the sources
        cdata = dict(index=[])
        for key in self.metric_dict.keys():
            cdata[key] = []
        self.selected_objects_source = ColumnDataSource(cdata)

        # Locations
        self.search_galaxy_source = ColumnDataSource(dict(
            xs=[self.umap_data[0, 0]],
            ys=[self.umap_data[0, 1]],
            #xs=[self.umap_data[embedding][0, 0]],
            #ys=[self.umap_data[embedding][0, 1]],
        ))

        # Geography coords
        #  The items need to include what is in the tooltips above
        geo_dict = dict(mercator_x=[], mercator_y=[])
        for key in self.metric_dict.keys():
            geo_dict[key] = []
        self.geo_source = ColumnDataSource(geo_dict)

    def process_image(self, im:np.ndarray):
        """Prep the image for plotting
        Allows for RGB

        No change for a 'grey-scale' image

        Args:
            im (np.ndarray): Input image

        Returns:
            np.ndarray: prepped image
        """
        if self.nchannel > 1:
            # This is necessary for displaying image properly
            alpha = np.full((im.shape[0], im.shape[1], 1), 255)
            im = np.concatenate((im, alpha), axis=2)
        else:
            return im

    def get_im_empty(self):
        """ Grab an empty image
        """
        if self.nchannel == 1:
            return np.zeros((self.imsize[0], self.imsize[1])).astype(np.uint8)
        else:
            return np.zeros((self.imsize[0], self.imsize[1], self.nchannel+1)).astype(np.uint8)

    def register_reset_on_double_tap_event(self, obj):
        """ Deal with a double click
        """
        obj.js_on_event(DoubleTap, CustomJS(args=dict(p=obj), code="""
            p.reset.emit()
            console.debug(p.x_range.start)
            """))

    def register_callbacks(self):
        """ Register all of the callback code
        """
        
        self.register_reset_on_double_tap_event(self.umap_figure)
        self.register_reset_on_double_tap_event(self.data_figure)
        for i in range(len(self.gallery_figures)):
            self.register_reset_on_double_tap_event(self.gallery_figures[i])

        # Buttons
        self.print_table.on_click(self.print_table_callback)
        self.prev_set.on_click(self.prev_set_callback)
        self.next_set.on_click(self.next_set_callback)

        # Non-dropdowns
        self.select_object.on_change('value', self.select_object_callback())
        self.search_object.on_change('value', self.search_object_callback())
        self.main_low.on_change('value', self.main_low_callback)
        self.main_high.on_change('value', self.main_high_callback)

        self.x_text.on_change('value', self.x_text_callback)

        # Dropdown's
        self.select_metric.on_click(self.update_color)
        '''
        self.select_umap.on_change('value', self.update_umap_figure())

        self.select_spectrum_plot_type.on_change('value', self.select_galaxy_callback())
        self.select_nof_stacks.on_change('value', self.select_nof_stacks_callback())
        self.select_stack_by.on_change('value', self.select_stack_by_callback())

        self.select_colormap.on_click(self.select_colormap_callback)
        '''

        self.umap_figure.on_event(PanEnd, self.reset_gallery_index)
        self.umap_figure.on_event(PanEnd, self.select_stacks_callback())

        self.selected_objects_source.selected.on_change(
            'indices', self.selected_objects_callback)

        self.umap_source_view.selected.on_change('indices', 
                                                 self.umap_source_callback)     

        self.select_score_table.js_on_change('value', CustomJS(
            args=dict(s1=self.umap_source_view, s2=self.selected_objects_source, 
                      s4=self.obj_links, s5=self.obj_ids), code="""
                    var inds = s1.attributes.selected['1d'].indices
                    var d1 = s1.data;
                    var d2 = s2.data;
                    d2.index = []
                    d2.score = []
                    d2.order = []
                    d2.info_id = []
                    d2.object_id = []
                    for (var i = 0; i < inds.length; i++) {
                        d2.index.push(d1['names'][inds[i]])
                        d2.score.push(d1['color_data'][inds[i]])
                        d2.order.push(0.0)
                        d2.info_id.push(s4[d1['names'][inds[i]]])
                        d2.object_id.push(s5[d1['names'][inds[i]]])
                    }
                    console.log(d2.index)
                    console.log('select_score_table_js')
                    s2.change.emit();
                """))

        ### !!!NOTE!!! careful with semicolons here, all vars except last need one!
        self.update_table.js_on_change('value', CustomJS(
            args=dict(s1=self.umap_source_view, 
                      s2=self.selected_objects_source, 
                      s3=self.selected_objects), code="""
                    var d2 = s2.data;
                    console.log(s3.attributes.data.index);
                    var selected_objects = s3.attributes.data.index;
                    var score = s3.attributes.data.score;
                    var order = s3.attributes.data.order;
                    var info_id = s3.attributes.data.info_id;
                    var object_id = s3.attributes.data.object_id
                    d2.index = []
                    d2.score = []
                    d2.order = []
                    d2.info_id = []
                    d2.object_id = []
                    var inds = []
                    console.log(selected_objects);
                    for (var i = 0; i < selected_objects.length; i++) {
                        inds.push(i)
                        d2.index.push(selected_objects[i])
                        d2.score.push(score[i])
                        d2.order.push(order[i])
                        d2.info_id.push(info_id[i])
                        d2.object_id.push(object_id[i])
                    }
                    s1.attributes.selected['1d'].indices = inds
                    s1.attributes.selected.attributes.indices = inds
                    console.log(s1)
                    console.log('update_table_js')
                    s2.change.emit();
                    s1.change.emit();
                """))

        self.internal_reset.js_on_change('value', CustomJS(
            args=dict(p=self.umap_figure), code="""
                        p.reset.emit()
                        """))

        self.umap_figure.on_event(PanEnd, self.update_umap_filter_event())
        self.umap_figure.on_event(Reset, self.update_umap_filter_reset())


    def reset_gallery_index(self):
        """ Reset stack index
        """
        self.gallery_index = 0


    def remove_ticks_and_labels(self, figure):
        """Simple cleaning method

        Args:
            figure (bokeh.figure): 
        """
        figure.xaxis.major_label_text_font_size = "0pt"
        figure.xaxis.axis_label_text_font_size = "0pt"
        figure.yaxis.major_label_text_font_size = "0pt"

        figure.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
        figure.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks

        figure.yaxis.major_tick_line_color = None  # turn off y-axis major ticks
        figure.yaxis.minor_tick_line_color = None  # turn off y-axis minor tick

    def set_colormap(self, metric:np.ndarray, metric_key:str):
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

        self.color_mapper.high = high
        self.color_mapper.low = low

        return

    def x_text_callback(self, attr, old, new):
        """Fuss with the low value of the main color bar

        Args:
            attr ([type]): [description]
            old ([type]): [description]
            new (str): New value
        """
        if new not in self.metric_dict.keys():
            print("Bad choice for x, not in metric dict")
            self.x_text.value = old 
        else:
            self.x_key = new
            self.set_umap_data()

    def main_low_callback(self, attr, old, new):
        """Fuss with the low value of the main color bar

        Args:
            attr ([type]): [description]
            old ([type]): [description]
            new (str): New value
        """
        self.color_mapper.low = float(new)

    def main_high_callback(self, attr, old, new):
        """Fuss with the high value of the main color bar

        Args:
            attr ([type]): [description]
            old ([type]): [description]
            new (str): New value
        """
        self.color_mapper.high = float(new)

    def print_table_callback(self, event):
        """Callback to print the table

        Args:
            event ([type]): [description]
        """
        if len(self.selected_objects_source.data['index']) == 0:
            print("Need to generate a Table first (with the lasso)!")
            return
        # Generate a pandas table
        df = pandas.DataFrame(dict(self.selected_objects_source.data))
        # Write
        df.to_csv('OS_viz.csv')
        print("Table written to: OS_viz.csv")

    def prev_set_callback(self, event):
        """Callback to previous gallery images

        Args:
            event ([type]): [description]
        """
        if len(self.selected_objects_source.data['index']) == 0:
            print("Need to use the lasso first!")
            return
        # 
        self.gallery_index -= self.nrow*self.ncol
        self.gallery_index = max(self.gallery_index, 0)
        print(f"Previous set of gallery images; zero={self.gallery_index}")
        self.stacks_callback()

    def next_set_callback(self, event):
        """Callback to next gallery images

        Args:
            event ([type]): [description]
        """
        if len(self.selected_objects_source.data['index']) == 0:
            print("Need to use the lasso first!")
            return
        # 
        nobj = len(self.selected_objects.data['index'])
        self.gallery_index += self.nrow*self.ncol
        self.gallery_index = min(self.gallery_index, nobj-1)
        print(f"Next set of gallery images; zero={self.gallery_index}")
        self.stacks_callback()

    def select_object_callback(self):
        if self.verbose:
            print('select object')
        def callback(attr, old, new):
            index = self.select_object.value
            index_str = str(index)
            if ',' in index_str:
                if self.verbose:
                    print('list input')
                selected_objects = index_str.replace(' ','').split(',')
                selected_objects = [int(s) for s in selected_objects]

                backgroud_objects = self.umap_source_view.data['names']
                self.get_new_view_keep_selected(backgroud_objects, selected_objects)

                return
            if self.verbose:
                print('galaxy callback')
            specobjid = str(self.search_object.value)
            new_specobjid = str(int(self.obj_links[int(index)]))
            #new_specobjid = str(
            #logger.debug(type(specobjid), specobjid, type(new_specobjid), new_specobjid)
            #print(specobjid, new_specobjid)
            if specobjid != new_specobjid:
                if self.verbose:
                    print('not equal')
                self.search_object.value = new_specobjid
            else:
                if self.verbose:
                    print('Update snapshot from select')
                self.update_snapshot()

        return callback

    def set_umap_data(self):
        self.umap_data = np.array([self.metric_dict[self.xkey], 
                                       self.metric_dict[self.ykey]]).T

    def search_object_callback(self):
        """ Call back method for search_obj
        """
        if self.verbose:
            print('search obj')
        def callback(attr, old, new):
            #logger.debug(self.search_object.value)
            objid_str = str(self.search_object.value)
            if ',' in objid_str:
                if self.verbose:
                    print('list input')
                selected_objects_ids = objid_str.replace(' ','').split(',')
                index_str = str(self.reverse_galaxy_links[selected_objects_ids[0]])
                for idx, specobjid in enumerate(selected_objects_ids[1:]):
                    index_str = '{}, {}'.format(index_str, str(self.reverse_galaxy_links[specobjid]))
                self.select_galaxy.value = index_str
                return

            if objid_str in self.reverse_obj_links:
                if self.verbose:
                    print('search galaxy')
                index = str(self.select_object.value)
                new_index = str(self.reverse_obj_links[objid_str])
                self.update_search_circle(new_index)
                if self.verbose:
                    print('search galaxy - updated circle')
                #logger.debug(type(index), index, type(new_index), new_index)
                if index != new_index:
                    self.select_object.value = new_index
                else:
                    print('Update obj from search')
                    self.update_snapshot()
                    self.plot_gallery()  # Resets color map

        return callback



    # TODO: can this be simplified?
    def select_stacks_callback(self):
        def callback(event):
            self.stacks_callback()
        return callback

    def selected_objects_callback(self, attr, old, new):
        """ Callback for indices changing """
        self.select_object.value = str(self.selected_objects.data['index'][new[0]])
    
    def umap_source_callback(self, attr, old, new):
        # Init
        tdict = {}
        for key in ['index']+list(self.metric_dict.keys()):
            tdict[key] = []
        # Fill
        for ii in new:
            idx = self.umap_source_view.data['names'][ii]
            # Loop on selected objects and galaxy table data
            # Selected objects
            tdict['index'].append(idx)
            # Metrics
            for key in self.metric_dict.keys():
                tdict[key].append(self.metric_dict[key][idx])
        # Update
        self.selected_objects_source.data = tdict.copy()
        self.selected_objects.data = tdict.copy()

        # Geo map
        if self.geo:
            mercator_x, mercator_y =  mercator_coord(
                np.array(tdict['lat']), np.array(tdict['lon']))
            #print(mercator_y)
            geo_dict = dict(mercator_x=mercator_x.tolist(), 
                            mercator_y=mercator_y.tolist())
            # Metrics
            for key in tdict.keys():
                geo_dict[key] = tdict[key]
            self.geo_source.data = geo_dict.copy()

    def stacks_callback(self):
        """ Callback method for lasso objects
        """
        if self.verbose: 
            print('select_objects_callback')
        selected_objects = self.selected_objects.data['index']
        selected_inds = np.array([int(s) for s in selected_objects])
        nof_selected = selected_inds.size
        inds_visible = selected_inds[self.gallery_index:self.gallery_index+
                                     self.nrow*self.ncol]

        xsize, ysize = self.imsize
        self.stacks_sources = []
        count = 0
        im_empty = self.get_im_empty()
        while count < self.nrow*self.ncol:
            if count < len(inds_visible):
                ind = inds_visible[count]
                im = self.process_image(self.images[ind])
                info_id = self.obj_links[ind]
                new_title = '{}'.format(int(info_id))
            else:
                im = im_empty
                new_title = ' '

            source = ColumnDataSource(
                data = {'image':[im], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
            )
            self.stacks_sources.append(source)
            self.gallery_figures[count].title.text = new_title
            #self.spectrum_stacks[count].data_source.data = dict(self.stacks_sources[count].data)

            count += 1
        self.plot_gallery()


    def plot_snapshot(self, init=False):
        """ Plot the current image with color mapper and bar
        """
        self.snap_color_mapper = LinearColorMapper(palette="Turbo256", 
                                         low=self.data_source.data['min'][0],
                                         high=self.data_source.data['max'][0])
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

    def update_snapshot(self):
        """ Update the Zoom-in image
        """
        if self.verbose:
            print("update snapshot")
        #TODO: BE CAREFUL W INDEX VS ID
        index = self.select_object.value
        specobjid = int(self.obj_links[int(index)])

        im = self.process_image(self.images[int(index)])
        xsize, ysize = self.imsize
        self.data_source = ColumnDataSource(
            data = {'image':[im], 'x':[0], 'y':[0], 
                    'dw':[xsize], 'dh':[ysize],  
                    'min': [np.min(im)], 'max': [np.max(im)]}
        )
        
        # More expensive
        self.plot_snapshot()
        self.data_figure.title.text = '{}'.format(int(specobjid))
        #self.link_div.text='<center>View galaxy in <a href="{}" target="_blank">{}</a></center>'.format(
        #    self.obj_link(int(self.select_galaxy.value)), 'SDSS object explorer')

        return

    def update_color(self, event):
        """ Update the color bar

        Args:
            event (bokey event): 
        """
        # Update?
        if event is not None:
            self.dropdown_dict['metric'] = event.item
            self.umap_figure_axes()

        metric_key = self.dropdown_dict['metric']
        metric = self.metric_dict[metric_key]
        #embedding = self.dropdown_dict['embedding']

        self.set_colormap(metric, metric_key)
        # Set limits
        self.main_low.value = str(self.color_mapper.low)
        self.main_high.value = str(self.color_mapper.high)

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

        selected_objects = self.selected_objects.data['index']
        background_objects = self.umap_source_view.data['names']

        self.get_new_view_keep_selected(background_objects, selected_objects)
        #self.select_score_table.value = self.select_score.value


    def update_umap_filter_event(self):
        """ Callback for UMAP

        Returns:
            callback: 
        """
        if self.verbose:
            print("update umap")
        def callback(event):
            print('update_umap_filter_event')

            ux = self.umap_source_view.data['xs']
            uy = self.umap_source_view.data['ys']

            px_start = self.umap_figure.x_range.start
            px_end = self.umap_figure.x_range.end
            py_start = self.umap_figure.y_range.start
            py_end = self.umap_figure.y_range.end

            if ( (px_start > np.min(ux) ) or
                 (px_end   < np.max(ux) ) or
                 (py_start > np.min(uy) ) or
                 (py_end   < np.max(uy) )   ):

                background_objects = get_decimated_region_points(
                    self.umap_figure.x_range.start,
                    self.umap_figure.x_range.end,
                    self.umap_figure.y_range.start,
                    self.umap_figure.y_range.end,
                    self.umap_source.data,
                    self.DECIMATE_NUMBER)

                print("umap selected:")
                selected_objects = self.selected_objects.data['index']
                self.get_new_view_keep_selected(background_objects, selected_objects)
            else:
                print('Pan event did not require doing anything')
                pass

        return callback

    def get_new_view_keep_selected(self, background_objects, 
                                   selected_objects_, custom_sd = None):
        """ Handle selected objects

        Args:
            background_objects ([type]): [description]
            selected_objects_ ([type]): [description]
            custom_sd ([type], optional): [description]. Defaults to None.
        """


        #embedding = self.dropdown_dict['embedding']
        print('get_new_view_keep_selected')
        _, _, is_relevant = get_relevant_objects_coords(self.umap_source.data)
        selected_objects = [s for s in selected_objects_ if is_relevant[int(s)]]
        selected_objects = np.array(selected_objects)
        background_objects = np.array(background_objects)

        nof_selected_objects = selected_objects.size

        max_nof_selected_objects = int(self.DECIMATE_NUMBER)/2
        if nof_selected_objects > max_nof_selected_objects:
            nof_selected_objects = max_nof_selected_objects
            new_objects = selected_objects[:nof_selected_objects]
        else:
            new_objects = np.concatenate([selected_objects, background_objects])
            new_objects, order = np.unique(new_objects, return_index=True)
            new_objects = new_objects[np.argsort(order)]
            new_objects = new_objects[:self.DECIMATE_NUMBER]

        new_objects = new_objects.astype(int)
        if custom_sd is None:
            metric = self.metric_dict[self.dropdown_dict['metric']]
        else:
            metric = custom_sd

        self.umap_source_view = ColumnDataSource(
                data=dict(xs=self.umap_data[new_objects, 0],
                          ys=self.umap_data[new_objects, 1],
                          color_data=metric[new_objects],
                          names=list(new_objects),
                          radius=[self.R_DOT] * len(new_objects),
                        ))
        self.points = np.array(new_objects)
        self.umap_scatter.data_source.data = dict(self.umap_source_view.data)

        if nof_selected_objects > 0:
            new_dict = dict(index=list(selected_objects))
            for key in self.metric_dict.keys():
                new_dict[key] = [self.metric_dict[key][s] for s in selected_objects]
            self.selected_objects.data = new_dict
            #order = np.array([float(o) for o in self.selected_objects.data['order']])
            #self.selected_objects.data = dict(
            #    index=list(selected_objects), 
            #    score=[-999999 if np.isnan(metric[s]) else metric[s] for s in selected_objects],
            #    order=list(order), 
            #    info_id=[self.obj_links[s] for s in selected_objects],
            #    object_id=[self.obj_ids[s] for s in selected_objects]
            #)
            self.update_table.value = str(np.random.rand())
        elif len(selected_objects_) > 0:
            self.selected_objects = ColumnDataSource(data=dict(index=[], score=[], order=[], info_id=[], object_id=[]))
            self.update_table.value = str(np.random.rand())
            self.internal_reset.value = str(np.random.rand())
        else:
            self.update_table.value = str(np.random.rand())

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

        return

    def update_search_circle(self, index):
        """ Move the search circle

        Args:
            index ([type]): [description]
        """
        if self.verbose:
            print("update search circle")
        self.search_galaxy_source.data = dict(
            xs=[self.umap_source.data['xs'][int(index)]],
            ys=[self.umap_source.data['ys'][int(index)]],
        )
        return

    def update_umap_filter_reset(self):
        """ Reset the x-y window
        """
        def callback(event):
            if self.verbose:
                print('reset double tap')

            background_objects = get_decimated_region_points(
                self.xlim[0],
                self.xlim[1],
                self.ylim[0],
                self.ylim[1],
                self.umap_source.data,
                self.DECIMATE_NUMBER)

            selected_objects = self.selected_objects.data['index']
            self.get_new_view_keep_selected(background_objects, selected_objects)

            self.umap_figure.x_range.start = self.xlim[0]

            self.umap_figure.x_range.end = self.xlim[1]

            self.umap_figure.y_range.start = self.ylim[0]

            self.umap_figure.y_range.end = self.ylim[1]

            xsize, ysize = self.imsize
            self.stacks_sources = []
            count = 0
            ncollage = self.nrow*self.ncol
            im_empty = self.get_im_empty()
            while count < ncollage:
                source = ColumnDataSource(
                    data = {'image':[im_empty], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
                )
                self.stacks_sources.append(source)
                self.gallery_figures[count].title.text = ' '
                self.spectrum_stacks[count].data_source.data = dict(self.stacks_sources[count].data)

                count += 1

        return callback


    def umap_figure_axes(self):
        """ Set the x-y axes
        """

        #embedding_name = self.dropdown_dict['embedding']
        embedding_name = f"{self.xkey}, {self.ykey}"
        metric_name = self.dropdown_dict['metric']
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

            # self.umap_figure.yaxis.axis_label = 'Log(OIII / Hb)'
            # if 'NII' in embedding_name:
            #     self.umap_figure.xaxis.axis_label = 'Log(NII / Ha)'
            # elif 'OI' in embedding_name:
            #     self.umap_figure.xaxis.axis_label = 'Log(OI / Ha)'
            # elif 'SII' in embedding_name:
            #     self.umap_figure.xaxis.axis_label = 'Log(SII / Ha)'

        return


def get_region_points(x_min, x_max, y_min, y_max, datasource):
    """ Get the points within the box

    Args:
        x_min ([type]): [description]
        x_max ([type]): [description]
        y_min ([type]): [description]
        y_max ([type]): [description]
        datasource ([type]): [description]

    Returns:
        [type]: [description]
    """
    IGNORE_TH = -9999
    xs = np.array(datasource['xs'])
    ys = np.array(datasource['ys'])
    cd = datasource['color_data']
    nof_objects = len(cd)
    if True:
        is_in_box = np.logical_and.reduce([xs >= x_min, xs <= x_max, ys >= y_min, ys <= y_max, ys > IGNORE_TH, xs > IGNORE_TH])
    else:
        is_in_box = np.logical_and.reduce([xs >= x_min, xs <= x_max, ys >= y_min, ys <= y_max])
    return np.where(is_in_box)[0]


def get_relevant_objects_coords(datasource):
    """ ??

    Args:
        datasource ([type]): [description]

    Returns:
        [type]: [description]
    """
    IGNORE_TH = -999
    xs = np.array(datasource['xs'])
    ys = np.array(datasource['ys'])
    relevant_objects = np.logical_and.reduce([ys > IGNORE_TH, xs > IGNORE_TH])
    return xs[relevant_objects], ys[relevant_objects], relevant_objects


def get_decimated_region_points(x_min, x_max, y_min, y_max, datasource, DECIMATE_NUMBER):
    """ ??

    Args:
        x_min ([type]): [description]
        x_max ([type]): [description]
        y_min ([type]): [description]
        y_max ([type]): [description]
        datasource ([type]): [description]
        DECIMATE_NUMBER ([type]): [description]

    Returns:
        [type]: [description]
    """
    is_in_box_inds = get_region_points(x_min, x_max, y_min, y_max, datasource)
    print('total points before decimation', len(is_in_box_inds))
    if len(is_in_box_inds) < DECIMATE_NUMBER:
        return is_in_box_inds
    random_objects_ = np.random.choice(is_in_box_inds, DECIMATE_NUMBER, replace=False)
    random_objects = [datasource['names'][r] for r in random_objects_]
    return random_objects



# CODE HERE AND DOWN IS FOR TESTING
def get_test_os_session(doc):
    images, objids, metric, umapd = grab_dum_data()
    # Instantiate
    sess = os_web(images, objids, metric, umapd)
    return sess(doc)

def get_modis_subset_os_session(doc):
    data_dict = grab_modis_subset()
    # Instantiate
    sess = os_web(data_dict)
    return sess(doc)

def grab_dum_data():
    nobj = 100
    dum_images = np.random.uniform(size=(nobj, 64, 64))
    dum_objids = np.arange(nobj)
    dum_umap = np.random.uniform(size=(nobj,2)) 
    dum_LL = np.random.uniform(low=0., high=100., size=nobj)
    dum_metric = dict(LL=dum_LL)
    dum_umapd = dict(UMAP=dum_umap)
    embed(header='NEED TO REFACTOR')
    #
    return dum_images, dum_objids, dum_metric, dum_umapd

def grab_modis_subset():
    # Load up 
    sst_dir='/data/Projects/Oceanography/AI/OOD/SST/MODIS_L2/PreProc/'
    data_file = os.path.join(sst_dir, 
                             'MODIS_R2019_2010_95clear_128x128_preproc_std.h5')
    results_path = '/data/Projects/Oceanography/AI/SSL/portal/'
    umaps_path = os.path.join(results_path, 'embeddings')

    # Images
    nimgs = 100000
    sub_idx = np.arange(nimgs)
    f = h5py.File(data_file, 'r') 
    images = f["valid"][sub_idx,0,:,:]
    f.close()

    # UMAP
    umap_file = os.path.join(umaps_path, 'UMAP_2010_valid_v1.npz')
    f = np.load(umap_file, allow_pickle=False)
    e1, e2 = f['e1'], f['e2']

    # Metrics
    results_file = os.path.join(results_path, 'ulmo_2010_valid_v1.parquet')
    res = pandas.read_parquet(results_file)
    metric_dict = {'LL': res.LL.values[sub_idx], 
                   'lat': res.lat.values[sub_idx],
                   'lon': res.lon.values[sub_idx],
                   'avgT': res.mean_temperature.values[sub_idx],
                   'DT': (res.T90-res.T10).values[sub_idx],
                   'obj_ID': sub_idx,
    }

    # Repack
    data_dict = {
        'images': images,
        'xy_scatter': dict(UMAP=(np.array([e1, e2]).T)[sub_idx]),
        'metrics': metric_dict,
    }

    return data_dict


def main(flg):
    flg = int(flg)

    # Deprecated test
    if flg & (2 ** 0):
        pass
    
    # Test main class
    if flg & (2 ** 1):
        dum_images, dum_objids, dum_metric, dum_umapd = grab_dum_data()
        sess = os_web(dum_images, dum_objids, dum_metric, dum_umapd)
        print("Success!")

    # Real deal
    if flg & (2 ** 2):
        server = Server({'/': get_test_os_session}, num_procs=1)
        server.start()
        print('Opening Bokeh application for test data on http://localhost:5006/')

        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()

    # Test modis subset
    if flg & (2 ** 3):
        data_dict = grab_modis_subset()
        sess = os_web(data_dict)

    if flg & (2 ** 4):
        server = Server({'/': get_modis_subset_os_session}, num_procs=1)
        server.start()
        print('Opening Bokeh application for MODIS subset on http://localhost:5006/')

        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        flg = 0
        #flg += 2 ** 0  # Test bokeh
        #flg += 2 ** 1  # Test object
        #flg += 2 ** 2  # Full Test 
        #flg += 2 ** 3  # Test load MODIS subset
        flg += 2 ** 4  # Full MODIS subset
    else:
        flg = sys.argv[1]

    main(flg)
