""" Bokeh portal for Ocean Sciences.  Based on Itamar Reiss' code 
and further modified by Kate Storrey-Fisher"""
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

from IPython import embed

class os_web(object):
    """ Primary class for the web site """
    def __init__(self, images, obj_ids, metric_dict, umap_data):
        # Slurp
        self.images = images
        self.obj_ids = obj_ids
        self.obj_links = obj_ids
        self.metric_dict = metric_dict
        self.umap_data = umap_data
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

        self.selected_objects = ColumnDataSource(
            data=dict(index=[], score=[], order=[], info_id=[], object_id=[]))

        self.dropdown_dict = {}
        self.dropdown_dict['embedding'] = 'UMAP'
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
                                column(self.umap_figure, self.gallery_figure), 
                                column(self.search_object, 
                                       self.data_figure, 
                                       self.selected_galaxies_table), #, self.title_div  
                            )
                                   #row(self.prev_button, self.next_button)),
                            )
                     )
        doc.title = 'OS Web Portal'

    def init_title_text_tables(self):

        self.main_title_div = Div(text='<center>OS Image Visualization Tool</center>', style={'font-size': '299%', 'color': 'black'}, sizing_mode="stretch_width")
        info_text = """
        <p><b>What is this?</b></p>
        <p>This is an interactive tool for visualizing the results of an anomaly detection approach on a set of galaxies. Each dot is a galaxy image from the Hyper Suprime-Cam survey. The distribution is a UMAP, a 1D representation of the images. The colors show how anomalous our generative adversarial network (GAN) found each galaxy. Choose the UMAP embedding and the colormapping on the right; the default is UMAP on autencoded residuals, colored by total anomaly score.</p>
        
        <p><b>Cool! How do I use it?</b></p>
        <ul>
            <li>Use the lasso to select a region of galaxies; these will appear in the bottom squares. Scroll through them with the 'Previous' and 'Next' buttons.
            <li>The selected galaxies will also appear in the table on the right; sort them by clicking the column names. Click a row to see the image in the large viewbox.
            <li>Use the zoom tool to see more galaxies in an area; only a subset are shown (and will be selected) on the main plot.
            <li>Type a galaxy ID in the text box and hit enter to jump to that image on the plot and in the viewbox.
            <li>Double click anywhere on the plot to reset it.
        </ul>
        <p>Find the code for this project on <a href="https://github.com/kstoreyf/anomalies-GAN-HSC">github</a>. Happy weird-galaxy-finding!</p>
        <p><b>Author:</b> Kate Storey-Fisher</p>
        <p><i>Adapted from the <a href="https://toast-docs.readthedocs.io/en/latest/">SDSS galaxy portal</a> by Itamar Reis</i></p>
        """
        self.info_div = Div(text=info_text, style={'font-size': '119%', 'color': 'black'})#, sizing_mode="stretch_width")

        self.selected_galaxies_columns = [
            TableColumn(field="index", title="Index"),
            TableColumn(field="info_id", title="Info ID"),
            TableColumn(field="object_id", title="Object ID"),
            TableColumn(field="score", title="Score", formatter=NumberFormatter(format = '0.0000')),
        ]
        self.select_object = TextInput(title='Select Object Index:', value=str(self.first_im_index))

        self.update_table = Select(title="Inactive", value="",
                                         options=[])
        self.search_object = TextInput(title='Select Object Info ID:')

    def gen_reverse_obj_links(self):
        """ makes a dictionary of info_ids to indices

        Returns:
            dict: 
        """
        gl = self.obj_links
        return {str(int(gl[v])): str(v) for v in range(len(gl))}

    def generate_buttons(self):
        self.select_score_table = Select(title="Inactive", value="",
                                         options=[])
        self.internal_reset = Select(title="Inactive", value="",
                                         options=[])

    def generate_figures(self):

        umap_plot_width = 800
        column_width = 350
        #taptool = TapTool(callback=self.select_galaxy_callback)
        self.umap_figure = figure(tools='lasso_select,tap,box_zoom,save,reset',
                                  plot_width=umap_plot_width,
                                  plot_height=600,
                                  #title='UMAP',
                                  toolbar_location="above", output_backend='webgl', )  # x_range=(-10, 10),
        #self.umap_figure.add_tools(taptool)

        # y_range=(-10, 10))
        #self.umap_figure.toolbar.active_scroll = 'auto'
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
            sfig = figure(tools="box_zoom,save,reset", plot_width=collage_im_width, plot_height=collage_im_width+title_height, 
            toolbar_location="above", output_backend='webgl', 
            x_range=(0,self.imsize[0]), 
            y_range=(0,self.imsize[1]))
            self.gallery_figures.append(sfig)

        self.gallery_figure = gridplot(self.gallery_figures, ncols=self.ncol)


        self.umap_colorbar = ColorBar(color_mapper=self.color_mapper, location=(0, 0), major_label_text_font_size='15pt', label_standoff=13)
        self.umap_figure.add_layout(self.umap_colorbar, 'right')

        self.umap_figure_axes()

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

        # Galaxy Table
        self.selected_galaxies_table = DataTable(
            source=self.selected_galaxies_source,
            columns=self.selected_galaxies_columns,
            width=column_width,
            height=200,
            scroll_to_selection=False)

    def generate_plots(self):
        print("gen plots")
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

        self.plot_snapshot()
        #color_mapper = LinearColorMapper(palette="Turbo256", 
        #                                 low=self.data_source.data['min'][0],
        #                                 high=self.data_source.data['max'][0])
        #self.data_image = self.data_figure.image(
        #    'image', 'x', 'y', 'dw', 'dh', source=self.data_source,
        #    color_mapper=color_mapper)
        
        #self.data_image.add_layout(color_bar, 'right')

        color_mapper = LinearColorMapper(palette="Turbo256", 
                                         low=self.data_source.data['min'][0],
                                         high=self.data_source.data['max'][0])
        self.spectrum_stacks = []
        for i in range(self.nrow*self.ncol):
            #spec_stack = self.gallery_figures[i].image_rgba(
            spec_stack = self.gallery_figures[i].image(
                'image', 'x', 'y', 'dw', 'dh', 
                source=self.stacks_sources[i],
                color_mapper=color_mapper)
            self.spectrum_stacks.append(spec_stack)

        self.gallery_figure = gridplot(self.gallery_figures, 
                                       ncols=self.ncol)

        self.umap_search_galaxy = self.umap_figure.circle(
            'xs', 'ys', source=self.search_galaxy_source, alpha=0.5,
            color='tomato', size=self.R_DOT*4, line_color="black", line_width=2)

        LINE_ARGS = dict(color="#3A5785", line_color=None)


    def generate_sources(self):
        """Load up the various data sources into Bokeh objects
        """

        # Unpack for convenience
        metric = self.metric_dict[self.dropdown_dict['metric']]
        embedding = self.dropdown_dict['embedding']

        self.set_colormap(metric)
        print('generate sources')
        self.umap_source = ColumnDataSource(
            #data=dict(xs=self.umap_data[self.select_umap.value][:, 0],
            #          ys=self.umap_data[self.select_umap.value][:, 1],
            data=dict(xs=self.umap_data[embedding][:, 0],
                      ys=self.umap_data[embedding][:, 1],
                      color_data=metric[:],
                      names=list(np.arange(len(metric))),
                      radius=[self.R_DOT] * len(metric)),
                      )

        self.xlim = (np.min(self.umap_source.data['xs']) - self.UMAP_XYLIM_DELTA, np.max(self.umap_source.data['xs']) + self.UMAP_XYLIM_DELTA)
        self.ylim = (np.min(self.umap_source.data['ys']) - self.UMAP_XYLIM_DELTA, np.max(self.umap_source.data['ys']) + self.UMAP_XYLIM_DELTA)

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

        points = get_decimated_region_points(self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1],
                                             self.umap_source.data, self.DECIMATE_NUMBER)

        self.umap_source_view = ColumnDataSource(
            data=dict(xs=self.umap_data[embedding][points, 0],
                      ys=self.umap_data[embedding][points, 1],
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

        self.selected_galaxies_source = ColumnDataSource(dict(
            index=[],
            score=[],
            order=[],
            info_id=[],
            object_id=[]
        ))

        self.search_galaxy_source = ColumnDataSource(dict(
            xs=[self.umap_data[embedding][0, 0]],
            ys=[self.umap_data[embedding][0, 1]],
        ))

    def process_image(self, im):
        if self.nchannel > 1:
            # This is necessary for displaying image properly
            alpha = np.full((im.shape[0], im.shape[1], 1), 255)
            im = np.concatenate((im, alpha), axis=2)
        else:
            return im

    def get_im_empty(self):
        if self.nchannel == 1:
            return np.zeros((self.imsize[0], self.imsize[1])).astype(np.uint8)
        else:
            return np.zeros((self.imsize[0], self.imsize[1], self.nchannel+1)).astype(np.uint8)

    def register_reset_on_double_tap_event(self, obj):
        obj.js_on_event(DoubleTap, CustomJS(args=dict(p=obj), code="""
            p.reset.emit()
            console.debug(p.x_range.start)
            """))

    def register_callbacks(self):
        
        self.register_reset_on_double_tap_event(self.umap_figure)
        self.register_reset_on_double_tap_event(self.data_figure)
        for i in range(len(self.gallery_figures)):
            self.register_reset_on_double_tap_event(self.gallery_figures[i])

        '''
        # Buttons
        self.show_anomalies.on_click(self.show_anomalies_callback)
        self.next_button.on_click(self.next_stack_index)
        self.prev_button.on_click(self.prev_stack_index)        
        '''

        # Non-dropdowns
        self.select_object.on_change('value', self.select_object_callback())
        self.search_object.on_change('value', self.search_object_callback())

        # Dropdown's
        #self.select_score.on_change('value', self.update_color())
        '''
        self.select_umap.on_change('value', self.update_umap_figure())

        self.select_spectrum_plot_type.on_change('value', self.select_galaxy_callback())
        self.select_nof_stacks.on_change('value', self.select_nof_stacks_callback())
        self.select_stack_by.on_change('value', self.select_stack_by_callback())

        self.select_colormap.on_click(self.select_colormap_callback)
        '''

        self.umap_figure.on_event(PanEnd, self.reset_stack_index)
        self.umap_figure.on_event(PanEnd, self.select_stacks_callback())

        #self.umap_figure.on_event(PanEnd, self.debug())

        self.selected_galaxies_source.selected.on_change(
            'indices', self.selected_objects_callback)
        '''
        self.selected_galaxies_source.selected.js_on_change(
            'indices', 
            CustomJS(
                args=dict(s1=self.selected_galaxies_source, 
                          sg=self.select_object), 
                code="""
                    var inds = s1.attributes.selected['1d'].indices
                    if (inds.length > 0) {
                    sg.value = String(s1.data.index[inds[0]]);
                    }
                    console.log(s1);
                    console.log('selected_galaxies_source_js')
                    """))
        '''

        # TODO -- Need to put this back!!
        self.umap_source_view.selected.on_change('indices', 
                                                 self.umap_source_callback)     

        self.select_score_table.js_on_change('value', CustomJS(
            args=dict(s1=self.umap_source_view, s2=self.selected_galaxies_source, 
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
                      s2=self.selected_galaxies_source, 
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


    def reset_stack_index(self):
        self.stack_index = 0


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

    def set_colormap(self, metric:np.ndarray):
        """Set the color map for the given metric

        Args:
            metric (np.ndarray): Metric of interest, e.g. LL
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

        self.color_mapper.high = high
        self.color_mapper.low = low

        return

    def select_object_callback(self):
        print('select object')
        def callback(attr, old, new):
            index = self.select_object.value
            index_str = str(index)
            if ',' in index_str:
                print('list input')
                selected_objects = index_str.replace(' ','').split(',')
                selected_objects = [int(s) for s in selected_objects]

                backgroud_objects = self.umap_source_view.data['names']
                self.get_new_view_keep_selected(backgroud_objects, selected_objects)

                return
            print('galaxy callback')
            specobjid = str(self.search_object.value)
            new_specobjid = str(int(self.obj_links[int(index)]))
            #new_specobjid = str(
            #logger.debug(type(specobjid), specobjid, type(new_specobjid), new_specobjid)
            #print(specobjid, new_specobjid)
            if specobjid != new_specobjid:
                print('not equal')
                self.search_object.value = new_specobjid
            else:
                print('Update snapshot from select')
                self.update_snapshot()

        return callback


    def search_object_callback(self):
        print('search obj')
        def callback(attr, old, new):
            #logger.debug(self.search_object.value)
            objid_str = str(self.search_object.value)
            if ',' in objid_str:
                print('list input')
                selected_objects_ids = objid_str.replace(' ','').split(',')
                index_str = str(self.reverse_galaxy_links[selected_objects_ids[0]])
                for idx, specobjid in enumerate(selected_objects_ids[1:]):
                    index_str = '{}, {}'.format(index_str, str(self.reverse_galaxy_links[specobjid]))
                self.select_galaxy.value = index_str
                return

            if objid_str in self.reverse_obj_links:
                print('search galaxy')
                index = str(self.select_object.value)
                new_index = str(self.reverse_obj_links[objid_str])
                self.update_search_circle(new_index)
                print('search galaxy - updated circle')
                #logger.debug(type(index), index, type(new_index), new_index)
                if index != new_index:
                    self.select_object.value = new_index
                else:
                    print('Update obj from search')
                    self.update_snapshot()

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
        for key in ['index', 'score', 'order', 'info_id', 'object_id']:
            tdict[key] = []
        # Fill
        for ii in new:
            idx = self.umap_source_view.data['names'][ii]
            # Loop on selected objects and galaxy table data
            '''
            for obj in [self.selected_objects, 
                        self.selected_galaxies_source]:
                obj.data['index'].append(idx)
                obj.data['info_id'].append(
                    self.obj_links[idx])
                obj.data['object_id'].append(
                    self.obj_ids[idx])
                obj.data['score'].append(
                    self.umap_source_view.data['color_data'][ii])
                obj.data['order'].append(0.0)
            '''
            # Selected objects
            tdict['index'].append(idx)
            tdict['info_id'].append(
                self.obj_links[idx])
            tdict['object_id'].append(
                self.obj_ids[idx])
            tdict['score'].append(
                self.umap_source_view.data['color_data'][ii])
            tdict['order'].append(0.0)
        # Update
        self.selected_galaxies_source.data = tdict.copy()
        #self.selected_galaxies_table.source.data = tdict.copy()
        self.selected_objects.data = tdict.copy()

    def stacks_callback(self):
        
        print('select_stacks_callback')
        selected_objects = self.selected_objects.data['index']
        selected_inds = np.array([int(s) for s in selected_objects])
        nof_selected = selected_inds.size
        inds_visible = selected_inds[self.stack_index:self.stack_index+self.nrow*self.ncol]

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
            self.spectrum_stacks[count].data_source.data = dict(self.stacks_sources[count].data)

            count += 1


    def plot_snapshot(self):
        color_mapper = LinearColorMapper(palette="Turbo256", 
                                         low=self.data_source.data['min'][0],
                                         high=self.data_source.data['max'][0])
        self.data_image = self.data_figure.image(
            'image', 'x', 'y', 'dw', 'dh', source=self.data_source,
            color_mapper=color_mapper)
        #color_bar = ColorBar(color_mapper=color_mapper, ticker= BasicTicker(),
        #             location=(0,0))

    def update_snapshot(self):
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



    def update_umap_filter_event(self):
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

        embedding = self.dropdown_dict['embedding']
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
                data=dict(xs=self.umap_data[embedding][new_objects, 0],
                          ys=self.umap_data[embedding][new_objects, 1],
                          color_data=metric[new_objects],
                          names=list(new_objects),
                          radius=[self.R_DOT] * len(new_objects),
                        ))
        self.points = np.array(new_objects)
        self.umap_scatter.data_source.data = dict(self.umap_source_view.data)

        if nof_selected_objects > 0:
            order = np.array([float(o) for o in self.selected_objects.data['order']])
            self.selected_objects.data = dict(
                index=list(selected_objects), 
                score=[-999999 if np.isnan(metric[s]) else metric[s] for s in selected_objects],
                order=list(order), 
                info_id=[self.obj_links[s] for s in selected_objects],
                object_id=[self.obj_ids[s] for s in selected_objects]
            )
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
        print("update search circle")
        self.search_galaxy_source.data = dict(
            xs=[self.umap_source.data['xs'][int(index)]],
            ys=[self.umap_source.data['ys'][int(index)]],
        )
        return

    def update_umap_filter_reset(self):
        def callback(event):
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

        #embedding_name = self.select_umap.value
        embedding_name = self.dropdown_dict['embedding']
        metric_name = self.dropdown_dict['metric']
        if metric_name == 'No color':
            self.umap_figure.title.text  = '{}'.format(embedding_name)
        else:
            self.umap_figure.title.text  = '{} - Colored by {}'.format(embedding_name , metric_name)
        self.umap_figure.title.text_font_size = '17pt'

        # TODO -- Fix this
        if 'ulmo' in embedding_name:
            self.umap_figure.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
            self.umap_figure.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks

            self.umap_figure.yaxis.minor_tick_line_color = None  # turn off y-axis major ticks
            self.umap_figure.yaxis.major_tick_line_color = None  # turn off y-axis minor ticks

            self.umap_figure.xaxis.major_label_text_font_size = "0pt"
            self.umap_figure.yaxis.major_label_text_font_size = "0pt"

            self.umap_figure.xaxis.axis_label_text_font_size = "0pt"
            self.umap_figure.yaxis.axis_label_text_font_size = "0pt"
        else:
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
    IGNORE_TH = -999
    xs = np.array(datasource['xs'])
    ys = np.array(datasource['ys'])
    relevant_objects = np.logical_and.reduce([ys > IGNORE_TH, xs > IGNORE_TH])
    return xs[relevant_objects], ys[relevant_objects], relevant_objects


def get_decimated_region_points(x_min, x_max, y_min, y_max, datasource, DECIMATE_NUMBER):
    is_in_box_inds = get_region_points(x_min, x_max, y_min, y_max, datasource)
    print('total points before decimation', len(is_in_box_inds))
    if len(is_in_box_inds) < DECIMATE_NUMBER:
        return is_in_box_inds
    random_objects_ = np.random.choice(is_in_box_inds, DECIMATE_NUMBER, replace=False)
    random_objects = [datasource['names'][r] for r in random_objects_]
    return random_objects




class test_portal(object):
    """ Test object for development"""
    def __init__(self):
        self.df = sea_surface_temperature.copy()
        self.source = ColumnDataSource(data=self.df)

        self.plot = figure(x_axis_type='datetime', y_range=(0, 25), y_axis_label='Temperature (Celsius)',
                    title="Sea Surface Temperature at 43.18, -70.43")
        self.plot.line('time', 'temperature', source=self.source)

        self.slider = Slider(start=0, end=30, value=0, step=1, title="Smoothing by N Days")
        self.slider.on_change('value', self.callback)

    def __call__(self, doc):
        doc.add_root(column(self.slider, self.plot))
        doc.theme = Theme(filename="theme.yaml")

    def callback(self, attr, old, new):
        if new == 0:
            data = self.df
        else:
            data = self.df.rolling(f"{new}D").mean()
        self.source.data = ColumnDataSource.from_df(data)

def get_test_session(doc):
    sess = test_portal()
    return sess(doc)

def get_test_os_session(doc):
    images, objids, metric, umapd = grab_dum_data()
    # Instantiate
    sess = os_web(images, objids, metric, umapd)
    return sess(doc)

def get_modis_subset_os_session(doc):
    images, objids, metric, umapd = grab_modis_subset()
    # Instantiate
    sess = os_web(images, objids, metric, umapd)
    return sess(doc)

def grab_dum_data():
    nobj = 100
    dum_images = np.random.uniform(size=(nobj, 64, 64))
    dum_objids = np.arange(nobj)
    dum_umap = np.random.uniform(size=(nobj,2)) 
    dum_LL = np.random.uniform(low=0., high=100., size=nobj)
    dum_metric = dict(LL=dum_LL)
    dum_umapd = dict(UMAP=dum_umap)
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
    nimgs = 10000
    sub_idx = np.arange(nimgs)
    f = h5py.File(data_file, 'r') 
    images = f["valid"][sub_idx,0,:,:]
    f.close()

    # UMAP
    umap_file = os.path.join(umaps_path, 'UMAP_2010_valid_v1.npz')
    f = np.load(umap_file, allow_pickle=False)
    e1, e2 = f['e1'], f['e2']
    umap_dict = {}
    umap_dict['UMAP'] = (np.array([e1, e2]).T)[sub_idx]

    # Metrics
    results_file = os.path.join(results_path, 'ulmo_2010_valid_v1.parquet')
    res = pandas.read_parquet(results_file)
    metric_dict = {'LL': res.LL.values[sub_idx]}

    return images, sub_idx, metric_dict, umap_dict


def main(flg):
    flg = int(flg)
    if flg & (2 ** 0):
        server = Server({'/': get_test_session}, num_procs=1)
        server.start()
        print('Opening Bokeh application on http://localhost:5006/')

        server.io_loop.add_callback(server.show, "/")
        server.io_loop.start()
    
    # Test class
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
        dum_images, dum_objids, dum_metric, dum_umapd = grab_modis_subset()
        sess = os_web(dum_images, dum_objids, dum_metric, dum_umapd)

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
