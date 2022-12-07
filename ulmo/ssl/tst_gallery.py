""" Code for running the OS Portal for a single image"""
import numpy as np
import os

import h5py

from bokeh.plotting import figure
from bokeh.transform import transform
from bokeh.layouts import column, gridplot, row
from bokeh.models import LinearColorMapper, ColorBar, ColumnDataSource, Range1d, CustomJS, Div, \
    CDSView, BasicTicker
from bokeh.core.enums import MarkerType
from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Scatter
from bokeh.models.annotations import Title
from bokeh.models.widgets import TextInput, DataTable, TableColumn, NumberFormatter

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

        self.debug = False

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

        # Glyph figure
        self.gen_glyph_plot()
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
        #self.plot_gallery()  # Resets color map

    def __call__(self, doc):
        '''
        doc.add_root(column(#row(self.main_title_div ), 
                            #row(#column(self.info_div), 
                                column(self.other_gallery_figure,
                                    #row(self.prev_set, self.next_set), 
                                    self.glyph_plot,
                            #        ), 
                            )
        ))
        '''
        doc.add_root(column(self.main_title_div, 
                            row(self.umap_figure, self.glyph_plot), 
                            self.gallery_figure,
                            row(self.prev_set, self.next_set), 
        ))
        doc.title = 'OS Gallery'

    '''
    def generate_figures(self):
        self.umap_plot_width = 800
        column_width = 500

        # Gallery
        self.init_other_gallery_figure()
        self.init_gallery_figure()

        #for i in range(len(self.gallery_figures)):
        #    self.remove_ticks_and_labels(self.gallery_figures[i])
        #    t = Title()
        #    t.text = ' '
        #    self.gallery_figures[i].title = t

        # Table
        self.selected_objects_table = DataTable(
            source=self.selected_objects_source,
            columns=self.selected_objects_columns,
            width=column_width,
            height=200,
            scroll_to_selection=False)
    '''

    def init_gallery_figure(self):
        # Gallery figure 
        title_height = 20
        buffer = 10*self.ncol
        collage_im_width = int((self.umap_plot_width-buffer)/self.ncol)
        self.gallery_figures = []
        for kk in range(self.nrow*self.ncol):
            sfig = figure(tools="box_zoom,save,reset", 
                          width=collage_im_width, 
                          height=collage_im_width+title_height, 
                          toolbar_location="above", #output_backend='webgl', 
                          x_range=(0,self.imsize[0]), y_range=(0,self.imsize[1]))
            self.gallery_figures.append(sfig)

        self.gallery_figure = gridplot(self.gallery_figures, ncols=self.ncol)

    def init_other_gallery_figure(self):

        def rimg(scl=1.):
            return scl*np.random.rand(64,64)

        collage_im_width = 250
        title_height = 20
        imsize = (64, 64)

        gallery_figs = []
        nrow = 2
        ncol = 3
        for _ in range(nrow*ncol):
            sfig = figure(tools="box_zoom,save,reset", 
                    width=collage_im_width, 
                    height=collage_im_width+title_height, 
                    toolbar_location="above", output_backend='webgl', 
                    x_range=(0,imsize[0]), y_range=(0,imsize[1]))
            gallery_figs.append(sfig)

        self.other_gallery_figure = gridplot(gallery_figs, ncols=ncol)

        snap_color_mapper = LinearColorMapper(palette="Turbo256", 
                                                low=-1., #self.data_source.data['min'][0],
                                                high=1.) #self.data_source.data['max'][0])

        # Fill
        stacks_sources = []
        xsize, ysize = imsize
        for _ in range(nrow*ncol):
            source = ColumnDataSource(
                        data = {'image':[rimg(scl=10.)], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
                    )
            stacks_sources.append(source)

        for i in range(nrow*ncol):
            print(f"Init fig: {np.std(stacks_sources[i].data['image'])}")
            spec_stack = gallery_figs[i].image(
                        'image', 'x', 'y', 'dw', 'dh', 
                        source=stacks_sources[i],
                        color_mapper=snap_color_mapper)


#    def generate_plots(self):
#        """Generate/init plots
#        """
#        # Gallery
#        self.plot_gallery()
#        #self.gallery_figure = gridplot(self.gallery_figures, 
#        #                               ncols=self.ncol)

    def plot_gallery(self):
        """ Plot the gallery of images
        """
        if self.debug:
            return
        self.snap_color_mapper = LinearColorMapper(palette="Turbo256", 
                                         low=self.data_source.data['min'][0],
                                         high=self.data_source.data['max'][0])
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

    def gen_glyph_plot(self):
        N = len(MarkerType)
        x = np.linspace(-2, 2, N)
        y = x**2
        markers = list(MarkerType)

        source = ColumnDataSource(dict(x=x, y=y, 
                                       markers=markers))

        self.glyph_plot = Plot(
            title=None, width=300, height=300,
            min_border=0, toolbar_location=None)

        glyph = Scatter(x="x", y="y", size=20, fill_color="#74add1", marker="markers")
        self.glyph_plot.add_glyph(source, glyph)

        xaxis = LinearAxis()
        self.glyph_plot.add_layout(xaxis, 'below')

        yaxis = LinearAxis()
        self.glyph_plot.add_layout(yaxis, 'left')

        self.glyph_plot.add_layout(Grid(dimension=0, ticker=xaxis.ticker))
        self.glyph_plot.add_layout(Grid(dimension=1, ticker=yaxis.ticker))

    def register_callbacks(self):
        """ Register all of the callback code
        """
        
        #self.register_reset_on_double_tap_event(self.umap_figure)
        #self.register_reset_on_double_tap_event(self.data_figure)
        #for i in range(len(self.gallery_figures)):
        #    self.register_reset_on_double_tap_event(self.gallery_figures[i])

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

        #self.umap_figure.on_event(PanEnd, self.reset_gallery_index)
        #self.umap_figure.on_event(PanEnd, self.select_stacks_callback())

        self.selected_objects_source.selected.on_change(
            'indices', self.selected_objects_callback)


        #self.internal_reset.js_on_change('value', CustomJS(
        #    args=dict(p=self.umap_figure), code="""
        #                p.reset.emit()
        #                """))

        #self.umap_figure.on_event(PanEnd, self.update_umap_filter_event())
        #self.umap_figure.on_event(Reset, self.update_umap_filter_reset())


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
        