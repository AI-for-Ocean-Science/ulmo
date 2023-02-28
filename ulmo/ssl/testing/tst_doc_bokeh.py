# imports
import numpy as np

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, LinearColorMapper, LinearAxis
from bokeh.layouts import column, gridplot, row

from bokeh.io import output_notebook

from bokeh.core.enums import MarkerType
from bokeh.io import curdoc, show
from bokeh.models import ColumnDataSource, Grid, LinearAxis, Plot, Scatter

from bokeh.server.server import Server  # THIS NEEDS TO STAY!

class TstBokeh(object):

    def __init__(self, verbose=True): 

        def rimg(scl=1.):
            return scl*np.random.rand(64,64)
        dummy_img = rimg()
        dummy_img.shape

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

        self.gallery_figure = gridplot(gallery_figs, ncols=ncol)

        snap_color_mapper = LinearColorMapper(palette="Turbo256", 
                                                low=-1., #self.data_source.data['min'][0],
                                                high=1.) #self.data_source.data['max'][0])

        stacks_sources = []
        xsize, ysize = imsize
        for _ in range(nrow*ncol):
            source = ColumnDataSource(
                        data = {'image':[rimg(scl=10.)], 'x':[0], 'y':[0], 'dw':[xsize], 'dh':[ysize]}
                    )
            stacks_sources.append(source)

        for i in range(nrow*ncol):
            spec_stack = gallery_figs[i].image(
                        'image', 'x', 'y', 'dw', 'dh', 
                        source=stacks_sources[i],
                        color_mapper=snap_color_mapper)



        N = len(MarkerType)
        x = np.linspace(-2, 2, N)
        y = x**2
        markers = list(MarkerType)

        source = ColumnDataSource(dict(x=x, y=y, markers=markers))

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

        #show(column(plot, gallery_figure))


    def __call__(self, doc):
        doc.add_root(column(self.glyph_plot, self.gallery_figure))
        doc.title = 'OS Web Portal'

# TESTING
if __name__ == "__main__":
    #tmp = OSSinglePortal(None, None)

    # Odd work-around
    def get_session(doc):
        sess = TstBokeh()
        #sess = OSSinglePortal(None, None)
        return sess(doc)

    # Do me!
    server = Server({'/': get_session}, num_procs=1)
    server.start()
    print('Opening Bokeh application for OS data on http://localhost:5006/')

    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
        