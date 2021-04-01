import numpy as np
import pandas as pd

from bokeh.layouts import layout
from bokeh.embed import file_html

from bokeh.io import show
from bokeh.io import output_notebook

from bokeh.models import Text
from bokeh.models import Title
from bokeh.models import Plot
from bokeh.models import Slider
from bokeh.models import Circle
from bokeh.models import Range1d
from bokeh.models import CustomJS
from bokeh.models import HoverTool
from bokeh.models import LinearAxis
from bokeh.models import ColumnDataSource
from bokeh.models import SingleIntervalTicker

from bokeh.palettes import Spectral6
from bokeh.plotting import figure, show
from bokeh.layouts import layout

WIDTH, HEIGHT = 2000, 400
NPOINTS = 3500
NEPOCHS = 1000

def add_axis(fig):
    color = '#000066'
    AXIS_FORMATS = dict(
        minor_tick_in=None,
        minor_tick_out=None,
        major_tick_in=None,
        major_label_text_font_size="11pt",
        major_label_text_font_style="normal",
        axis_label_text_font_size="11pt",
        axis_label_text_font_style="normal",
        axis_line_color=color,
        major_tick_line_color=color,
        major_label_text_color=color,
        major_tick_line_cap="round",
        axis_line_cap="round",
        axis_line_width=1.5,
        major_tick_line_width=1.5,
    )

    angstrom = u'\u212b'
    xaxis = LinearAxis(
        ticker     = SingleIntervalTicker(interval=500),
        axis_label = "Wavelength [" + angstrom + "]",
        **AXIS_FORMATS
    )
    yaxis = LinearAxis(
        ticker     = SingleIntervalTicker(interval=.25),
        axis_label = "Flux",
        bounds = (-1.01,1.01),
        **AXIS_FORMATS
    )

    fig.add_layout(xaxis, 'below')
    fig.add_layout(yaxis, 'left')


def get_data():
    x = np.arange(0,NPOINTS)
    y = np.random.normal(loc=0, scale=.2, size=(NEPOCHS,NPOINTS))
    return x, y
    
arr_x, arr_y = get_data()

epochs = [x for x in range(NEPOCHS)]
sources = dict()
xname = 'wavelength'
yname = 'flux'
for epoch in range(NEPOCHS):
    sources['source'+str(epoch)] = ColumnDataSource(data={xname: arr_x.tolist(),
                                                 yname: arr_y[epoch].tolist()})


def create_figure():
    p = figure(plot_width=WIDTH, plot_height=HEIGHT)
    p.axis.visible = False
    p.title = Title(text='Generated Spectra', text_font_size='12pt')
    p.y_range = Range1d(-1,1)
    return p

p = create_figure()
add_axis(p)

renderer_source = sources['source0']
graph_color = '#660044'
p.circle(x=xname, y=yname, size=1, color=graph_color, source=renderer_source)
p.line(x=xname, y=yname, color=graph_color, source=renderer_source)

dict_of_sources = dict(zip([x for x in epochs], ['source{}'.format(x) for x in epochs]))
js_source_array = str(dict_of_sources).replace("'", "")

code = """
    var epoch = slider.value;
    var sources = {js_source_array};
    var new_source_data = sources[epoch].data;
    renderer_source.data = new_source_data;
    renderer_source.change.emit();
""".format(js_source_array=js_source_array)

callback = CustomJS(args=sources, code=code)

slider = Slider(start=0, end=NEPOCHS-1, value=0, step=1, title="Epoch", width=WIDTH)
slider.js_on_change("value", callback)

callback.args["renderer_source"] = renderer_source
callback.args["slider"] = slider


display = layout([[p], [slider]])
show(display)
