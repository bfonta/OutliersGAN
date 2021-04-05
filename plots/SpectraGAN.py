import os, copy
import numpy as np
import pandas as pd
import h5py

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

FILENAME = os.path.basename(__file__) + '.hdf5'
WIDTH, HEIGHT = 2000, 400
NPOINTS = 3500
NEPOCHS = 3
NSTATS = 2000
NBINS = 100

def add_axis(fig, xlabel, ylabel, xinterval=500, yinterval=.25, xbounds=None, ybounds=None):
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

    xaxis = LinearAxis(
        ticker     = SingleIntervalTicker(interval=xinterval),
        axis_label = xlabel,
        **AXIS_FORMATS
    )
    if xbounds:
        xaxis.bounds = xbounds

    yaxis = LinearAxis(
        ticker     = SingleIntervalTicker(interval=yinterval),
        axis_label = ylabel,
        **AXIS_FORMATS
    )
    if ybounds:
        yaxis.bounds = ybounds

    fig.add_layout(xaxis, 'below')
    fig.add_layout(yaxis, 'left')

def create_figure(w, h, title, xlims=None, ylims=None):
    p = figure(plot_width=w, plot_height=h)
    p.axis.visible = False
    p.title = Title(text=title, text_font_size='12pt')
    if ylims:
        p.y_range = Range1d(ylims[0],ylims[1])
    if xlims:
        p.x_range = Range1d(xlims[0],xlims[1])
    return p

def add_graph(fig, xname, yname, renderer_source, color):
    fig.circle(x=xname, y=yname, size=1, color=color, source=renderer_source)
    fig.line(x=xname, y=yname, color=color, source=renderer_source)

def add_bars(fig, xname, yname, width, renderer_source, color):
    fig.vbar(x=xname, top=yname, bottom=0, width=width, fill_color=color, source=renderer_source)

def add_vline(fig, xname, yname, width, renderer_source, color):
    fig.line(x=xname, y=yname, line_width=width, color=color, source=renderer_source)

def get_normal_data():
    x = np.arange(0,NPOINTS)
    y = np.random.normal(loc=0, scale=.2, size=(NEPOCHS,NPOINTS))
    return x, y

def get_stats_data():
    x = np.random.normal(loc=1, scale=.2, size=(NEPOCHS,NSTATS))
    y = np.random.normal(loc=1, scale=.2, size=(NEPOCHS,NSTATS))
    return x, y

with h5py.File(FILENAME, 'w') as f1:
    arr_x, arr_y = get_normal_data()
    arr_mean, arr_std = get_stats_data()
    dset1 = f1.create_dataset("wavelength", (1,NPOINTS), dtype='f', data=arr_x)
    dset2 = f1.create_dataset("flux", (NEPOCHS,NPOINTS), dtype='f', data=arr_y)
    dset3 = f1.create_dataset("means", (NEPOCHS,NSTATS), dtype='f', data=arr_mean)
    dset4 = f1.create_dataset("stds", (NEPOCHS,NSTATS), dtype='f', data=arr_std)

with h5py.File(FILENAME, 'r') as f2:
    #print(list(f2.keys()))
    arr_x =    f2['wavelength'][0,:]
    arr_y =    f2['flux'][:]
    arr_mean = f2['means'][:]
    arr_std =  f2['stds'][:]

epochs = [x for x in range(NEPOCHS)]
s1, s2 = dict(), dict()
xs1, ys1 = 'wavelength', 'flux'
xs2, ys2 = ('means', 'stds'), ('meanscounts', 'stdscounts')
xse, yse = ('mean1x', 'mean2x'), ('mean1y', 'mean2y')

hstdmax, hmeanmax = -999, -999
hstdmin, hmeanmin = 999, 999
for epoch in range(NEPOCHS):
    s1name, s2name = 'source1_{}'.format(epoch), 'source2_{}'.format(epoch)
    s1[s1name] = ColumnDataSource(data={xs1: arr_x.tolist(),
                                        ys1: arr_y[epoch].tolist()})

    hmean, edgmean = np.histogram(arr_mean[epoch], NBINS)
    cmean = (edgmean[:-1] + edgmean[1:])/2
    hstd, edgstd = np.histogram(arr_std[epoch], NBINS)
    cstd = (edgstd[:-1] + edgstd[1:])/2
    s2[s2name] = ColumnDataSource(data={xs2[0]: cmean.tolist(),
                                        ys2[0]: hmean.tolist(),
                                        xs2[1]: cstd.tolist(),
                                        ys2[1]: hstd.tolist()})
    if np.max(hstd) > hstdmax:
        hstdmax = np.max(hstd)
    if np.min(hstd) < hstdmin:
        hstdmin = np.min(hstd)
    if np.max(hmean) > hmeanmax:
        hmeanmax = np.max(hmean)
    if np.min(hmean) < hmeanmin:
        hmeanmin = np.min(hmean)

sextraname = 'sextra'
sextra = ColumnDataSource(data={xse[0]: [1, 1], yse[0]: [hmeanmin, hmeanmax],
                                xse[1]: [1, 1], yse[1]: [hstdmin, hstdmax]})

dict_sources_1 = dict(zip(epochs, ['source1_{}'.format(x) for x in epochs]))
dict_sources_2 = dict(zip(epochs, ['source2_{}'.format(x) for x in epochs]))
js_sources_1 = str(dict_sources_1).replace("'", "")
js_sources_2 = str(dict_sources_2).replace("'", "")
js_sources_e = sextraname

p1 = create_figure(WIDTH, HEIGHT, 'Generated Spectrum', ylims=[-1,1])
add_axis(p1, 'Wavelength [' + u'\u212b' + ']', 'Flux', ybounds=(-2.01,1.01))
add_graph(p1, xs1, ys1, s1['source1_0'], '#660044')

p_mean = create_figure(int(WIDTH/2), HEIGHT, 'Generated Means', xlims=[0.,2.], ylims=[hmeanmin,hmeanmax])
add_axis(p_mean, 'Means [flux units]', 'Counts', xinterval=.25, yinterval=10)
add_bars(p_mean, xs2[0], ys2[0], edgmean[1]-edgmean[0], s2['source2_0'], '#660044')
add_vline(p_mean, xse[0], yse[0], 3., sextra, '#008000')

p_std = create_figure(int(WIDTH/2), HEIGHT, 'Generated Standard Deviations', xlims=[0.,2.], ylims=[hstdmin,hstdmax])
add_axis(p_std, 'Standard Deviations [flux units]', 'Counts', xinterval=.25, yinterval=10)
add_bars(p_std, xs2[1], ys2[1], edgstd[1]-edgstd[0], s2['source2_0'], '#660044')
add_vline(p_std, xse[1], yse[1], 3., sextra, '#008000')

code = """
    var epoch = slider.value;
    var s2 = {js_sources_2};
    var new_s2 = s2[epoch].data;
    s2_update.data = new_s2;
    s2_update.change.emit();
""".format(js_sources_1=js_sources_1, js_sources_2=js_sources_2,
           js_sources_e=js_sources_e,
           vlinemax1=hmeanmax, vlinemax2=hstdmax)

"""
    var s1 = {js_sources_1};
    var se = {js_sources_e};
    var new_s1 = s1[epoch].data;
    s1_update.data = new_s1;
    s1_update.change.emit();

    var d = s2_update.data;

    var ncounts1_update = 0;
    var mean1_update = 0.;
    for (var i = 0; i<d['means'].length; i++) {{
        mean1_update += d['meanscounts'][i] * d['means'][i];
        ncounts1_update += d['meanscounts'][i];
    }}
    mean1_update /= ncounts1_update;
    se.data['mean1x'][0] = mean1_update;
    se.data['mean1x'][1] = mean1_update;
    se.data['mean1y'][0] = 0;
    se.data['mean1y'][1] = {vlinemax1};

    var ncounts2_update = 0;
    var mean2_update = 0.;
    for (var i = 0; i<d['means'].length; i++) {{
        mean2_update += d['meanscounts'][i] * d['means'][i];
        ncounts2_update += d['meanscounts'][i];
    }}
    mean2_update /= ncounts2_update;
    se.data['mean2x'][0] = mean2_update;
    se.data['mean2x'][1] = mean2_update;
    se.data['mean2y'][0] = 0;
    se.data['mean2y'][1] = {vlinemax2};
    se.change.emit();
"""

callback_args = copy.deepcopy(s1)
callback_args.update(s2)
callback_args.update({sextraname: sextra})
callback = CustomJS(args=callback_args, code=code)
print(callback_args)
    
slider = Slider(start=0, end=NEPOCHS-1, value=0, step=1, title="Epoch", width=WIDTH)
slider.js_on_change("value", callback)

callback.args['s1_update'] = s1['source1_0']
callback.args['s2_update'] = s2['source2_0'] #SOME WEIRD BUG: DOES NOT WORK WITH 'source2_1', for instance
callback.args['slider'] = slider

display = layout([[p1], [p_mean, p_std], [slider]])
show(display)
