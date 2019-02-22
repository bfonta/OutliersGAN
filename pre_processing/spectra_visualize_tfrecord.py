from astropy.io import fits
import tensorflow as tf
import numpy as np
from src.data.data import read_spectra_data as read_data
from src.utilities import PlotGenSamples

def _plot(data, params_data, name, n=5):
    p = PlotGenSamples(nrows=n, ncols=1)
    p.plot_spectra(data[:n], params_data[0][:n], name)

batch_size = 512
dataset_size = 255483
files_path = '/fred/oz012/Bruno/data/spectra/boss/cmass/'
dataset = read_data(files_path+'spectra.tfrecord', 3500)
dataset = dataset.repeat(1).batch(batch_size)
nbatches = int(np.ceil(dataset_size/batch_size))

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    sess.run(iterator.initializer)
    for item in range(5):
        inputs, *params = sess.run(next_element)
        _plot(inputs, params, name='tfrecord_data_'+str(item))

"""
local_path = '/fred/oz012/Bruno/data/spectra/boss/loz/7415/spec-7415-57097-0197.fits'
with fits.open(local_path) as hdu:
    flux_ = hdu[1].data['flux'].astype(np.float32)
    lam_ = np.power( 10, hdu[1].data['loglam'] ).astype(np.float32)
                       
    print(flux_)
    print(min(flux_))
"""
