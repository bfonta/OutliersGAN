from astropy.io import fits
import tensorflow as tf
import numpy as np
from src.data.tfrecords import read_spectra_data as read_data
from src.utilities import PlotGenSamples

def _plot(data, params_data, name, n=5):
    p = PlotGenSamples(nrows=n, ncols=1)
    p.plot_spectra(data[:n], params_data[0][:n], name)

batch_size = 1
dataset_size = 22791
files_path = '/fred/oz012/Bruno/data/spectra/qso_zWarning/'
dataset = read_data(files_path+'spectra2_.tfrecord', 3500)
dataset = dataset.repeat(1).batch(batch_size)
nbatches = int(np.ceil(dataset_size/batch_size))

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

counter = 0
with tf.Session() as sess:
    sess.run(iterator.initializer)
    for item in range(2):
        inputs, *params = sess.run(next_element)
        counter += 1
        if counter%10000==0:
            print(counter)
        _plot(inputs, params, name='tfrecord_data_'+str(item))

