import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os
import glob
import numpy as np
import tensorflow as tf
from scipy import interpolate

def write_to_file(fname, *args):
    with open(fname, 'a') as f:
        for x, *data in zip(*args):
            f.write('{},'.format(x))
            for i in range(len(args)-1):
                if i==len(args)-2:
                    f.write('{}\n'.format(data[i]))
                else:
                    f.write('{},'.format(data[i]))

def read_from_file(fname, splitter=','):
    with open(fname) as f:
        for values in f:
            values = values.split(splitter)
            ncols = len(values)
            break
        data = [[] for i in range(ncols)]
    with open(fname) as f:
        for values in f:
            values = values.replace('\n','').split(splitter)
            for i,val in enumerate(values):
                data[i].append(float(val))
    return [data[i] for i in range(ncols)]

def plot_predictions(pred, name):
    """
    Only works if 'pred' is a list of lists. 
    Each nested list must contain the predictions for a dataset
    """
    for i in range(len(pred)):
        plt.scatter(np.arange(len(pred[i])), pred[i], label=str(i))
    plt.legend()
    plt.savefig('/fred/oz012/Bruno/figs/{}.png'.format(name))
    plt.close()

def log_tf_files(layers_names, loss, scope):
    gr = tf.get_default_graph()
    num_layers = len(layers_names)
    for i in range(num_layers):
        weight = gr.get_tensor_by_name(scope+'/'+layers_names[i]+'/kernel:0')
        grad = tf.gradients(loss, weight)[0]
        mean = tf.reduce_mean(tf.abs(grad))
        tf.summary.scalar(scope+'_mean{}'.format(i + 1), mean)
        tf.summary.histogram(scope+'_gradients{}'.format(i + 1), grad)
        tf.summary.histogram(scope+'_weights{}'.format(i + 1), weight)

def reject_outliers(data, m=5):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    scale = d / (mdev if mdev else 1.)
    data[scale > m] = 0.
    return data

def reject_outliers2(data):
    for i in range(len(data)-1):
        if (data[i]>0 and data[i+1]>0) or (data[i]<0 and data[i+1]<0):
            if abs(data[i] - data[i+1]) > 2.5:
                data[i+1] = data[i] + np.random.normal(0., 0.2) 
        else:
            if abs(data[i] - data[i+1]) > 7:
                data[i+1] = data[i] + np.random.normal(0., 0.2) 
            
    return data
        
def tboard_concat(samples, side):
    tboard_sample = np.concatenate(samples[:side], axis=1)
    for s in range(1,side):
        _n = s * side
        _sample = samples[_n:_n+side]
        tboard_sample = np.concatenate((tboard_sample, np.concatenate(_sample, axis=1)), axis=0)
    return tboard_sample

def resampling_1d(x, y, bounds=(3750,7000), size=3500):
    """Perform 1d interpolation
    Arguments:
    -> x, y: x and y data dimensions. They should include one axis only."""
    f = interpolate.interp1d(x=x, y=y, kind='linear', assume_sorted=True)
    xnew = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), size)
    ynew = f(xnew)
    return xnew, ynew

def is_invalid(arr):
    def _check_mostly_zeros(arr):
        """Returns True if more than 95% of the elements of arr are zero"""
        count = 0
        for elem in arr:
            if elem == 0.:
                count += 1
        if count>0.95*len(arr):
            print("ERROR: _check_mostly_zeros")
            return True
        return False
    def _check_infinite(arr):
        """Returns True if there is at least one element in arr is either infinite or nan"""
        return not np.all(np.isfinite(arr))
    return _check_mostly_zeros(arr) or _check_infinite(arr)

class PlotGenSamples():
    def __init__(self, ncols=6, nrows=6, figsize=(16,12)):
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize

    def plot_spectra(self, samples, lambdas, name='', fix_size=True):
        self.fig, self.ax = plt.subplots(nrows=self.nrows, ncols=self.ncols, 
                                         squeeze=False, sharex=True, 
                                         figsize=self.figsize)
        self.fig.subplots_adjust(hspace=0.07)
        i = 0
        for irow in range(self.nrows):
            for icol in range(self.ncols):
                self.ax[irow, icol].grid()
                self.ax[irow, icol].set_ylabel('Flux')
                if fix_size:
                    self.ax[irow, icol].plot(lambdas[i].reshape(3500), samples[i].reshape(3500))
                else:
                    self.ax[irow, icol].plot(lambdas[i], samples[i])
                i = i + 1
        plt.xlabel('Wavelength [A]')
        plt.savefig('/fred/oz012/Bruno/figs/{}.png'.format(name))
        plt.close()

    def plot_mnist(self, samples, name):
        self.fig = plt.figure(figsize=(6,6))
        self.gs = gridspec.GridSpec(self.nrows,self.ncols)
        self.gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(self.gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(28,28), cmap='Greys_r')
        plt.savefig('/fred/oz012/Bruno/figs/{}.png'.format(name))
        plt.close()

    def plot_cifar10(self, samples, name):
        self.fig = plt.figure(figsize=(6,6))   
        self.gs = gridspec.GridSpec(self.nrows,self.ncols)
        self.gs.update(wspace=0.05, hspace=0.05)
        for i, sample in enumerate(samples):
            ax = plt.subplot(self.gs[i])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(sample.reshape(32,32,3))
        plt.savefig('/fred/oz012/Bruno/figs/{}.png'.format(name))
        plt.close()
