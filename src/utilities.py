from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf

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


class PlotGenSamples():
    def __init__(self, ncols=6, nrows=6):
        self.nrows = nrows
        self.ncols = ncols

    def plot_spectra(self, samples, name=''):
        self.fig, self.ax = plt.subplots(nrows=3, ncols=3, figsize=(8,6))
        i = 0
        for irow in range(self.nrows):
            for icol in range(self.ncols):
                #self.ax[irow, icol] = self.fig.add_subplot(self.ncols*self.nrows,irow,icol)
                self.ax[irow, icol].plot(np.arange(3500), samples[i].reshape(3500))
                #ax.set_xticklabels([])
                #ax.set_yticklabels([])
                #ax.set_aspect('equal')
                i = i + 1
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


def log_tf_files(num_layers, loss, player='D'):
    gr = tf.get_default_graph()
    for i in range(num_layers):
        weight = gr.get_tensor_by_name(player+'/layer{}/kernel:0'.format(i + 1))
        grad = tf.gradients(loss, weight)[0]
        mean = tf.reduce_mean(tf.abs(grad))
        tf.summary.scalar(player+'_mean{}'.format(i + 1), mean)
        tf.summary.histogram(player+'_gradients{}'.format(i + 1), grad)
        tf.summary.histogram(player+'_weights{}'.format(i + 1), weight)
