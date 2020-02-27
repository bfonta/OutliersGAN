import h5py
import numpy as np
np.set_printoptions(threshold=np.nan)
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse

"""
Instructions

This macro performs dimensionality reduction over the output os trained layers of a tensorflow model.

Steps:
1) Run a Tensorflow model (for the case of my project it was a GAN) with '--mode save_features'. See the macro 'spectra_dcgan.py' for a good example. The output of the layers will be stored according to the provided path.
2) Use this macro with '--mode write' using '--read_path' followed by the name of the file just saved and using '--write_path' to specify the name of the output (dimensionality reduced) file.
3) Reuse the former file to plot in different way. Use '--mode read' for this, specifying '--read_path'.

Example:
1) tf.hdf5 -> file with some layers outputs
2) python dim_reduction.py --mode write --read_path tf.hdf5 --write_path red.hdf5
3) python dim_reduction.py --mode read --read_path red.hdf5

Depending on the method used (PCA, tSNE, ...) the third step may be much faster than the second one. A plot with the reduced data is shown after steps 2) and 3). Change the size of the data points in the plot() function if needed.
"""

def write():
    with h5py.File(FLAGS.read_path, 'r') as f:
        group = f['data']
        dset1 = group['spectra']
        dset2 = group['spectra_additional']
        data1 = dset1[:,:]
        data2 = dset2[:,:]

    #standardisation
    print("Number of zeros: {}.".format(data1.size-np.count_nonzero(data1)))
    print("Number of zeros: {}.".format(data2.size-np.count_nonzero(data2)))
    data1 = (data1 - data1.mean(axis=0, keepdims=True)) / data1.std(axis=0, keepdims=True)
    data2 = (data2 - data2.mean(axis=0, keepdims=True)) / data2.std(axis=0, keepdims=True)
    print("Shapes:")
    print(data1.shape)
    print(data2.shape)

    #model fitting
    model_pca = PCA(n_components=3)
    model_tsne = TSNE(n_components=3)
    data1 = model_pca.fit_transform(data1)
    if(FLAGS.with_tsne):
        X1 = model_tsne.fit_transform(data1)
    else:
        X1 = np.zeros((1,1))
    #print(model.explained_variance_ratio_[:10])
    #print(model.explained_variance_ratio_.cumsum()[:10])

    data2 = model_pca.fit_transform(data2)
    #print(model.explained_variance_ratio_[:10])
    #print(model.explained_variance_ratio_.cumsum()[:10])
    if(FLAGS.with_tsne):
        X2 = model_tsne.fit_transform(data2)
    else:
        X2 = np.zeros((1,1))

    with h5py.File(FLAGS.write_path, 'w') as f:
        g1 = f.create_group('PCA')
        d11 = g1.create_dataset('reduced_data1', (data1.shape[0], data1.shape[1]), dtype=np.float32) 
        d12 = g1.create_dataset('reduced_data2', (data2.shape[0], data2.shape[1]), dtype=np.float32) 
        g2 = f.create_group('tSNE')
        d21 = g2.create_dataset('reduced_data1', (X1.shape[0], X1.shape[1]), dtype=np.float32) 
        d22 = g2.create_dataset('reduced_data2', (X2.shape[0], X2.shape[1]), dtype=np.float32) 

        d11[:,:] = data1
        d12[:,:] = data2
        d21[:,:] = X1
        d22[:,:] = X2

    return data1, data2, X1, X2

def read():
    with h5py.File(FLAGS.read_path, 'r') as f:
        g1 = f['PCA']
        g2 = f['tSNE']
        d11 = g1['reduced_data1']
        d12 = g1['reduced_data2']
        d21 = g2['reduced_data1']
        d22 = g2['reduced_data2']
        
        pca1 = d11[:,:]
        pca2 = d12[:,:]
        tsne1 = d21[:,:]
        tsne2 = d22[:,:]

    return pca1, pca2, tsne1, tsne2

def parser(parser):
    parser.add_argument('--mode',
                        type=str,
                        default='read',
                        help='Read already existing reduced data hdf5 file or write a new one.')
    parser.add_argument('--read_path',
                        type=str,
                        help='Path of saved hdf5 file.')
    parser.add_argument('--write_path',
                        type=str,
                        default='file.hdf5',
                        help='Path of saved hdf5 file.')
    parser.add_argument('--with_tsne',
                        type=int,
                        default=0,
                        help='Runs very slow tSNE.'
    )
    return parser.parse_known_args()

def plot(name, d1, d2, xlim, ylim):
    rel_size = .005
    plt.figure(figsize=(13,6))
    plt.scatter(d1[:,0],d1[:,1],s=d1[:,2]*rel_size*10,color='olive',label='trained galaxy data',alpha=.5)
    plt.scatter(d2[:,0],d2[:,1],s=d2[:,2]*rel_size,color='orangered',label='non-trained qso data',alpha=0.5)
    plt.legend()
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.savefig(name)
    plt.close()

FLAGS, _ = parser(argparse.ArgumentParser())

if FLAGS.mode == 'write':
    p1, p2, t2, t1 = write()
elif FLAGS.mode == 'read':
    p1, p2, t1, t2 = read()
    """
    p1 = PCA(n_components=4).fit_transform(p1)
    p2 = PCA(n_components=4).fit_transform(p2)
    p1 = TSNE(n_components=3).fit_transform(p1)
    p2 = TSNE(n_components=3).fit_transform(p2)
    """
else:
    raise ValueError('The parsed argument does not exist.')

plot('pca.png', p1, p2, [-100,100], [-100,100])
if(FLAGS.with_tsne):
    plot('tsne.png', t1, t2, [-100,20], [-25,50])    
