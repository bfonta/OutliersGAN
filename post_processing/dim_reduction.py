import h5py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse

def write():
    with h5py.File('2.hdf5', 'r') as f:
        group = f['data']
        dset1 = group['fashion_mnist']
        dset2 = group['mnist']
        data1 = dset1[:,:]
        data2 = dset2[:,:]

    data1 = data1[:,:100]
    data2 = data2[:,:100]
    assert data1.shape == data2.shape

    data1 = PCA(n_components=50).fit_transform(data1)
    X1 = TSNE(n_components=2).fit_transform(data1)
    data2 = PCA(n_components=50).fit_transform(data2)
    X2 = TSNE(n_components=2).fit_transform(data2)

    with h5py.File('reduced_data.hdf5', 'w') as f:
        g1 = f.create_group('PCA')
        d1 = g1.create_dataset('reduced_data', (data1.shape[0], data1.shape[1], 2), dtype=np.float32) 
        g2 = f.create_group('tSNE')
        d2 = g2.create_dataset('reduced_data', (X1.shape[0], X1.shape[1], 2), dtype=np.float32) 

        d1[:,:,0] = data1
        d1[:,:,1] = data2
        d2[:,:,0] = X1
        d2[:,:,1] = X2

    return data1, data2, X1, X2

def read():
    with h5py.File('reduced_data.hdf5', 'r') as f:
        g1 = f['PCA']
        g2 = f['tSNE']
        d1 = g1['reduced_data']
        d2 = g2['reduced_data']
        
        pca1 = d1[:,:,0]
        pca2 = d1[:,:,1]
        tsne1 = d2[:,:,0]
        tsne2 = d2[:,:,1]

    return pca1, pca2, tsne1, tsne2

def parser(parser):
    parser.add_argument('--checkpoint',
                        type=int,
                        default=0,
                        help='Checkpoint prefix number.')
    parser.add_argument('--mode',
                        type=str,
                        default='train',
                        help='Mode to run the model.')
    return parser.parse_known_args()

def plot(tsne1, tsne2):
    plt.scatter(tsne1[:,0], tsne1[:,1], color='orange', label='trained', alpha=0.5)
    plt.scatter(tsne2[:,0], tsne2[:,1], color='green', label='not trained', alpha=0.5)
    plt.legend()
    plt.savefig('dim_reduction.png')
    plt.show()

FLAGS, _ = parser(argparse.ArgumentParser())

if FLAGS.mode == 'write':
    _, _, t1, t2 = write()
    plot(t1, t2)
elif FLAGS.mode == 'read':
    _, _, t1, t2 = read()
    plot(t1, t2)    
else:
    raise ValueError('The parsed argument does not exist.')
