import h5py
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse

def write():
    with h5py.File(FLAGS.read_path, 'r') as f:
        group = f['data']
        dset1 = group['spectra']
        dset2 = group['spectra_additional']
        data1 = dset1[:,:]
        data2 = dset2[:,:]

    #standardisation
    data1 = (data1 - data1.mean(axis=0, keepdims=True)) / data1.std(axis=0, keepdims=True)
    data2 = (data2 - data2.mean(axis=0, keepdims=True)) / data2.std(axis=0, keepdims=True)
    print("Shapes:")
    print(data1.shape)
    print(data2.shape)

    #model fitting
    model = PCA(n_components=3)
    data1 = model.fit_transform(data1)
    X1 = np.zeros((1,1))#TSNE(n_components=3).fit_transform(data1)
    #print(model.explained_variance_ratio_[:10])
    #print(model.explained_variance_ratio_.cumsum()[:10])

    data2 = model.fit_transform(data2)
    #print(model.explained_variance_ratio_[:10])
    #print(model.explained_variance_ratio_.cumsum()[:10])
    X2 = np.zeros((1,1))#TSNE(n_components=3).fit_transform(data2)

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
    return parser.parse_known_args()

def plot(d1, d2):
    plt.figure(figsize=(15,7))
    plt.scatter(d1[:,0],d1[:,1],s=d1[:,2]*0.0002,color='olive',label='trained',alpha=.5)
    plt.scatter(d2[:,0],d2[:,1],s=d2[:,2]*0.01,color='orangered',label='not trained',alpha=0.2)
    plt.legend()
    plt.ylim([-20,20])
    plt.xlim([-20,20])
    plt.savefig('dim_reduction2.png')
    plt.show()

FLAGS, _ = parser(argparse.ArgumentParser())

if FLAGS.mode == 'write':
    p1, p2, t2, t1 = write()
    plot(p1, p2)
elif FLAGS.mode == 'read':
    p1, p2, t1, t2 = read()
    """
    p1 = PCA(n_components=4).fit_transform(p1)
    p2 = PCA(n_components=4).fit_transform(p2)
    p1 = TSNE(n_components=3).fit_transform(p1)
    p2 = TSNE(n_components=3).fit_transform(p2)
    """
    plot(p1, p2)    
else:
    raise ValueError('The parsed argument does not exist.')
