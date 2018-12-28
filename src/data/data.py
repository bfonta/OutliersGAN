import os
import glob
import numpy as np
import tensorflow as tf
from astropy.io import fits
import matplotlib.pyplot as plt
from ..data.spectra_generation import gen_spectrum_2lines

def _bytes_feature(Value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=Value))

def _int_feature(Value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[Value]))

def _float_feature(Value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=Value))

def _remove_nans(Arr):
    nan_idxs = np.where(np.isnan(Arr))[0]
    Arr[nan_idxs]=0

def _remove_negs(Arr):
    nan_idxs = np.where(Arr<0)[0]
    Arr[nan_idxs]=0

def _remove_small(Arr):
    nan_idxs = np.where(np.logical_and(Arr>0,Arr<1e-8))[0]
    Arr[nan_idxs]=0

def real_spectra_to_tfrecord(Filename, DataFolder, ShardsNumber=1):
    """
    Saves spectra in the TFRecord format. 
    The last shard is equal or smaller than all others.
        Arguments:
        -> DataFolder (string): folder where the spectra ('.fits' files) are stored
        -> Filename (string): name of the file where the data is going to be stored
        -> ShardsNumber (int): number of shards for splitting the data
        """
    if not os.path.isdir(DataFolder):
        raise ValueError('The specified', DataFolder, 'folder does not exist.')
    if type(DataFolder) != str or type(Filename) != str or type(ShardsNumber) != int:
        raise TypeError('')
    spectra = glob.glob(os.path.join(DataFolder,"*.fits"))    
    spectra_number = len(spectra)
    files, ext = Filename.split('.')

    shard_width = int(spectra_number/ShardsNumber)+1
    for i_shard in range(ShardsNumber):
        with tf.python_io.TFRecordWriter(files+str(i_shard)+'.'+ext) as _w:
            for i_spectrum in range(i_shard*shard_width,(i_shard+1)*shard_width): 
                if i_spectrum%5000==0:
                    print('{} iteration'.format(i_spectrum))
                if i_spectrum>=spectra_number:
                    break
                with fits.open(spectra[i_spectrum]) as hdu:
                    f = hdu[1].data['flux']
                    f = f[0:3750]
                    f = f[250:]
                    #_remove_nans(f)
                    #_remove_negs(f)
                    #_remove_small(f)
                    f = f.astype(np.float)
                    print(type(f))
                Example = tf.train.Example(features=tf.train.Features(feature={
                    #'spectrum_raw': _bytes_feature(f.tostring())
                    'spectrum_raw': _float_feature(f.tolist())
                }))
                _w.write(Example.SerializeToString())

def gen_spectra_to_tfrecord(nspectra, spectra_size, filename, nshards=1, norm=False):
    """
    Saves spectra in the TFRecord format. 
    The last shard is equal or smaller than all others.
        Arguments:
        -> nspectra (int): number of spectra to save on disk
        -> nshards (int): number of shards for splitting the data
        """
    nspectra_per_shard_float = float(nspectra)/float(nshards)
    if nspectra_per_shard_float%1==0:
        nspectra_per_shard = int(nspectra_per_shard_float)
    else:
        nspectra_per_shard = int(nspectra_per_shard_float) + 1

    for i_shard in range(nshards):
        with tf.python_io.TFRecordWriter(filename+str(i_shard)+'.tfrecord') as _w:
            if i_shard == nshards-1: #last shard
                nspectra_per_shard = nspectra - i_shard*nspectra_per_shard
            _, data, labels = gen_spectrum_2lines(nspectra_per_shard, spectra_size, norm=norm)
            print('Shard {} with {} spectra'.format(i_shard, nspectra_per_shard))
            for j in range(nspectra_per_shard):
                Example = tf.train.Example(features=tf.train.Features(feature={
                    'spectrum': _float_feature(data[j].tolist()),
                    'label': _int_feature(labels[j])
            }))
                _w.write(Example.SerializeToString())

def read_spectra_data(filename, spectrum_length, data_folder=''):
    """
    Converts TFRecord files into a shuffled TensorFlow Dataset. 
    Arguments: 
    -> filename(string): The main name of the files to load (excluding the shard number).
    -> spectrum_length (int): The length of the spectra stored in the tfrecord files.
    -> data_folder (string): The folder where the tfrecord files are stored.
    Returns: A mapped tf.data.Dataset (pictures, labels)
    """
    if type(filename) != str or type(data_folder) != str:
        raise TypeError('The name of the file and the name of the folder must be strings.')
    filename, ext = filename.split('.')
    if ext!='tfrecord':
        raise ValueError('Only TFRecord files can be read.')
    files = []
    i_file = 0
    while True:
        f = filename+str(i_file)+'.'+ext
        if os.path.isfile(os.path.join(data_folder,f)):
            files.append(f)
        else:
            break
        i_file += 1
    nshards = i_file+1

    def parser_func(tfrecord):
        features = {'spectrum': tf.FixedLenFeature(shape=[spectrum_length], dtype=tf.float32),
                    'label': tf.FixedLenFeature((), tf.int64)}
        parsed_features = tf.parse_single_example(tfrecord, features)
        return {'data': parsed_features['spectrum']}, parsed_features['label']

    dataset = tf.data.Dataset.list_files(files).shuffle(nshards) #dataset of filenames
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=nshards)
    dataset = dataset.map(map_func=parser_func, 
                          num_parallel_calls=32) #number of available CPUs per node in OzStar
    return dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
