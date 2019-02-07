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

def _remove_nans(arr):
    nan_idxs = np.where(np.isnan(arr))[0]
    arr[nan_idxs]=0

def _remove_negs(arr):
    nan_idxs = np.where(arr<0)[0]
    arr[nan_idxs]=0

def _remove_small(arr):
    nan_idxs = np.where(np.logical_and(arr>0,arr<1e-8))[0]
    Arr[nan_idxs]=0


def _is_invalid(arr, l):
    def _check_only_zeros(arr):
        """
        Returns True if all the elements of arr are zero
        """
        return not np.any(arr)

    def _check_infinite(arr):
        """
        Returns True if there is at least one element in arr is either infinite or nan
        """
        return not np.all(np.isfinite(arr))
        
    def _check_wrong_length(arr, l):
        if len(arr)==l:
            return False
        else:
            return True

    return _check_only_zeros(arr) or _check_infinite(arr) or _check_wrong_length(arr, l)


def real_spectra_to_tfrecord(filename, table_path, shards_number=1):
    """
    Saves spectra in the TFRecord format. 
    The last shard is equal or smaller than all others.
        Arguments:
        -> data_folder (string): folder where the spectra ('.fits' files) are stored
        -> filename (string): name of the file where the data is going to be stored
        -> shards_number (int): number of shards for splitting the data
        """
    import pandas as pd
    df = pd.read_csv(table_path)
    nspectra = len(df.axes[0])

    shard_width = int(nspectra/shards_number)+1
    err_counter = 0
    for i_shard in range(shards_number):
        with tf.python_io.TFRecordWriter(filename+str(i_shard)+'.tfrecord') as _w:
            for i_spectrum in range(i_shard*shard_width,(i_shard+1)*shard_width): 
                if i_spectrum%2000==0:
                    print('{} iteration'.format(i_spectrum), flush=True)
                if i_spectrum>=nspectra:
                    break
                with fits.open(df['local_path'].iloc[i_spectrum]) as hdu:
                    flux_ = hdu[1].data['flux'].astype(np.float32)
                    lam_ = np.power( 10, hdu[1].data['loglam'] ).astype(np.float32)
                    rshift_ = df['redshift'].iloc[i_spectrum].astype(np.float32)
                    lam_ = lam_ / (1 + rshift_)

                    #reasonable selection
                    #obtained after checking the minima and maxima of all spectra
                    #and looking at individual spectra
                    #such that the most important features are present
                    #in a range of 3500 data points
                    selection = (lam_>3400) & (lam_<6900)
                    lam_= lam_[selection]
                    flux_= flux_[selection]

                    if _is_invalid(flux_, 3500):
                        err_counter += 1
                        continue

                    Example = tf.train.Example(features=tf.train.Features(feature={
                        'spectrum': _float_feature(flux_.tolist()),
                        'wavelength': _float_feature(lam_.tolist()),
                        'redshift': _float_feature([rshift_])
                    }))
                _w.write(Example.SerializeToString())
    print("Number of invalid spectra:", err_counter, flush=True)
    print("Number of valid spectra:", nspectra-err_counter, flush=True)


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
        feats = {'spectrum': tf.FixedLenFeature((spectrum_length), tf.float32),
                 'wavelength': tf.FixedLenFeature((spectrum_length), tf.float32),
                 'redshift': tf.FixedLenFeature((), tf.float32)}
        pfeats = tf.parse_single_example(tfrecord, feats)
        return pfeats['spectrum'], pfeats['wavelength'], pfeats['redshift']


    dataset = tf.data.Dataset.list_files(files).shuffle(nshards) #dataset of filenames
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=nshards)
    dataset = dataset.map(map_func=parser_func, 
                       num_parallel_calls=32) #number of available CPUs per node in OzStar
    return dataset.shuffle(buffer_size=100000, reshuffle_each_iteration=True)
