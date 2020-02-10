import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from astropy.io import fits
import matplotlib.pyplot as plt
from ..data.spectra_generation import gen_spectrum_2lines
from ..data.fits import to_fits
from ..utilities import resampling_1d, is_invalid

def _bytes_feature(Value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[Value]))

def _int_feature(Value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[Value]))

def _float_feature(Value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=Value))

def real_spectra_to_tfrecord(filename, table_path, nshards=1, write_fits=False):
    """
    Saves spectra in the TFRecord format. 
    The last shard is equal or smaller than all others.
        Arguments:
        -> filename (string): name of the file where the data is going to be stored
        -> table_path (string): csv file
        -> nshards (int): number of shards for splitting the data
    """
    df = pd.read_csv(table_path)
    nspectra = len(df.axes[0])
    #bounds = 3750, 7000 #used for all spectra except qso
    bounds = 1800, 4150
    tot_length = 3500
    shard_width = int(nspectra/nshards)+1
    err_counter = 0
    for i_shard in range(nshards):
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

                    if is_invalid(flux_):
                        err_counter += 1
                        continue

                    if lam_[0] < bounds[0] and lam_[-1] > bounds[1]:
                        lam_, flux_ = resampling_1d(x=lam_, y=flux_, bounds=bounds, size=tot_length)
                        if write_fits:
                            to_fits(x=lam_, y=flux_, name='data_'+str(i_spectrum))
                            quit()
                    else:
                        print("ERROR: {}".format(i_spectrum))
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
            wavelength, data, labels = gen_spectrum_2lines(nspectra_per_shard, spectra_size, norm=norm)
            print('Shard {} with {} spectra'.format(i_shard, nspectra_per_shard))
            for j in range(nspectra_per_shard):
                #wavelength and redhsift are provided only for compatibility with real spectra
                Example = tf.train.Example(features=tf.train.Features(feature={
                    'spectrum': _float_feature(data[j].tolist()),
                    'wavelength': _float_feature(wavelength[j].tolist()),
                    'redshift': _float_feature([0.])
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
    dataset = dataset.map(map_func=parser_func, num_parallel_calls=32) #number of available CPUs per node in OzStar
    return dataset.shuffle(buffer_size=7000, reshuffle_each_iteration=True)


def pictures_to_tfrecord(filename, data_folder, nshards=1):
    """
    Saves spectra in the TFRecord format. 
    The last shard is equal or smaller than all others.
        Arguments:
        -> filename (string): name of the file where the data is going to be stored
        -> data_folder (string): folder where the pictures ('.jpg' files) are stored
        -> nshards (int): number of shards for splitting the data
        """    
    from PIL import Image

    fnames = glob.glob(os.path.join(data_folder, '*.jpg'))
    npics = len(fnames)
    shard_width = int(npics/nshards)+1

    for i_shard in range(nshards):
        with tf.python_io.TFRecordWriter(filename+str(i_shard)+'.tfrecord') as _w:
            for ipic in range(i_shard*shard_width,(i_shard+1)*shard_width): 
                if ipic%2000==0:
                    print('{} iteration'.format(ipic), flush=True)
                if ipic>=npics:
                    break
                with Image.open(fnames[ipic]) as im:
                    im = im.resize((109,89), Image.ANTIALIAS)
                    im = np.array(im).astype(np.float32)
                    im = np.transpose(im, (2,0,1))
                    Example = tf.train.Example(features=tf.train.Features(feature={
                        #'pic': _float_feature(im.flatten().tolist()),
                        'pic': _bytes_feature(im.tostring()),
                    }))
                    _w.write(Example.SerializeToString())
            quit()


def read_pictures_data(filename, pic_size=(), data_folder=''):
    """
    Converts TFRecord files into a shuffled TensorFlow Dataset. 
    Arguments: 
    -> filename(string): The main name of the files to load (excluding the shard number).
    -> pic_size (int): The length of the spectra stored in the tfrecord files.
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
            files.append(os.path.join(data_folder,f))
        else:
            break
        i_file += 1
    nshards = i_file+1

    def parser_func(tfrecord):
        #feats = {'pic': tf.FixedLenFeature((pic_size[0]*pic_size[1]*pic_size[2]), tf.float32)}
        feats = {'pic': tf.FixedLenFeature((), tf.string)}
        pfeats = tf.parse_single_example(tfrecord, feats)
        pic =  tf.decode_raw(pfeats['pic'], tf.float32)
        return tf.reshape(pic, (pic_size[0],pic_size[1],pic_size[2]))

    dataset = tf.data.Dataset.list_files(files).shuffle(nshards) #dataset of filenames
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=nshards)
    dataset = dataset.map(map_func=parser_func, num_parallel_calls=32) #number of available CPUs per node in OzStar
    
    return dataset.shuffle(buffer_size=7000, reshuffle_each_iteration=True)
