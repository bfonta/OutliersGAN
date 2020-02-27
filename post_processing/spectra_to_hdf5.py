import os
import numpy as np
import tensorflow as tf
import h5py

def read_tfrecord(filename, spectrum_length, batch_size, data_folder=''):
    filename, ext = filename.split('.')
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

    dataset = tf.data.Dataset.list_files(files).shuffle(nshards)
    dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=nshards)
    dataset = dataset.map(map_func=parser_func, num_parallel_calls=32) 
    dataset =  dataset.shuffle(buffer_size=7000, reshuffle_each_iteration=True)
    return dataset.repeat(1).batch(batch_size)

        
with h5py.File('/fred/oz012/Bruno/data/hdf5/plain_pca.hdf5', 'w') as f:
    path = '/fred/oz012/Bruno/data/spectra/'
    data_paths = (path + 'gal_starforming_starburst_zWarning/spectra.tfrecord', 
                  path + 'qso_zWarning/spectra.tfrecord')
    spectrum_length = 3500
    bs = 512
    ninputs = 22791
    
    datasets = (read_tfrecord(filename=data_paths[0], spectrum_length=spectrum_length, batch_size=bs),
                read_tfrecord(filename=data_paths[1], spectrum_length=spectrum_length, batch_size=bs))
    data_sizes = (315082, 345960)
    nbatches = (int(np.ceil(data_sizes[0]/bs)), 
                int(np.ceil(data_sizes[1]/bs)))
    iterators = (datasets[0].make_initializable_iterator(), 
                 datasets[1].make_initializable_iterator())
    next_elements = (iterators[0].get_next(),
                     iterators[1].get_next())

    group = f.create_group('data')
    dset = group.create_dataset('spectra', (ninputs, spectrum_length), dtype=np.float32) 

    save_start = 0
    count = 0
    with tf.Session() as sess:
        sess.run(iterators[0].initializer)
        for b in range(nbatches[0]):
            try:
                inputs, *params = sess.run(next_elements[0])

                #Makes sure that the number of input data is the one the user wants
                if (b+1)*bs > ninputs:
                    inputs = inputs[:ninputs - b*bs]

                dset[save_start:save_start+len(inputs), :] = inputs
                save_start += len(inputs)

                #Stopping criterion
                if len(inputs) != bs:
                    break

            except tf.errors.OutOfRangeError:
                raise

    save_start = 0
    count = 0

    dset = group.create_dataset('spectra_additional', (ninputs, spectrum_length), dtype=np.float32) 
    with tf.Session() as sess:
        sess.run(iterators[1].initializer)
        for b in range(nbatches[1]):
            try:
                inputs, *params = sess.run(next_elements[1])
                
                #Makes sure that the number of input data is the one the user wants
                if (b+1)*bs > ninputs:
                    inputs = inputs[:ninputs - b*bs]

                dset[save_start:save_start+len(inputs), :] = inputs
                save_start += len(inputs)

                #Stopping criterion
                if len(inputs) != bs:
                    break

            except tf.errors.OutOfRangeError:
                raise
