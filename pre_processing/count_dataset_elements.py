import tensorflow as tf
from src.data.tfrecords import read_spectra_data as read_data

batch_size = 1
files_path = '/fred/oz012/Bruno/data/spectra/qso_zWarning/'
dataset = read_data(files_path+'spectra.tfrecord', 3500) #reads all shards: spectra2_0.tfrecord, ...
dataset = dataset.repeat(1).batch(batch_size)

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

counter = 0
with tf.Session() as sess:
    sess.run(iterator.initializer)
    while True:
        try:
            inputs, *params = sess.run(next_element)
            counter += 1
            if counter%10000==0:
                print(counter)
        except tf.errors.OutOfRangeError:
            print('Number of elements inside the dataset: ', counter)
            break

