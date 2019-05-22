import os
import tensorflow as tf
from src.models.gans import DCGAN
import argparse
from src import argparser

def main(argv=None):
    nepochs = 5
    checkpoint_dir = '/fred/oz012/Bruno/checkpoints/' + str(FLAGS.checkpoint)
    with tf.Session() as sess:
        dcgan = DCGAN(sess=sess,
                      in_width=3500,
                      in_height=1,
                      nchannels=1,
                      batch_size=512,
                      noise_dim=100,
                      checkpoint_dir=checkpoint_dir,
                      data_name='spectra',
                      files_path='/fred/oz012/Bruno/data/spectra/boss/loz/tmp/',
                      dataset_size = 350000)

        if FLAGS.mode == 'train':
            dcgan.train(nepochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
