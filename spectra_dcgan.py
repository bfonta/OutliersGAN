import os
import tensorflow as tf
from src.models.gans import DCGAN
import argparse
from src import argparser

def main(argv=None):
    nepochs = 1000
    checkpoint_dir = '/fred/oz012/Bruno/checkpoints/' + str(FLAGS.checkpoint) + '/'
    tensorboard_dir = '/fred/oz012/Bruno/tensorboard/' + str(FLAGS.checkpoint) + '/'

    with tf.Session() as sess:
        dcgan = DCGAN(sess=sess,
                      in_height=3500,
                      in_width=1,
                      nchannels=1,
                      batch_size=512,
                      noise_dim=100,
                      mode='original',
                      opt_pars=(0.00002, 0.9, 0.999),
                      d_iters=1,
                      data_name='spectra',
                      dataset_size=345960,
                      pics_save_names=('spectra_data_v6_', 'spectra_gen_v6_'),
                      files_path='/fred/oz012/Bruno/data/spectra/boss/loz/',
                      checkpoint_dir=checkpoint_dir,
                      tensorboard_dir=tensorboard_dir)

        if FLAGS.mode == 'train':
            dcgan.train(nepochs, drop_d=0.0, drop_g=0.0)
        elif FLAGS.mode == 'generate':
            dcgan.generate(N=3, n_per_plot=5, name='generate')
        elif FLAGS.mode == 'predict':
            dcgan.predict(n_pred=514)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
