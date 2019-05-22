import os
import tensorflow as tf
from src.models.gans import DCGAN
import argparse
from src import argparser

def main(argv=None):
    nepochs = 1000
    checkpoint_dir = '/fred/oz012/Bruno/checkpoints/' + str(FLAGS.checkpoint)
    with tf.Session() as sess:
        dcgan = DCGAN(sess=sess,
                      in_height=32,
                      in_width=32,
                      nchannels=3,
                      batch_size=128,
                      noise_dim=100,
                      mode='wgan-gp',
                      opt_pars=(0.0001, 0., 0.9),
                      d_iters=5,
                      data_name='cifar10',
                      checkpoint_dir=checkpoint_dir)

        if FLAGS.mode == 'train':
            dcgan.train(nepochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
