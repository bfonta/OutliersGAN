import os
import tensorflow as tf
from src.models.gans import DCGAN
import argparse
from src import argparser

def main(argv=None):
    nepochs = 1000
    checkpoint_dir = '/fred/oz012/Bruno/checkpoints/' + str(FLAGS.checkpoint)
    tensorboard_dir = '/fred/oz012/Bruno/tensorboard/' + str(FLAGS.checkpoint)

    with tf.Session() as sess:
        dcgan = DCGAN(sess=sess,
                      in_height=28,
                      in_width=28,
                      nchannels=1,
                      batch_size=128,
                      noise_dim=100,
                      mode='original',
                      opt_pars=(0.0001, 0.5, 0.999),
                      d_iters=1,
                      data_name='mnist',
                      pics_save_names=('mnist_data_','mnist_gen_'),
                      checkpoint_dir=checkpoint_dir,
                      tensorboard_dir=tensorboard_dir)

        if FLAGS.mode == 'train':
            dcgan.train(nepochs)
        elif FLAGS.mode == 'generate':
            dcgan.generate()
        elif FLAGS.mode == 'predict':
            dcgan.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
