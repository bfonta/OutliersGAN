import os
import tensorflow as tf
from src.models.gans import ResNetGAN
import argparse
from src import argparser

def main(argv=None):
    nepochs = 1000
    checkpoint_dir = '/fred/oz012/Bruno/checkpoints/' + str(FLAGS.checkpoint) + '/'    
    with tf.Session() as sess:
        resnet = ResNetGAN(sess=sess,
                           in_height=32,
                           in_width=32,
                           nchannels=3,
                           batch_size=128,
                           noise_dim=100,
                           checkpoint_dir=checkpoint_dir,
                           data_name='cifar10')

        if FLAGS.mode == 'train':
            resnet.train(nepochs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
