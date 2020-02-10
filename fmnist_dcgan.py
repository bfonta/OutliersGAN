import argparse
from src import argparser
import tensorflow as tf
from src.models.gans import DCGAN

def main(argv=None):
    nepochs = 500
    checkpoint_dir = '/fred/oz012/Bruno/checkpoints/' + str(FLAGS.checkpoint) + '/'
    tensorboard_dir = '/fred/oz012/Bruno/tensorboard/' + str(FLAGS.checkpoint) + '/'

    with tf.Session() as sess:
        dcgan = DCGAN(sess=sess,
                      in_height=28,
                      in_width=28,
                      nchannels=1,
                      batch_size=512,
                      noise_dim=100,
                      mode='original',
                      opt_pars=(0.0001, 0.5, 0.999),
                      d_iters=1,
                      data_name='fashion_mnist',
                      pics_save_names=('fashion_mnist_data_v2_','fashion_mnist_gen_v2_'),
                      checkpoint_dir=checkpoint_dir,
                      tensorboard_dir=tensorboard_dir)
            
        if FLAGS.mode == 'train':
            dcgan.train(nepochs, drop_d=0., drop_g=0.)

        elif FLAGS.mode == 'generate':        
            dcgan.generate(N=1, n_per_plot=5, name='generate')

        elif FLAGS.mode == 'predict':        
            dcgan.predict(n_pred=1)

        elif FLAGS.mode == 'save_features':        
            dcgan.save_features(ninputs=50000, save_path='tensorflow', additional_data_name='mnist')

        else:
            raise ValueError('The specified mode is not supported.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
