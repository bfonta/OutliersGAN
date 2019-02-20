import os
import tensorflow as tf
from src.models.gans import DCGAN
import argparse
from src import argparser

def main(argv=None):
    nepochs = 2
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
                      checkpoint_dir=checkpoint_dir,
                      tensorboard_dir=tensorboard_dir)
            
        if FLAGS.mode == 'train':
            dcgan.train(nepochs, drop_d=0., drop_g=0.)

        elif FLAGS.mode == 'generate':        
            dcgan.generate(n=512, name='generate')

"""
def generate(n, name, dire):
    import numpy as np
    from src.utilities import PlotGenSamples

    latest_checkpoint = tf.train.latest_checkpoint(dire)
    saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')

    G_output = tf.get_default_graph().get_tensor_by_name('G/output:0')
    gen_data_ph = tf.get_default_graph().get_tensor_by_name('G/gen_data_ph:0')
    dropout_prob_ph = tf.get_default_graph().get_tensor_by_name('dropout_prob_ph:0')
    batch_size_ph = tf.get_default_graph().get_tensor_by_name('batch_size_ph:0')

    print(G_output)
    print(gen_data_ph)
    print(dropout_prob_ph)
    print(batch_size_ph)

    #iterator = self.dataset.make_initializable_iterator()
    #next_element = iterator.get_next()
    #init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
    #self.sess.run(iterator.initializer)
    #_, *params = self.sess.run(next_element)

    with tf.Session() as sess:
        saver.restore(sess, latest_checkpoint)
        gen_samples = sess.run(G_output, 
                               feed_dict={gen_data_ph:np.random.normal(loc=0.0,scale=1.,size=[n,100]),
                                          dropout_prob_ph: 0., 
                                          batch_size_ph: n})

        p = PlotGenSamples()
        p.plot_mnist(gen_samples[:36], name)


"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
