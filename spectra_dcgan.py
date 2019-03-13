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
                      mode='wgan-gp',
                      opt_pars=(0.00005, 0.0, 0.9),
                      d_iters=5,
                      data_name='spectra',
                      #dataset_size=345960,
                      dataset_size=315082, #Invalid: 391672
                      pics_save_names=('spec_data_grid_','spec_gen_grid_'),
                      #files_path='/fred/oz012/Bruno/data/spectra/boss/loz/',
                      files_path='/fred/oz012/Bruno/data/spectra/legacy/legacy_bit6/',
                      files_name='spectra_grid',
                      checkpoint_dir=checkpoint_dir,
                      tensorboard_dir=tensorboard_dir)

        if FLAGS.mode == 'train':
            dcgan.train(nepochs, drop_d=0.0, drop_g=0.0, flip_prob=0.15, restore=False)

        elif FLAGS.mode == 'generate':
            dcgan.generate(N=1, n=5, name='generate', write_fits=True)

        elif FLAGS.mode == 'predict':
            dcgan.predict(n_pred=514)
        
        elif FLAGS.mode == 'save_features':
            inp = 315082
            dcgan.save_features(ninputs=inp, 
                                save_path='/fred/oz012/Bruno/data/hdf5/tf_legacy_loz_'+str(inp)+'.hdf5', 
                        additional_data_name='spectra',
                        additional_data_path='/fred/oz012/Bruno/data/spectra/boss/loz/',
                        #additional_data_path='/fred/oz012/Bruno/data/spectra/legacy/outliers/',
                        additional_ninputs=inp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
