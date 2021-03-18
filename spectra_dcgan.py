import os
import tensorflow as tf
from src.models.gans import DCGAN
import argparse
from src import argparser

def main(argv=None):
    if FLAGS.noGPU:
        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
    else:
        sess = tf.Session()
    with sess:
        dcgan = DCGAN(
            sess=sess,
            in_height=3500,
            in_width=1,
            nchannels=1,
            batch_size=512,
            noise_dim=100,
            mode='wgan-gp',
            opt_pars=(0.00005, 0.0, 0.9),
            d_iters=5,
            data_name='spectra',
            dataset_size=dataset_size_d[FLAGS.checkpoint],  
            pics_save_names=pics_save_names_d[FLAGS.checkpoint],
            files_path=files_path_d[FLAGS.checkpoint],
            files_name=files_name_d[FLAGS.checkpoint],
            checkpoint_dir=checkpoint_dir,
            tensorboard_dir=tensorboard_dir,
        )

        if FLAGS.mode == 'train':
            dcgan.train(nepochs, drop_d=0.0, drop_g=0.0, flip_prob=0.15, restore=False)

        elif FLAGS.mode == 'generate':
            dcgan.generate(N=15, n=10, name=FLAGS.fname, write_fits=True)

        elif FLAGS.mode == 'predict':
            dcgan.predict(n_pred=514)

        elif FLAGS.mode == 'save_features':
            inp = 22791
            dcgan.save_features(
                ninputs=inp,
                save_path='/fred/oz012/Bruno/data/hdf5/tf_qso_gal_' + str(inp),
                additional_files_name='spectra',
                additional_data_path='/fred/oz012/Bruno/data/spectra/gal_starforming_starburst_zWarning/',
                additional_ninputs=inp,
            )

def set_param_dicts():
    d_ds, d_pn, d_fn, d_fp = [dict() for _ in range(4)]
    main_path = '/fred/oz012/Bruno/data/spectra/'
    
    idx = 71 #legacy data; extra loss term; grid; name: wgangp_grid_newloss
    d_ds[idx] = 391672
    d_pn[idx] = ('leg_data_', 'leg_gen_')
    d_fn[idx] = 'spectra_grid'
    d_fp[idx] = os.path.join(main_path, 'legacy/legacy_bit6/')

    idx = 72 #galaxy with STARFORMING and STARBURST plus zWarning=0; extra loss term; grid; name: gal_zWarning
    d_ds[idx] = 321520
    d_pn[idx] = ('gal_zWarning_data_', 'gal_zWarning_gen_')
    d_fn[idx] = 'spectra'
    d_fp[idx] = os.path.join(main_path, 'gal_starforming_starburst_zWarning/')

    idx = 73 #qso plus zWarning=0; extra loss term; grid; name: qso_zWarning
    d_ds[idx] = 22791
    d_pn[idx] = ('qso_zWarning_data_', 'qso_zWarning_gen_')
    d_fn[idx] = 'spectra'
    d_fp[idx] = os.path.join(main_path, 'qso_zWarning/')

    idx = 74 #qso plus zWarning=0; extra loss term; grid; different range: 1800-4150 A; name: qso2_zWarning
    d_ds[idx] = 2346
    d_pn[idx] = ('qso2_zWarning_data_', 'qso2_zWarning_gen_')
    d_fn[idx] = 'spectra2' #'spectra2_linearlysampled_': used when Karl asked for linear sampling
    d_fp[idx] = os.path.join(main_path, 'qso_zWarning/')

    idx = 999 #used for tests
    #assert (filename not in [v for k,v in d_fn.items()]), 'Please do not repeat filenames. This could lead to overwrites.'
    d_ds[idx] = d_ds[73]
    d_pn[idx] = ('TEST_data_', 'TEST_gen_')
    d_fn[idx] = d_fn[73]
    d_fp[idx] = d_fp[73] 

    return d_ds, d_pn, d_fn, d_fp
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)

    dataset_size_d, pics_save_names_d, files_name_d, files_path_d = set_param_dicts()

    nepochs = 10
    checkpoint_dir = '/fred/oz012/Bruno/checkpoints/' + str(FLAGS.checkpoint) + '/'
    tensorboard_dir = '/fred/oz012/Bruno/tensorboard/' + str(FLAGS.checkpoint) + '/'
    
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
