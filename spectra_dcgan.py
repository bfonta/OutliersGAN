import os
import tensorflow as tf
from src.models.gans import DCGAN
import argparse
from src import argparser

def main(argv=None):
    nepochs = 1000
    checkpoint_dir = "/fred/oz012/Bruno/checkpoints/" + str(FLAGS.checkpoint) + "/"
    tensorboard_dir = "/fred/oz012/Bruno/tensorboard/" + str(FLAGS.checkpoint) + "/"

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
            mode="wgan-gp",
            opt_pars=(0.00005, 0.0, 0.9),
            d_iters=5,
            data_name="spectra",
            dataset_size=22791,  
            pics_save_names=("qso_zWarning_data_", "qso_zWarning_gen_"),
            files_path="/fred/oz012/Bruno/data/spectra/qso_zWarning/",
            #files_name="spectra2_linearlysampled_", #used when Karl asked for linear sampling
            files_name="spectra2_", 
            checkpoint_dir=checkpoint_dir,
            tensorboard_dir=tensorboard_dir,
        )

        if FLAGS.mode == "train":
            dcgan.train(nepochs, drop_d=0.0, drop_g=0.0, flip_prob=0.15, restore=False)

        elif FLAGS.mode == "generate":
            dcgan.generate(N=15, n=10, name=FLAGS.fname, write_fits=True)

        elif FLAGS.mode == "predict":
            dcgan.predict(n_pred=514)

        elif FLAGS.mode == "save_features":
            inp = 22791
            dcgan.save_features(
                ninputs=inp,
                save_path="/fred/oz012/Bruno/data/hdf5/tf_qso_gal_" + str(inp),
                additional_files_name="spectra",
                additional_data_path="/fred/oz012/Bruno/data/spectra/gal_starforming_starburst_zWarning/",
                additional_ninputs=inp,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
