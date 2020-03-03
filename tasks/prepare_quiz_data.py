import os
import tensorflow as tf
from src.models.gans import DCGAN
import luigi

class FakeFITSProducer(luigi.Task):
    path = luigi.Parameter() #Example: quiz/origdata_qso/
    prefix = luigi.Parameter(default='gen')
    checkpoint = luigi.IntParameter()
    nbatches = luigi.IntParameter(default=15)
    nfits_per_batch = luigi.IntParameter(default=10)
    
    def requires(self):
        return [
            MakeDirectory(path=os.path.dirname(self.path)),
        ]
                
    def output(self):
        return None

    def run(self):
        base_path = '/fred/oz012/Bruno'
        checkpoint_dir = os.path.join( base_path, 'checkpoints', str(self.checkpoint)+'/' )
        tensorboard_dir = os.path.join( base_path, 'tensorboard', str(self.checkpoint)+'/' )

        print(checkpoint_dir)
        print(tensorboard_dir)
        with tf.Session() as sess:
            dcgan = DCGAN(sess=sess,
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
                          files_path=os.path.join(base_path, 'data/spectra/qso_zWarning/'),
                          files_name="spectra2_", 
                          checkpoint_dir=checkpoint_dir,
                          tensorboard_dir=tensorboard_dir,
            )
            print(os.path.join(self.path, 'gen_'))
            dcgan.generate(N=self.nbatches, n=self.nfits_per_batch, name=os.path.join(self.path, self.prefix), write_fits=True)

class RealFITSProducer(luigi.Task):
    path = luigi.Parameter()

    def requires(self):
        return [
            MakeDirectory(path=os.path.dirname(self.path)),
        ]

    def output(self):
        return None

    def run(self):
        pass

class MakeDirectory(luigi.Task):
    path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.path)

    def run(self):
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

if __name__ == '__main__':
   luigi.run()
