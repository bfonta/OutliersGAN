import os
from subprocess import call
import tensorflow as tf
from src.models.gans import DCGAN
from post_processing.random_spectra_selection import main as rsselection
import luigi

class SetQuiz(luigi.Task):
    path = luigi.Parameter()
    checkpoint = luigi.IntParameter()
    data_type = luigi.Parameter() #qso or gal
    nbatches = luigi.IntParameter(default=15)
    nfits_per_batch = luigi.IntParameter(default=10)
    
    def requires(self):
        return [
            FakeFITSProducer(path=os.path.dirname(self.path),
                             checkpoint=self.checkpoint,
                             nbatches=self.nbatches,
                             nfits_per_batch=self.nfits_per_batch),
            RealFITSProducer(path=os.path.dirname(self.path),
                             data_type=self.data_type,
                             nbatches=self.nbatches,
                             nfits_per_batch=self.nfits_per_batch)
        ]

    def output(self):
        return luigi.LocalTarget('SetQuiz_out.txt')

    def run(self):
        call('bash quiz/run.sh test '+str(self.nbatches), shell=True)
        
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
        return luigi.LocalTarget(self.path)

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
    data_type = luigi.Parameter() #qso or gal
    prefix = luigi.Parameter(default='data')
    nbatches = luigi.IntParameter(default=15)
    nfits_per_batch = luigi.IntParameter(default=10)
    
    def requires(self):
        return [
            MakeDirectory(path=os.path.dirname(self.path)),
        ]

    def output(self):
        return luigi.LocalTarget(self.path)

    def run(self):
        if self.data_type != 'qso' and self.data_type != 'gal':
            raise ValueError('The data_type must be "qso" or "gal"!')
        rsselection(os.path.join(self.path, self.prefix), self.data_type, self.nbatches, self.nfits_per_batch)

class MakeDirectory(luigi.Task):
    path = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(self.path)

    def run(self):
        print("MAKEDIRS!!!!!!!!!!!! ", self.path)
        if not os.path.isdir(self.path):
            os.makedirs(self.path)

if __name__ == '__main__':
   luigi.run()
