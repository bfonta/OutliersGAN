import os
import abc
import h5py
import time
import logging
import traceback
import numpy as np
#np.set_printoptions(threshold=np.nan)
import tensorflow as tf
from src.utilities import log_tf_files, tboard_concat
from src.utilities import PlotGenSamples, plot_predictions
from src.models.utilities import minibatch_discrimination, noise
from src.data.tfrecords import read_spectra_data as read_data
from src.data.fits import to_fits

class _GAN(abc.ABC):
    """
    Base class for implementing GAN-like models.
    """
    def __init__(self, sess, mode, checkpoint_dir, tensorboard_dir, 
                 in_width, in_height, nchannels,
                 batch_size, noise_dim, opt_pars, d_iters=1, 
                 dataset_size=None, data_name='mnist', files_name=None, files_path=None,
                 pics_save_names=(None,None)):
        """
        Arguments:

        -> sess: the tensorflow tf.Session() where the model will run
        -> in_height: first data dimension (height in the case of pictures)
        -> in_width: second data dimension (width in the case of pictures)
        -> n_channels: third data dimension (number of colour channels in the case of pictures)
        -> batch_size: size of each data batch. It affects minibatch discrimination and 
           batch normalization
        -> noise_dim: dimension of the generator's input noise
        -> dataset_size: number of items in the dataset
        -> checkpoint_dir: folder which will store model data. Needed for using the model after training.
        -> tensorboard_dir: folder which will store Tensorboard data (Tensorflow visualization tool)
        -> data_name: name of the dataset to be used. Options: mnist, fashion_mnist, cifar10, spectra
        -> mode: whether to train a modified gan version or not. Options: original, wgan-gp
        -> opt_pars: Tuple representing adam optimizer parameters: (learning_rate, beta1, beta2) 
        -> pics_save_names: Tuple containing the names of the data and generated pictures to be saved
        -> d_iters: number of discriminator/critic iterations per generator iteration
        -> files_path: where to read the 'spectra' data from. Ignored when using other 'data_name' 
           options 
        -> files_name: name of the data files. If 'None', data_name is instead used

        Return:
        -> nothing

        Notes:
        - placeholders are usually denoted with the '_ph' substring. They must be fed with scalars or tensors.
        """
        self.sess = sess
        self.in_height = in_height
        self.in_width = in_width
        self.nchannels = nchannels
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.dataset_size = dataset_size
        self.checkpoint_dir = checkpoint_dir
        self.tensorboard_dir = tensorboard_dir
        self.data_name = data_name
        self.mode = mode
        if self.mode not in ('original', 'wgan-gp'):
            raise ValueError('Insert a valid mode.')
        self.opt_pars = opt_pars
        self.pics_save_names = pics_save_names
        if self.pics_save_names == (None,None):
            self.pics_save_names = (self.data_name+'_data_', self.data_name+'_gen_')
        self.d_iters = d_iters
        self.files_path = files_path
        self.files_name = files_name

        self.alpha = 0.2 #leaky relu parameter
        self.image_datasets = {'mnist', 'fashion_mnist', 'cifar10'}
        self.d_layers_names, self.g_layers_names = (() for _ in range(2))

        logging.basicConfig(filename='/tmp/'+self.files_name+'_'
                            +str(checkpoint_dir.split('/')[-2])+'.out')

    @abc.abstractmethod
    def _generator(self, noise, drop):
        raise NotImplementedError('Please implement the generator in your subclass.')

    @abc.abstractmethod
    def _discriminator(self, noise, drop):
        raise NotImplementedError('Please implement the discriminator in your subclass.')

    def _load_data(self, ret=False, files_name=None, data_name=None, data_path=None, data_size=None):
        """
        Loads the training data (must be stored in TFRecord format).
        
        Arguments:
        -> ret: when ret==True the function returns the data and does not store it as a class variable.
           This is useful when using datasets that were not used to train the model.
        -> files_name: name of the files of the dataset to be considered. If 'None' data_name will
           be instead used.
        -> data_name: name of the dataset to be considered. Ex: mnist, spectra
        -> data_path: path of the dataset to be considered
        -> data_size: number of elements of the dataset to be considered

        Returns:
        -> shuffled tf.Dataset when ret==True
        -> number of batches (calculated using data_size) when ret==True
        -> nothing when ret==False
        """
        try:
            if (data_name == None or data_name in self.image_datasets) and data_path is not None:
                raise ValueError('It makes no sense to specify the data_path with this data_name.')
            if data_path != None and ret==False:
                raise ValueError('You cannot save a dataset as a class variable [ret==False]'
                                 'when using a custom data_path.')

            if data_name is None:
                data_name = self.data_name
            if files_name is None:
                if self.files_name is None:
                    files_name = self.data_name
                else:
                    files_name = self.files_name
            if data_size is None:
                data_size = self.dataset_size
            if data_path is None:
                data_path = self.files_path

            if data_name in self.image_datasets:
                if data_name == 'mnist':
                    (train_data, train_labels), _ = tf.keras.datasets.mnist.load_data()
                elif data_name == 'fashion_mnist':
                    (train_data, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
                elif data_name == 'cifar10':
                    (train_data, train_labels), _ = tf.keras.datasets.cifar10.load_data()

                train_data = train_data / 255 if data_size==None else train_data[:data_size] / 255
                dataset_size = len(train_data)
                dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
                dataset = dataset.shuffle(buffer_size=dataset_size).repeat(1).batch(self.batch_size)
                nbatches = int(np.ceil(dataset_size/self.batch_size))

                if ret:
                    return dataset, nbatches

                self.dataset = dataset
                self.nbatches = nbatches

            else:
                if self.files_path is None:
                    raise ValueError('The path of the training data files must be specified.')
                if data_size is None:
                    raise ValueError('The size of the training data must be specified.')

                if data_name == 'spectra':
                    dataset = read_data(data_path+files_name+'.tfrecord', self.in_height)
                nbatches = int(np.ceil(data_size/self.batch_size))

                if ret:
                    return dataset.batch(self.batch_size), nbatches

                self.dataset = dataset.repeat(1).batch(self.batch_size)
                self.nbatches = nbatches

        except Exception as e:
            logging.error(traceback.format_exc())
            raise

    def _pre_process(self, inputs, params):
        if self.data_name == 'mnist' or self.data_name == 'fashion_mnist':
            return np.expand_dims(inputs, axis=3)
        elif self.data_name == 'spectra':
            mean = np.mean(inputs, axis=1, keepdims=True)
            diff = inputs - mean
            std = np.sqrt(np.mean(diff**2, axis=1, keepdims=True))
            inputs = diff / std / 30
            return np.expand_dims(np.expand_dims(inputs, axis=2), axis=3)
        return inputs

    def _build_model(self, generator, discriminator, lr=0.0002, beta1=0.5, beta2=0.999):
        self.batch_size_ph = tf.placeholder(tf.int32, shape=[], name='batch_size_ph') 
        self.real_labels_ph = tf.placeholder(tf.float32, name='real_labels_ph')
        self.fake_labels_ph = tf.placeholder(tf.float32, name='fake_labels_ph')
        self.dropout_prob_ph = tf.placeholder(tf.float32, shape=(), name='dropout_prob_ph')
        self.batch_norm_ph = tf.placeholder_with_default(True, shape=(), name='batch_norm_ph')
        
        with tf.variable_scope('G'):
            self.gen_data_ph = tf.placeholder(tf.float32, shape=[None, self.noise_dim], 
                                              name='gen_data_ph')
            self.G_sample = generator(self.gen_data_ph, drop=self.dropout_prob_ph)

        with tf.variable_scope('D') as scope:
            self.data_ph = tf.placeholder(tf.float32, 
                                          shape=[None, self.in_height, self.in_width, self.nchannels], 
                                          name='data_ph')   
            self.D_real_logits, self.D_real = discriminator(self.data_ph, 
                                                            drop=self.dropout_prob_ph)

        with tf.variable_scope('D', reuse=True):
            self.D_fake_logits, self.D_fake = discriminator(self.G_sample, 
                                                            drop=self.dropout_prob_ph)

        #Gradient Penalty
        if self.mode=='wgan-gp':
            epsilon = tf.random_uniform(shape=[self.batch_size_ph,1,1,1], 
                                        minval=0., maxval=1., name='epsilon')
            interpolate = self.G_sample + epsilon * ( self.data_ph - self.G_sample )

            with tf.variable_scope('D', reuse=True):
                gradients = tf.gradients(discriminator(interpolate, drop=self.dropout_prob_ph),
                                         [interpolate])[0]

            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients)))
            gradient_penalty = tf.reduce_mean( tf.square(slopes - 1.) )
            self.D_loss = tf.reduce_mean(self.D_fake_logits) - tf.reduce_mean(self.D_real_logits)
            self.G_loss = -tf.reduce_mean(self.D_fake_logits)
            self.lambda_gp = tf.placeholder_with_default(10., shape=(), name='lambda_gp_ph')
            self.D_loss += self.lambda_gp * gradient_penalty * tf.maximum(0., -self.D_loss)

        else:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits
            D_loss_real = cross_entropy(logits=self.D_real_logits, labels=self.real_labels_ph)
            D_loss_fake = cross_entropy(logits=self.D_fake_logits, labels=self.fake_labels_ph)
            self.D_loss = tf.reduce_mean(D_loss_real + D_loss_fake) 
            self.G_loss = tf.reduce_mean(cross_entropy(logits=self.D_fake_logits,
                                                       labels=self.real_labels_ph)) 

        #summaries visualized in Tensorboard
        with tf.variable_scope('D_statistics'):
            tf.summary.scalar('loss_d', self.D_loss)
            log_tf_files(layers_names=self.d_layers_names, loss=self.D_loss, scope='D')

        with tf.variable_scope('G_statistics'):
            tf.summary.scalar('loss_g', self.G_loss)    
            log_tf_files(layers_names=self.g_layers_names, loss=self.G_loss, scope='G')

        #how many pictures to stack on each side
        if self.data_name in self.image_datasets:
            self.side = 9 
        else:
            self.side = 1

        self.real_pics_ph = tf.placeholder(tf.float32, 
                    shape=[1,self.in_height*self.side,self.in_width*self.side,self.nchannels],
                    name='real_pics_ph')
        tf.summary.image('real_pic', self.real_pics_ph, max_outputs=1)
        self.gen_pics_ph = tf.placeholder(tf.float32, 
                            shape=[1,self.in_height*self.side,self.in_width*self.side,self.nchannels], 
                            name='gen_pics_ph')
        tf.summary.image('gen_pic', self.gen_pics_ph, max_outputs=1)
        #####################################

        D_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)
        G_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)
        D_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D')
        G_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G')
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.D_train_op = D_optimizer.minimize(self.D_loss, var_list=D_trainable_vars, 
                                                   name='D_train_op')
            self.G_train_op = G_optimizer.minimize(self.G_loss, var_list=G_trainable_vars, 
                                                   name='G_train_op')

    def train(self, nepochs, drop_d=0.7, drop_g=0.3, flip_prob=0.05, restore=False):
        """
        Performs the training of the GAN.

        Arguments:
        -> nepochs: number of training epochs (one epoch corresponds to looking at every data item once)
        -> drop_d: dropout in the discriminator
        -> drop_g: dropout in the generator
        -> flip_prob: label flipping probability for the discriminator
        -> restore: train starting from most recent checkpoint (defined by self.checkpoint_dir)
        """
        self._build_model(self._generator, self._discriminator, 
                          lr=self.opt_pars[0], beta1=self.opt_pars[1], beta2=self.opt_pars[2])

        summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)

        iterator = self.dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        minval, maxval= 0.85, 1.1 #smoothing

        dropout_prob_D = drop_d
        dropout_prob_G = drop_g

        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
        if restore:
            saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_dir))

        init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        self.sess.run(init)

        for epoch in range(nepochs):
            print("Epoch: {}".format(epoch), flush=True)
            self.sess.run(iterator.initializer)

            for batch in range(self.nbatches):
                inputs, *params = self.sess.run(next_element)
                inputs = self._pre_process(inputs, params)

                for _ in range(self.d_iters):
                    #label flipping and smoothing
                    real = np.random.uniform(low=minval, high=maxval, size=(len(inputs),1))
                    fake = np.zeros(shape=(len(inputs),1))
                    _idxs = np.random.choice(np.arange(len(inputs)), 
                                             size=(int(flip_prob*len(inputs))), replace=False)
                    real[_idxs] = 0.
                    fake[_idxs] = np.random.uniform(low=minval, high=maxval, 
                                                    size=(int(flip_prob*len(inputs)),1))

                    #train discriminator
                    noise_D = noise(len(inputs), self.noise_dim)
                    _, D_loss_c, D_real_c, D_fake_c = self.sess.run([self.D_train_op, self.D_loss,
                                                                     self.D_real, self.D_fake],
                                        feed_dict={self.data_ph: inputs, self.gen_data_ph: noise_D,
                                                   self.dropout_prob_ph: dropout_prob_D,
                                                   self.batch_size_ph: len(inputs),
                                                   self.real_labels_ph: real, self.fake_labels_ph: fake})


                noise_G = noise(len(inputs), self.noise_dim)

                #flipping and smoothing for the generator
                real = np.random.uniform(low=minval, high=maxval, size=(len(inputs),1))
                fake = np.zeros(shape=(len(inputs),1))
                _idxs = np.random.choice(np.arange(len(inputs)), 
                                         size=(int(flip_prob*len(inputs))), replace=False)
                real[_idxs] = 0.
                fake[_idxs] = np.random.uniform(low=minval, high=maxval, 
                                                size=(int(flip_prob*len(inputs)),1))
                """
                #no smoothing or flipping
                real = np.ones(shape=(len(inputs),1))
                fake = np.zeros(shape=(len(inputs),1))
                """
                #train generator
                _, G_loss_c  = self.sess.run([self.G_train_op, self.G_loss],
                                      feed_dict={self.data_ph: inputs, self.gen_data_ph: noise_G,
                                                 self.dropout_prob_ph: dropout_prob_G,
                                                 self.batch_size_ph: len(inputs),
                                                 self.real_labels_ph: real, self.fake_labels_ph: fake})

            saver.save(self.sess, self.checkpoint_dir, global_step=1000)

            sample = self.sess.run(self.G_sample, feed_dict={self.gen_data_ph: noise_G,
                                                   self.dropout_prob_ph: dropout_prob_G,
                                                   self.batch_size_ph: self.batch_size,
                                                   self.real_labels_ph: real, self.fake_labels_ph: fake})

            tboard_sample = np.expand_dims(tboard_concat(sample, self.side),0)
            tboard_inputs = np.expand_dims(tboard_concat(inputs, self.side),0)

            summ = self.sess.run(summary, 
                                 feed_dict={self.data_ph: inputs, self.gen_data_ph: noise_G,
                                            self.dropout_prob_ph: dropout_prob_G,
                                            self.batch_size_ph: len(inputs),
                                            self.real_labels_ph: real, self.fake_labels_ph: fake,
                                            self.real_pics_ph: tboard_inputs,
                                            self.gen_pics_ph: tboard_sample})
        
            train_writer.add_summary(summ, epoch*self.nbatches+(batch+1))            
            
            #print generated samples
            self._plot(inputs, params, self.pics_save_names[0]+str(epoch), n=5)
            self._plot(sample, params, self.pics_save_names[1]+str(epoch), n=5)

    def _plot(self, data, params_data, name, n=5, mode='normal'):
        if mode=='normal':
            if self.data_name == 'mnist' or self.data_name == 'fashion_mnist':
                p = PlotGenSamples()
                p.plot_mnist(data[:36], name)
            elif self.data_name == 'cifar10':
                p = PlotGenSamples()
                p.plot_cifar10(data[:36], name)
            elif self.data_name == 'spectra':
                p = PlotGenSamples(nrows=n, ncols=1)
                p.plot_spectra(data[:n], params_data[0][:n], name)
            else: 
                raise Exception('What should I plot?')
        elif mode=='predictions':
            if self.data_name == 'mnist' or self.data_name == 'fashion_mnist':
                p = PlotGenSamples()
                p.plot_mnist(data[:36], name)
            elif self.data_name == 'cifar10':
                p = PlotGenSamples()
                p.plot_cifar10(data[:36], name)
            elif self.data_name == 'spectra':
                p = PlotGenSamples(nrows=n, ncols=1)
                p.plot_spectra(data[:n], params_data[0][:n], name)
            else: 
                raise Exception('What should I plot?')
            

    def predict(self, n_pred):
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
        D_logit = tf.get_default_graph().get_tensor_by_name('D/logit:0')
        D_noise = noise(n_pred, self.noise_dim)
            
        gen_data_ph = tf.get_default_graph().get_tensor_by_name('G/gen_data_ph:0')
        data_ph = tf.get_default_graph().get_tensor_by_name('D/data_ph:0')
        dropout_prob_ph = tf.get_default_graph().get_tensor_by_name('dropout_prob_ph:0')
        batch_size_ph = tf.get_default_graph().get_tensor_by_name('batch_size_ph:0')
        batch_norm_ph = tf.get_default_graph().get_tensor_by_name('batch_norm_ph:0')

        iterator = self.dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        self.sess.run(iterator.initializer)
        saver.restore(self.sess, latest_checkpoint)

        ps = [] #used for plotting
        for b in range(self.nbatches):
            inputs, *params = self.sess.run(next_element)
            inputs = self._pre_process(inputs, params)

            #Makes sure that the number of predictions is the one the user wants
            if (b+1)*self.batch_size > n_pred:
                inputs = inputs[:n_pred - b*self.batch_size]

            predictions = self.sess.run(D_logit, 
                feed_dict={data_ph: inputs, gen_data_ph: D_noise,
                           dropout_prob_ph: 0.,
                           batch_norm_ph: False})
            for i,pred in enumerate(predictions, start=1):
                print('Batch: {} Prediction {}: {}'.format(b+1, i, pred[0]))

            #Store predictions for plotting
            ps.append(np.array(predictions))    

            #Stopping criterion
            if len(inputs) != self.batch_size:
                break

        plot_predictions(ps, '1')

    def generate(self, N, n, name, write_fits=False):
        """
        Generate fake spectra using trained GAN model.

        Arguments:
        -> N: number of files to create (plots or FITS)
        -> n: number of spectra to store in each file
        -> name: name of the files (numbers are added according to N)
        -> write_fits: whether to write the spectra to FITS files with WCS conversion. This assumes 
        that a fixed grid was defined.

        Returns:
        -> nothing
        """
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
        G_output = tf.get_default_graph().get_tensor_by_name('G/output:0')
        gen_data_ph = tf.get_default_graph().get_tensor_by_name('G/gen_data_ph:0')
        dropout_prob_ph = tf.get_default_graph().get_tensor_by_name('dropout_prob_ph:0')
        batch_size_ph = tf.get_default_graph().get_tensor_by_name('batch_size_ph:0')

        if self.data_name == 'spectra':
            iterator = self.dataset.make_initializable_iterator()
            self.sess.run(iterator.initializer)
            next_element = iterator.get_next()
            _, *params = self.sess.run(next_element)

        saver.restore(self.sess, latest_checkpoint)

        for i in range(N):
            if not os.path.isdir( os.path.join(os.path.dirname(name), 'batch'+str(i)) ):
                os.makedirs( os.path.join(os.path.dirname(name), 'batch'+str(i)) )
            gen_samples = self.sess.run(G_output, 
                                        feed_dict={gen_data_ph: noise(n,self.noise_dim),
                                                   dropout_prob_ph: 0., 
                                                   batch_size_ph: n})
            self._plot(gen_samples, params, os.path.basename(name)+str(i), n=n)
            if write_fits:
                init_l = np.log10(params[0][0][0]) #initial wavelength
                delta_l = np.log10(params[0][0][1])-np.log10(params[0][0][0]) #wavelengthth bin width
                assert np.isclose(delta_l, np.log10(params[0][0][1000])-np.log10(params[0][0][999]),
                                  atol=1e-6) #check bin uniformity
                """ #used when Karl asked for linear sampling
                init_l = params[0][0][0] #initial wavelength
                delta_l = params[0][0][1] - params[0][0][0] #wavelengthth bin width
                assert np.isclose(delta_l, params[0][0][1000]-params[0][0][999], atol=1e-6) #check bin uniformity
                """
                gen_samples = gen_samples.reshape((n,gen_samples.shape[1]))
                print(name+str(i), init_l, delta_l)
                path = os.path.join( os.path.dirname(name), 'batch'+str(i), os.path.basename(name) )
                to_fits(gen_samples, path, params=(1., delta_l, init_l))

    def save_features(self, ninputs, save_path, 
                      additional_files_name=None, additional_data_path=None, additional_ninputs=None):
        """
        Saves features obtained by running the data through a saved discriminator model.
        The layer from where the data is retrieved is not the last one.
        By selecting a layer before the last, one hopes to extract some features the net has learned.
        The featues are save into a hdf5 file for further analysis.
        Notes: 1. 'BiasAdd' is the tensorflow operation that adds a bias to the input data.
               2. 'MatMul' is instead used if one wants to consider the weights only.
        ---Arguments---
        ->ninputs: 
           number of inputs whose features will be saved. The inputs used will be different
           for different runs, since the datasets are always shuffled.
        ->save_path: 
           where to save the hdf5 file. The extension should not be included.
        ->additional_files_name: 
           name of the files to be saved. This allows the user to save the features for more than one 
           dataset. In order to see the discriminator response, it is useful to be able to run data on 
           a model that was trained using different data.
        ->additional_data_path: 
           path to the folder where the additional dataset is stored. The dataset should be stored 
           in that folder in one or more Tfrecord shards with the following names:  
           additional_data_name0.tfrecord, additional_data_name1.tfrecord, ...
        ->additional_ninputs: 
           number of samples to consider for the TFRecords additional dataset. Defaults to the number 
           defined in the original dataset. The user has to make sure that the data size is correct 
           for the dataset specified in the 'additional_data_path', otherwise the number of batches
           will be wrong; if it is too large, the code will throw an out of bounds error; if it is
           too small, part of the dataset will not be considered when obtaining the features.
        """ 
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)
        saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
        D_noise = noise(ninputs, self.noise_dim)

        D_feats = tf.get_default_graph().get_tensor_by_name('D/layer2/BiasAdd:0')
        D_feats_shape = D_feats.get_shape()
        D_feats = tf.reshape(D_feats, 
                             shape=[-1,D_feats_shape[1]*D_feats_shape[2]*D_feats_shape[3]])
        D_feats2 = tf.get_default_graph().get_tensor_by_name('D/layer4/BiasAdd:0')
        D_feats2_shape = D_feats2.get_shape()
        D_feats2 = tf.reshape(D_feats2, 
                              shape=[-1,D_feats2_shape[1]*D_feats2_shape[2]*D_feats2_shape[3]])
            
        data_ph = tf.get_default_graph().get_tensor_by_name('D/data_ph:0')
        dropout_prob_ph = tf.get_default_graph().get_tensor_by_name('dropout_prob_ph:0')
        batch_size_ph = tf.get_default_graph().get_tensor_by_name('batch_size_ph:0')
        batch_norm_ph = tf.get_default_graph().get_tensor_by_name('batch_norm_ph:0')

        iterator = self.dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        self.sess.run(iterator.initializer)

        if additional_ninputs is None:
            additional_ninputs = self.dataset_size
        if additional_files_name is not None:
            dataset2, nbatches2 = self._load_data(ret=True, 
                                                  data_name=self.data_name,
                                                  files_name=additional_files_name,
                                                  data_path=additional_data_path,
                                                  data_size=additional_ninputs)
            iterator2 = dataset2.make_initializable_iterator()
            next_element2 = iterator2.get_next()
            self.sess.run(iterator2.initializer)

        saver.restore(self.sess, latest_checkpoint)

        save_start = 0
        M = D_feats.get_shape()[1] + D_feats2.get_shape()[1]
        
        with h5py.File(save_path+'.hdf5', 'w') as f:
            
            group = f.create_group('data')
            dset = group.create_dataset(self.data_name, (ninputs, M), dtype=np.float32) 

            for b in range(self.nbatches):
                inputs, *params = self.sess.run(next_element)
                inputs = self._pre_process(inputs, params)

                #Makes sure that the number of input data is the one the user wants
                if (b+1)*self.batch_size > ninputs:
                    inputs = inputs[:ninputs - b*self.batch_size]

                feats, feats2 = self.sess.run([D_feats, D_feats2],
                                      feed_dict={data_ph: inputs, #gen_data_ph: D_noise,
                                                 dropout_prob_ph: 0.,
                                                 batch_norm_ph: False})
                feats = np.concatenate((feats,feats2), axis=1)
                dset[save_start:save_start+len(feats), :] = feats
                save_start += len(feats)

                #Stopping criterion
                if len(inputs) != self.batch_size:
                    break

            save_start = 0
            count = 0
            if additional_files_name is not None:
                dset = group.create_dataset(self.data_name+'_additional', 
                                            (additional_ninputs, M), dtype=np.float32) 
                for b in range(nbatches2):
                    try:
                        inputs, *params = self.sess.run(next_element2)
                        inputs = self._pre_process(inputs, params)
                    
                        #Makes sure that the number of input data is the one the user wants
                        if (b+1)*self.batch_size > additional_ninputs:
                            inputs = inputs[:additional_ninputs - b*self.batch_size]

                        count += len(inputs)
                        feats, feats2 = self.sess.run([D_feats,D_feats2],
                                              feed_dict={data_ph: inputs, #gen_data_ph: D_noise,
                                                         dropout_prob_ph: 0.,
                                                         batch_norm_ph: False})
                        feats = np.concatenate((feats,feats2), axis=1)
                        dset[save_start:save_start+len(feats), :] = feats
                        save_start += len(feats)

                    except tf.errors.OutOfRangeError:
                        raise IndexError('Make sure the size of the additional data set is'
                                         'equal or smaller than the actual number of samples.')
                        
                print("Number of inputs: ", count)

##################################################################################################
class DCGAN(_GAN):
    """
    Generative Adversarial Network using a Deep Convolutional architecture
    Reference: Radford et al; ArXiv: 1511.06434
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_data()

        if self.data_name in self.image_datasets:
            self.filters = [128, 256, 512]
        elif self.data_name == 'spectra':
            self.filters = [128, 256, 512, 1024]

    def _discriminator(self, x, drop):
        def conv_block(nn, conv_channels, kernel_size, strides, name):
            nn = tf.layers.conv2d(nn, filters=conv_channels, kernel_size=kernel_size, 
                                  strides=strides, activation=None, padding='same', 
                                  name=name)
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn, training=self.batch_norm_ph)
            return tf.layers.dropout(nn, rate=drop)

        net = tf.reshape(x, shape=[-1, self.in_height, self.in_width, self.nchannels])

        if self.data_name == 'mnist':
            #add zeros for the more convenient 32x32 shape
            net = tf.pad(net, paddings=[[0,0],[2,2],[2,2],[0,0]], mode='CONSTANT', constant_values=0.)

        if self.data_name in self.image_datasets:
            self.d_layers_names += ('layer1',)
            net = conv_block(net, conv_channels=self.filters[0], 
                             kernel_size=[5,5], strides=[2,2], name=self.d_layers_names[-1])
            self.d_layers_names += ('layer2',)
            net = conv_block(net, conv_channels=self.filters[1], 
                             kernel_size=[5,5], strides=[2,2], name=self.d_layers_names[-1])
            self.d_layers_names += ('layer3',)
            net = conv_block(net, conv_channels=self.filters[2], 
                             kernel_size=[5,5], strides=[2,2], name=self.d_layers_names[-1])

        elif self.data_name == 'spectra':
            self.d_layers_names += ('layer1',)
            net = conv_block(net, conv_channels=self.filters[0], 
                             kernel_size=[10,1], strides=[5,1], name=self.d_layers_names[-1])
            self.d_layers_names += ('layer2',)
            net = conv_block(net, conv_channels=self.filters[1],
                             kernel_size=[10,1], strides=[5,1], name=self.d_layers_names[-1])
            self.d_layers_names += ('layer3',)
            net = conv_block(net, conv_channels=self.filters[2], 
                             kernel_size=[10,1], strides=[5,1], name=self.d_layers_names[-1])
            self.d_layers_names += ('layer4',)
            net = conv_block(net, conv_channels=self.filters[3], 
                             kernel_size=[10,1], strides=[4,1], name=self.d_layers_names[-1])

        final_shape = net.get_shape()
        net = tf.reshape(net, shape=[-1,self.filters[-1]*final_shape[1]*final_shape[2]])

        if self.mode != 'wgan-gp':
            net = minibatch_discrimination(net, num_kernels=30, kernel_dim=20, 
                                           name='minibatch_discrimination')
        #self.d_layers_names += ('final_dense',)
        #net = tf.layers.dense(net, 1024, activation=None, name=self.d_layers_names[-1])
        self.d_layers_names += ('dense_output',)
        net = tf.layers.dense(net, 1, activation=None, name=self.d_layers_names[-1])
        return net, tf.nn.sigmoid(net, name='logit')
    
    def _generator(self, noise, drop):
        if self.data_name in self.image_datasets:
            init_height, init_width = 4, 4
        elif self.data_name == 'spectra':
            init_height, init_width = 7, 1

        def conv_block(nn, conv_channels, kernel_size, strides, name):
            nn = tf.layers.conv2d_transpose(nn, filters=conv_channels, 
                                            kernel_size=kernel_size, strides=strides,
                                            activation=None, padding='same', name=name)
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn, training=self.batch_norm_ph)
            nn = tf.layers.dropout(nn, rate=drop)
            return nn
        
        net = tf.reshape(noise, shape=[-1, noise.shape[1]])
        self.g_layers_names += ('dense_input',)
        net = tf.layers.dense(net, init_height*init_width*self.filters[-1], 
                              activation=None, name=self.g_layers_names[-1])
        net = tf.layers.dropout(net, rate=drop)
        net = tf.reshape(net, shape=[-1,init_height,init_width,self.filters[-1]])

        if self.data_name in self.image_datasets:
            self.g_layers_names += ('layer1',)
            net = conv_block(net, conv_channels=self.filters[-2], 
                             kernel_size=[5,5], strides=[2,2], name=self.g_layers_names[-1])
            self.g_layers_names += ('layer2',)
            net = conv_block(net, conv_channels=self.filters[-3], 
                             kernel_size=[5,5], strides=[2,2], name=self.g_layers_names[-1])
            self.g_layers_names += ('layer3',)
            net = tf.layers.conv2d_transpose(net, filters=self.nchannels, kernel_size=[5,5], 
                                             strides=[2,2], activation=None, padding='same', 
                                             name=self.g_layers_names[-1])
            if self.data_name == 'mnist' or self.data_name == 'fashion_mnist':
                net = tf.slice(net, begin=[0,2,2,0], size=[-1,self.in_width,self.in_height,-1])
            net = tf.divide( tf.add(tf.nn.tanh(net),1.), 2.)

        elif self.data_name == 'spectra':
            self.g_layers_names += ('layer1',)
            net = conv_block(net, conv_channels=self.filters[-2], 
                             kernel_size=[10,1], strides=[4,1], name=self.g_layers_names[-1])
            self.g_layers_names += ('layer2',)
            net = conv_block(net, conv_channels=self.filters[-3], 
                             kernel_size=[10,1], strides=[5,1], name=self.g_layers_names[-1])
            self.g_layers_names += ('layer3',)
            net = conv_block(net, conv_channels=self.filters[-4], 
                             kernel_size=[10,1], strides=[5,1], name=self.g_layers_names[-1])
            self.g_layers_names += ('layer4',)
            net = tf.layers.conv2d_transpose(net, filters=self.nchannels, 
                                             kernel_size=[10,1], strides=[5,1], 
                                             activation=None, padding='same', 
                                             name=self.g_layers_names[-1])
            if self.mode != 'wgan-gp':
                net = tf.layers.batch_normalization(net, training=self.batch_norm_ph)
            net = tf.nn.tanh(net)

        return tf.reshape(net, shape=[-1, self.in_height, self.in_width, self.nchannels], 
                          name='output')        


#################################################################################################
class ResNetGAN(_GAN):
    """
    Generative Adversarial Network built using residual blocks.
    The architecture is heavily based on residual networks (ResNets).
    Reference: He et al; ArXiv:1512.03385
    The generator is similar to the original ResNet discriminator, but using transposed convolutions
    instead; it represents an original contribution.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_data()
        #self.filters = [64, 128, 256]
        self.filters = [32, 64, 128]

    def _discriminator(self, x, drop):        
        def conv_block(nn, conv_channels, dim_stride, name1, name2):
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn, training=self.batch_norm)
            nn = tf.layers.conv2d(nn, filters=conv_channels, kernel_size=[3, 3], 
                                  strides=[dim_stride, dim_stride],
                                  activation=None, padding='same', 
                                  kernel_initializer=tf.initializers.orthogonal(),
                                  name=name1)
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn, training=self.batch_norm)
            nn = tf.layers.conv2d(nn, filters=conv_channels, kernel_size=[3, 3], 
                                  strides=[1,1],
                                  activation=None, padding='same', 
                                  name=name2)
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn, training=self.batch_norm)
            return nn

            
        def double_res_block(inputs, conv_channels, names):
            if conv_channels == self.filters[0]: dim_stride = 1
            else: dim_stride = 2

            nn = conv_block(inputs, conv_channels=conv_channels,
                             dim_stride=dim_stride,
                             name1=names[0], name2=names[1])

            if dim_stride == 2:
                inputs = tf.layers.conv2d(inputs, filters=conv_channels, kernel_size=[1, 1], 
                                          strides=[2,2],
                                          activation=None, padding='same', 
                                          kernel_initializer=tf.initializers.orthogonal(),
                                          name='input_conv_'+names[2])

            inputs = tf.add(nn, inputs)
            nn = conv_block(inputs, conv_channels=conv_channels,
                             dim_stride=1,
                             name1=names[2], name2=names[3])
            return tf.add(nn, inputs)
            

        inputs = tf.reshape(x, shape=[-1, self.in_height, self.in_width, self.nchannels])

        if self.data_name == 'mnist' or self.data_name == 'fashion_mnist':
            #reshape (adding zeros) to the more convenient 32x32 shape
            inputs = tf.pad(inputs, paddings=[[0,0],[2,2],[2,2],[0,0]], 
                            mode='CONSTANT', constant_values=0.)

        inputs = tf.layers.conv2d(inputs, filters=self.filters[0], kernel_size=[7, 7], 
                                  strides=[2, 2],
                                  activation=None, padding='same', 
                                  kernel_initializer=tf.initializers.orthogonal(),
                                  name='first_conv_layer')

        self.d_layers_names += ('layer1_1', 'layer1_2', 'layer1_3', 'layer1_4')
        inputs = double_res_block(inputs, conv_channels=self.filters[0], names=self.d_layers_names[-4:])
        self.d_layers_names += ('layer2_1', 'layer2_2', 'layer2_3', 'layer2_4')
        inputs = double_res_block(inputs, conv_channels=self.filters[1], names=self.d_layers_names[-4:])
        self.d_layers_names += ('layer3_1', 'layer3_2', 'layer3_3', 'layer3_4')
        net = double_res_block(inputs, conv_channels=self.filters[2], names=self.d_layers_names[-4:])

        final_shape = net.get_shape()
        net = tf.reshape(net, shape=[-1,self.filters[2]*final_shape[1]*final_shape[2]])
        if self.mode != 'wgan-gp':
            net = minibatch_discrimination(net, num_kernels=20, kernel_dim=10)
        net = tf.nn.leaky_relu(net, alpha=self.alpha)
        if self.mode != 'wgan-gp':
            net = tf.layers.batch_normalization(net, training=self.batch_norm)
        self.d_layers_names += ('dense_output',)
        net = tf.layers.dense(net, 1, activation=None, name=self.d_layers_names[-1])
        return net, tf.nn.sigmoid(net, name='logit')

    
    def _generator(self, noise, drop):
        initial_shape = 4

        def conv_block(nn, conv_channels, dim_stride, name1, name2):
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn, training=self.batch_norm)
            nn = tf.layers.conv2d_transpose(nn, filters=conv_channels, kernel_size=[3,3], 
                                            strides=[dim_stride,dim_stride],
                                            kernel_initializer=tf.initializers.orthogonal(),
                                            activation=None, padding='same', name=name1)
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn, training=self.batch_norm)
            nn = tf.layers.conv2d_transpose(nn, filters=conv_channels, kernel_size=[3,3], 
                                            strides=[1,1],
                                            activation=None, padding='same', name=name2)
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn, training=self.batch_norm)
            return nn

            
        def double_res_block(inputs, conv_channels, pad_input, names):
            if conv_channels == self.filters[-1]: dim_stride = 1
            else: dim_stride = 2

            nn = conv_block(inputs, conv_channels=conv_channels,
                            dim_stride=dim_stride,
                            name1=names[0], name2=names[1])

            if dim_stride == 2:
                inputs = tf.layers.conv2d_transpose(inputs, filters=int(int(inputs.shape[3])/2), 
                                                kernel_size=[1,1], 
                                                strides=[dim_stride,dim_stride],
                                                activation=None, padding='same', 
                                                kernel_initializer=tf.initializers.orthogonal(),
                                                name='input_conv'+names[2])

            inputs = tf.add(nn, inputs)
            nn = conv_block(inputs, conv_channels=conv_channels,
                            dim_stride=1,
                            name1=names[2], name2=names[3])
            return tf.add(nn, inputs)
        
        net = tf.reshape(noise, shape=[-1, noise.shape[1]])
        self.g_layers_names += ('dense_input',)
        inputs = tf.layers.dense(noise, initial_shape*initial_shape*self.nchannels, 
                                 activation=None, name=self.g_layers_names[-1])
        inputs = tf.reshape(inputs, shape=[-1,initial_shape,initial_shape,self.nchannels])
        self.g_layers_names += ('first_conv_layer',)
        inputs = tf.layers.conv2d_transpose(inputs, filters=1, 
                                            kernel_size=[3,3], 
                                            strides=[2,2],
                                            activation=None, padding='same', 
                                            name=self.g_layers_names[-1])

        self.g_layers_names += ('layer1_1', 'layer1_2', 'layer1_3', 'layer1_4')
        inputs = double_res_block(inputs, conv_channels=self.filters[-1], pad_input=initial_shape/2, 
                                  names=self.g_layers_names[-4:])
        self.g_layers_names += ('layer2_1', 'layer2_2', 'layer2_3', 'layer2_4')
        inputs = double_res_block(inputs, conv_channels=self.filters[-2], pad_input=initial_shape, 
                                  names=self.g_layers_names[-4:])
        self.g_layers_names += ('layer3_1', 'layer3_2', 'layer3_3', 'layer3_4')
        net = double_res_block(inputs, conv_channels=self.filters[-3], pad_input=initial_shape*2, 
                               names=self.g_layers_names[-4:])

        net = tf.nn.leaky_relu(net, alpha=self.alpha)
        if self.mode != 'wgan-gp':
            net = tf.layers.batch_normalization(net, training=self.batch_norm)
        net = tf.layers.conv2d_transpose(net, filters=self.nchannels, kernel_size=[7,7], 
                                         strides=[1,1],
                                         activation=None, padding='same', 
                                         kernel_initializer=tf.initializers.orthogonal(),
                                         name='last_convolution')

        if self.data_name == 'mnist' or self.data_name == 'fashion_mnist':
            #crop the outer parts of the images to retrieve the original 28x28 size
            net = tf.slice(net, begin=[0,2,2,0], size=[-1,self.in_height,self.in_width,-1])

        if self.data_name in self.image_datasets:
            net = tf.divide( tf.add(tf.nn.tanh(net),1.), 2.)
        else:
            net = tf.nn.tanh(net)

        return tf.reshape(net, shape=[-1, self.in_height, self.in_width, self.nchannels], name='output')
