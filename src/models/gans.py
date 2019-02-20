import os
import numpy as np
#np.set_printoptions(threshold=np.nan)
import tensorflow as tf
from src.utilities import log_tf_files, tboard_concat
from src.utilities import PlotGenSamples
from src.models.utilities import minibatch_discrimination, noise
from src.data.data import read_spectra_data as read_data

class _GAN():
    """
    Base class for all the implemented GAN-like models.
    """
    def __init__(self, sess, checkpoint_dir, tensorboard_dir, in_width=28, in_height=28, nchannels=1,
                 batch_size=128, noise_dim=100, dataset_size=None, data_name='mnist', mode='original', 
                 opt_pars=(0.0001,0.5,0.999), pics_save_names=(None,None), d_iters=1, files_path=None):
        """
        Arguments:

        sess: the tensorflow tf.Session() where the model will run
        in_height: first data dimension (height in the case of pictures)
        in_width: second data dimension (width in the case of pictures)
        n_channels: third data dimension (number of colour channels in the case of pictures)
        batch_size: size of each data batch. It affects minibatch discrimination and batch normalization
        noise_dim: dimension of the generator's input noise
        dataset_size: number of items in the dataset
        checkpoint_dir: folder which will store model data. Needed for using the model after training.
        tensorboard_dir: folder which will store Tensorboard data (Tensorflow visualization tool)
        data_name: name of the dataset to be used. Options: mnist, fashion_mnist, cifar10, spectra
        mode: whether to train a modified gan version or not. Options: original, wgan-gp
        opt_pars: Tuple representing adam optimizer parameters: (learning_rate, beta1, beta2) 
        pics_save_names: Tuple containing the names of the data and generated pictures to be saved
        d_iters: number of discriminator/critic iterations per generator iteration
        files_path: where to read the 'spectra' data from. Ignored when using other 'data_name' options 
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

        self.alpha = 0.2 #leaky relu parameter
        self.image_datasets = {'mnist', 'fashion_mnist', 'cifar10'}
        self.d_layers_names, self.g_layers_names = (() for _ in range(2))

    def _load_data(self):
        if self.data_name in self.image_datasets:
            if self.data_name == 'mnist':
                (train_data, train_labels), _ = tf.keras.datasets.mnist.load_data()
            elif self.data_name == 'fashion_mnist':
                (train_data, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
            elif self.data_name == 'cifar10':
                (train_data, train_labels), _ = tf.keras.datasets.cifar10.load_data()

            train_data = train_data / 255.
            dataset_size = len(train_data)
            dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
            self.dataset = dataset.shuffle(buffer_size=dataset_size).repeat(1).batch(self.batch_size)
            self.nbatches = int(np.ceil(dataset_size/self.batch_size))
        
        else:
            if self.files_path == None:
                raise ValueError('The path of the training data files must be specified.')
            if  self.dataset_size == None:
                raise ValueError('The size of the training data must be specified.')

            if self.data_name == 'spectra':
                dataset = read_data(self.files_path+'spectra.tfrecord', self.in_height)

            self.dataset = dataset.repeat(1).batch(self.batch_size)
            self.nbatches = int(np.ceil(self.dataset_size/self.batch_size))

    def _pre_process(self, inputs, params):
        if self.data_name == 'mnist' or self.data_name == 'fashion_mnist':
            return np.expand_dims(inputs, axis=3)
        elif self.data_name == 'spectra':
            mean = np.mean(inputs, axis=1, keepdims=True)
            diff = inputs - mean
            std = np.sqrt(np.mean(diff**2, axis=1, keepdims=True))
            inputs = diff / std / 10
            return np.expand_dims(np.expand_dims(inputs, axis=2), axis=3)
        return inputs

    def _build_model(self, generator, discriminator, lr=0.0002, beta1=0.5, beta2=0.999):
        self.batch_size_ph = tf.placeholder(tf.int32, shape=[], name='batch_size_ph') 
        self.real_labels_ph = tf.placeholder(tf.float32, name='real_labels_ph')
        self.fake_labels_ph = tf.placeholder(tf.float32, name='fake_labels_ph')
        self.dropout_prob_ph = tf.placeholder(tf.float32, shape=(), name='dropout_prob_ph')
        
        with tf.variable_scope('G'):
            self.gen_data_ph = tf.placeholder(tf.float32, shape=[None, self.noise_dim], 
                                              name='gen_data_ph')
            self.G_sample = generator(self.gen_data_ph, drop=self.dropout_prob_ph)

        with tf.variable_scope('D') as scope:
            self.data_ph = tf.placeholder(tf.float32, 
                                          shape=[None, self.in_height, self.in_width, self.nchannels], 
                                          name='data_ph')   
            self.D_real_logits, self.D_real = discriminator(self.data_ph, drop=self.dropout_prob_ph)

        with tf.variable_scope('D', reuse=True):
            self.D_fake_logits, self.D_fake = discriminator(self.G_sample, drop=self.dropout_prob_ph)

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
            self.D_loss += self.lambda_gp * gradient_penalty

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

        #how many pictures to stack in each side
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
        
        self.D_train_op = D_optimizer.minimize(self.D_loss, var_list=D_trainable_vars, 
                                               name='D_train_op')
        self.G_train_op = G_optimizer.minimize(self.G_loss, var_list=G_trainable_vars, 
                                               name='G_train_op')

    def train(self, nepochs, drop_d=0.7, drop_g=0.3):
        """
        Arguments:

        nepochs: number of training epochs (one epoch corresponds to looking at every data item once)
        drop_d: dropout in the discriminator
        drop_g: dropout in the generator
        """
        self._build_model(self._generator, self._discriminator, 
                          lr=self.opt_pars[0], beta1=self.opt_pars[1], beta2=self.opt_pars[2])

        summary = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(self.tensorboard_dir, self.sess.graph)

        iterator = self.dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        flip_prob = 0.05 #label flipping for the discriminator
        minval, maxval= 0.85, 1.1 #smoothing

        dropout_prob_D = drop_d
        dropout_prob_G = drop_g

        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

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

                #no flipping or smoothing for the generator
                real = np.ones(shape=(len(inputs),1))
                fake = np.zeros(shape=(len(inputs),1))

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

    def _plot(self, data, params_data, name, n=5):
        if self.data_name == 'mnist' or self.data_name == 'fashion_mnist':
            p = PlotGenSamples()
            p.plot_mnist(data[:36], name)
            print(name)
        elif self.data_name == 'cifar10':
            p = PlotGenSamples()
            p.plot_cifar10(data[:36], name)
        elif self.data_name == 'spectra':
            p = PlotGenSamples(nrows=n, ncols=1)
            p.plot_spectra(data[:n], params_data[0][:n], name)
        else: 
            raise Exception('What should I plot?')


    def predict(self, checkpoint, noise_size, n_predictions=1):
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint)
        saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
        D_logit = tf.get_default_graph().get_tensor_by_name('D/discriminator_logit:0')
        D_noise = noise(n_predictions, noise_size)
        
        (train_data, train_labels), _ = load_data()
        train_data = train_data[:n_predictions]
        train_data = train_data / 255.
    
        gen_data_ph = tf.get_default_graph().get_tensor_by_name('G/gen_data_ph:0')
        data_ph = tf.get_default_graph().get_tensor_by_name('D/data_ph:0')
        dropout_prob_ph = tf.get_default_graph().get_tensor_by_name('dropout_prob_ph:0')
        batch_size_ph = tf.get_default_graph().get_tensor_by_name('batch_size_ph:0')

        with tf.Session() as sess:
            saver.restore(sess, latest_checkpoint)
            predictions = sess.run(D_logit, feed_dict={data_ph: train_data, gen_data_ph: D_noise,
                                                       dropout_prob_ph: 0., 
                                                       batch_size_ph: 1})
        print(predictions)
        quit()
        for i,pred in enumerate(predictions):
            print('Prediction {}: {}'.format(i, pred))

    def generate(self, n, name):
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
        gen_samples = self.sess.run(G_output, 
                                    feed_dict={gen_data_ph: noise(n,self.noise_dim),
                                               dropout_prob_ph: 0., 
                                               batch_size_ph: n})

        self._plot(gen_samples, params, name, n=n)

##################################################################################################
class DCGAN(_GAN):
    """
    Generative Adversarial Network using a Deep Convolutional architecture
    Reference: Radford et al; ArXiv: 1511.06434
    """
    def __init__(self, sess, checkpoint_dir, tensorboard_dir, in_height=28, in_width=28, nchannels=1,
                 batch_size=128, noise_dim=100, dataset_size=None,
                 data_name='mnist', mode='original', opt_pars=(0.0001, 0.5, 0.999), 
                 pics_save_names=(None,None), d_iters=1, files_path=None):
        super().__init__(sess=sess, in_height=in_height, in_width=in_width, nchannels=nchannels,
                         batch_size=batch_size, noise_dim=noise_dim, dataset_size=dataset_size, 
                         checkpoint_dir=checkpoint_dir, tensorboard_dir=tensorboard_dir,
                         mode=mode, opt_pars=opt_pars, pics_save_names=pics_save_names,
                         d_iters=d_iters, data_name=data_name, files_path=files_path)
        super()._load_data()

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
                nn = tf.layers.batch_normalization(nn)
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
                nn = tf.layers.batch_normalization(nn)
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
                net = tf.layers.batch_normalization(net)
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
    def __init__(self, sess, checkpoint_dir, tensorboard_dir, in_height=28, in_width=28, nchannels=1,
                 batch_size=128, noise_dim=100, dataset_size=None,
                 data_name='mnist', mode='original', opt_pars=(0.00005, 0.9, 0.999),
                 pics_save_names=(None,None), d_iters=1, files_path=None):
        super().__init__(sess=sess, in_height=in_height, in_width=in_width, nchannels=nchannels,
                         batch_size=batch_size, noise_dim=noise_dim, dataset_size=dataset_size,
                         data_name=data_name, mode=mode,
                         checkpoint_dir=checkpoint_dir, tensorboard_dir=tensorboard_dir,
                         opt_pars=opt_pars, pics_save_names=pics_save_names,
                         d_iters=d_iters, files_path=files_path)
        super()._load_data()
        #self.filters = [64, 128, 256]
        self.filters = [32, 64, 128]
        #super()._build_model(self._generator, self._discriminator, 
        #                     lr=opt_pars[0], beta1=opt_pars[1], beta2=opt_pars[2])


    def _discriminator(self, x, drop):        
        def conv_block(nn, conv_channels, dim_stride, name1, name2):
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn)
            nn = tf.layers.conv2d(nn, filters=conv_channels, kernel_size=[3, 3], 
                                  strides=[dim_stride, dim_stride],
                                  activation=None, padding='same', 
                                  kernel_initializer=tf.initializers.orthogonal(),
                                  name=name1)
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn)
            nn = tf.layers.conv2d(nn, filters=conv_channels, kernel_size=[3, 3], 
                                  strides=[1,1],
                                  activation=None, padding='same', 
                                  name=name2)
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn)
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
            net = tf.layers.batch_normalization(net)
        self.d_layers_names += ('dense_output',)
        net = tf.layers.dense(net, 1, activation=None, name=self.d_layers_names[-1])
        return net, tf.nn.sigmoid(net, name='logit')

    
    def _generator(self, noise, drop):
        initial_shape = 4

        def conv_block(nn, conv_channels, dim_stride, name1, name2):
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn)
            nn = tf.layers.conv2d_transpose(nn, filters=conv_channels, kernel_size=[3,3], 
                                            strides=[dim_stride,dim_stride],
                                            kernel_initializer=tf.initializers.orthogonal(),
                                            activation=None, padding='same', name=name1)
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn)
            nn = tf.layers.conv2d_transpose(nn, filters=conv_channels, kernel_size=[3,3], 
                                            strides=[1,1],
                                            activation=None, padding='same', name=name2)
            nn = tf.nn.leaky_relu(nn, alpha=self.alpha)
            if self.mode != 'wgan-gp':
                nn = tf.layers.batch_normalization(nn)
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
            net = tf.layers.batch_normalization(net)
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
