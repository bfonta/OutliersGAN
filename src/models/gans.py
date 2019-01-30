import os
import numpy as np
import tensorflow as tf
from src.utilities import log_tf_files
from src.utilities import PlotGenSamples
from src.decorators import type_check
from src.models.utilities import minibatch_discrimination, noise
    
class _GAN():
    """
    Base class for all the implemented GAN-like models.
    """
    def __init__(self, sess, in_height=28, in_width=28, nchannels=1,
                 batch_size=128, noise_dim=100, 
                 checkpoint_dir='/fred/oz012/Bruno/checkpoints/0/'):
            self.sess = sess
            self.in_height = in_height
            self.in_width = in_width
            self.nchannels = nchannels
            self.batch_size = batch_size
            self.noise_dim = noise_dim
            self.checkpoint_dir = checkpoint_dir
            
    @type_check
    def _load_data(self, data_name):
        if data_name == 'mnist':
            (train_data, train_labels), _ = tf.keras.datasets.mnist.load_data()
        elif data_name == 'fashion_mnist':
            (train_data, train_labels), _ = tf.keras.datasets.fashion_mnist.load_data()
        elif data_name == 'cifar10':
            (train_data, train_labels), _ = tf.keras.datasets.cifar10.load_data()
        train_data = train_data / 255.
        dataset_size = len(train_data)
        dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        self.dataset = dataset.shuffle(buffer_size=dataset_size).repeat(1).batch(self.batch_size)
        self.nbatches = int(np.ceil(dataset_size/self.batch_size))

    @type_check
    def _build_model(self, generator, discriminator, lr=0.0002, beta1=0.5):
        self.dropout_prob_ph = tf.placeholder(tf.float32, shape=(), 
                                              name='dropout_prob_ph')
        
        with tf.variable_scope('G'):
            self.gen_data_ph = tf.placeholder(tf.float32, shape=[None, self.noise_dim], 
                                              name='gen_data_ph')
            self.G_sample = generator(self.gen_data_ph, drop=self.dropout_prob_ph)

        with tf.variable_scope('D') as scope:
            self.data_ph = tf.placeholder(tf.float32, 
                                          shape=[None, self.in_height, self.in_width], 
                                          name='data_ph')   
            self.D_real_logits, self.D_real = discriminator(self.data_ph, 
                                                            drop=self.dropout_prob_ph)

        with tf.variable_scope('D', reuse=True):
            self.D_fake_logits, self.D_fake = discriminator(self.G_sample, 
                                                            drop=self.dropout_prob_ph)

        self.batch_size_ph = tf.placeholder(tf.int32, shape=[], 
                                            name='batch_size_ph') 
        self.real_labels_ph = tf.placeholder(tf.float32, 
                                             name='real_labels_ph')
        self.fake_labels_ph = tf.placeholder(tf.float32, 
                                             name='fake_labels_ph')

        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits
        D_loss_real = cross_entropy(logits=self.D_real_logits, labels=self.real_labels_ph)
        D_loss_fake = cross_entropy(logits=self.D_fake_logits, labels=self.fake_labels_ph)
        self.D_loss = tf.reduce_mean(D_loss_real + D_loss_fake) 
        self.G_loss = tf.reduce_mean(cross_entropy(logits=self.D_fake_logits,
                                                   labels=self.real_labels_ph)) 
            
        self._summaries(self.D_loss, self.G_loss)
            
        D_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
        G_optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1)
        D_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D')
        G_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G')
        
        self.D_train_op = D_optimizer.minimize(self.D_loss, var_list=D_trainable_vars, 
                                               name='D_train_op')
        self.G_train_op = G_optimizer.minimize(self.G_loss, var_list=G_trainable_vars, 
                                               name='G_train_op')

    @type_check
    def train(self, nepochs):
        summary = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

        iterator = self.dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        flip_prob = 0.0 #label flipping
        flip_arr = np.random.binomial(n=1, p=flip_prob, size=(nepochs, self.nbatches))
        minval = 0.85 #smoothing

        dropout_prob_D = 0.7
        dropout_prob_G = 0.3

        init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        self.sess.run(init)
        train_writer = tf.summary.FileWriter('tensorboard/'+str(np.random.randint(0,99999)), 
                                             self.sess.graph)

        for epoch in range(nepochs):
            print("Epoch: {}".format(epoch), flush=True)
            self.sess.run(iterator.initializer)

            for batch in range(self.nbatches):
                inputs, _ = self.sess.run(next_element)

                #label flipping
                if  flip_arr[epoch][batch] == 1:
                    real = np.zeros(shape=(len(inputs),1))
                    fake = np.full(shape=(len(inputs),1), 
                                   fill_value=np.random.uniform(low=minval, high=1.))
                else:
                    real = np.full(shape=(len(inputs),1), 
                                   fill_value=np.random.uniform(low=minval, high=1.))
                    fake = np.zeros(shape=(len(inputs),1))

                #train discriminator
                noise_D = noise(len(inputs), self.noise_dim)
                _, D_loss_c, D_real_c, D_fake_c = self.sess.run([self.D_train_op, self.D_loss,
                                                                 self.D_real, self.D_fake],
                              feed_dict={self.data_ph: inputs, self.gen_data_ph: noise_D,
                                         self.dropout_prob_ph: dropout_prob_D,
                                         self.batch_size_ph: len(inputs),
                                         self.real_labels_ph: real, self.fake_labels_ph: fake})

                #train generator
                noise_G = noise(len(inputs), self.noise_dim)
                _, G_loss_c  = self.sess.run([self.G_train_op, self.G_loss],
                                      feed_dict={self.data_ph: inputs, self.gen_data_ph: noise_G,
                                                 self.dropout_prob_ph: dropout_prob_G,
                                                 self.batch_size_ph: len(inputs),
                                                 self.real_labels_ph: real, self.fake_labels_ph: fake})

            summ = self.sess.run(summary, 
                                 feed_dict={self.data_ph: inputs, self.gen_data_ph: noise_G,
                                            self.dropout_prob_ph: dropout_prob_G,
                                            self.batch_size_ph: len(inputs),
                                            self.real_labels_ph: real, self.fake_labels_ph: fake})
        
            train_writer.add_summary(summ, epoch*self.nbatches+(batch+1))
            saver.save(self.sess, self.checkpoint_dir)

            #print generated samples
            sample = self.sess.run(self.G_sample, feed_dict={self.gen_data_ph: noise_G,
                                                   self.dropout_prob_ph: dropout_prob_G,
                                                   self.batch_size_ph: self.batch_size,
                                                   self.real_labels_ph: real, self.fake_labels_ph: fake})
            self._plot(inputs, sample, 
                       ['mnist_data'+str(epoch), 'mnist_gen'+str(epoch)])

    def _plot(self, real_data, gen_data, names):
        p = PlotGenSamples()
        p.plot_mnist(real_data[:36], names[0])
        p.plot_mnist(gen_data[:36], names[1])

    def _summaries(self, loss_d, loss_g):
        tf.summary.scalar('loss_d', loss_d)
        tf.summary.scalar('loss_g', loss_g)    
        #log_tf_files(num_layers=3, loss=loss_g, player='G')
        #log_tf_files(num_layers=3, loss=loss_d, player='D')


##################################################################################################
class DCGAN(_GAN):
    """
    Generative Adversarial Network using a Deep Convolutional architecture
    Reference: Redford et al; ArXiv: 1511.06434
    """
    def __init__(self, sess, in_height=28, in_width=28, nchannels=1,
                 batch_size=128, noise_dim=100, 
                 checkpoint_dir='/fred/oz012/Bruno/checkpoints/0/',
                 dataset='mnist'):
        super().__init__(sess=sess, in_height=in_height, in_width=in_width, nchannels=nchannels,
                         batch_size=batch_size, noise_dim=noise_dim, 
                         checkpoint_dir=checkpoint_dir)
        
        super()._load_data(dataset)

        self.filters = [128, 256, 512]
        super()._build_model(self._generator, self._discriminator)

    def _discriminator(self, x, drop):
        def conv_block(nn, conv_channels, name):
            alpha = 0.2
            nn = tf.layers.conv2d(nn, filters=conv_channels, kernel_size=[5, 5], 
                                  strides=[2,2], activation=None, padding='same', 
                                  name=name)
            nn = tf.layers.batch_normalization(nn)
            
            nn = tf.nn.leaky_relu(nn, alpha=alpha)
            nn = tf.nn.relu(nn)

            nn = tf.layers.dropout(nn, rate=drop)
            return nn

        """
        #DENSE CHECK
        net = tf.reshape(x, shape=[-1,28*28])
        net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
        net = tf.layers.dense(net, 1, activation=None)
        """

        net = tf.reshape(x, shape=[-1, self.in_height, self.in_width, self.nchannels])

        """
        if self.in_height==28 and self.in_width==28:
            #add zeros for the more convenient 32x32 shape
            net = tf.pad(net, paddings=[[0,0],[2,2],[2,2],[0,0]], mode='CONSTANT', constant_values=0.)
        """
        net = conv_block(net, conv_channels=self.filters[0], name='layer1')
        net = conv_block(net, conv_channels=self.filters[1], name='layer2')
        net = conv_block(net, conv_channels=self.filters[2], name='layer3')

        final_shape = net.get_shape()
        net = tf.reshape(net, shape=[-1,self.filters[2]*final_shape[1]*final_shape[2]])
        net = minibatch_discrimination(net, num_kernels=20, kernel_dim=10)
        net = tf.layers.dense(net, 1, activation=None, 
                              name='layer_dense')

        return net, tf.nn.sigmoid(net, name='logit')
    
    def _generator(self, noise, drop):
        init_height, init_width = 4, 4

        def conv_block(nn, conv_channels, name):
            nn = tf.layers.conv2d_transpose(nn, filters=conv_channels, kernel_size=[5,5], 
                                            strides=[2,2],
                                            activation=None, padding='same', name=name)
            nn = tf.layers.batch_normalization(nn)
            nn = tf.nn.relu(nn)
            nn = tf.layers.dropout(nn, rate=0.2)
            return nn
        
        net = tf.reshape(noise, shape=[-1, noise.shape[1]])
        """
        DENSE CHECK
        net = tf.layers.dense(net, 1024, activation=tf.nn.relu)
        net = tf.layers.dense(net, 28*28, activation=tf.nn.tanh)
        """   

        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)

        net = tf.layers.dense(net, init_height*init_width*self.filters[-1], activation=None)
        net = tf.layers.dropout(net, rate=drop)
        net = tf.reshape(net, shape=[-1,init_height,init_width,self.filters[-1]])

        net = conv_block(net, conv_channels=self.filters[-2], name='layer1')
        net = conv_block(net, conv_channels=self.filters[-3], name='layer2')

        net = tf.layers.conv2d_transpose(net, filters=1, kernel_size=[5,5], 
                                         strides=[2,2], activation=None, padding='same', 
                                         name='layer3')

        if self.in_height==28 and self.in_width==28:
            net = tf.slice(net, begin=[0,2,2,0], size=[-1,self.in_width,self.in_height,-1])

        net = tf.nn.tanh(net)
        net = tf.layers.dropout(net,rate=0.2)

        return tf.reshape(net, shape=[-1, self.in_width, self.in_height], 
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
    def __init__(self, sess, in_height=28, in_width=28, nchannels=1,
                 batch_size=128, noise_dim=100, 
                 checkpoint_dir='/fred/oz012/Bruno/checkpoints/0/'):
        super().__init__(sess=sess, in_height=in_height, in_width=in_width, nchannels=nchannels,
                 batch_size=batch_size, noise_dim=noise_dim, 
                 checkpoint_dir='/fred/oz012/Bruno/checkpoints/0/')
    
        super()._load_data('mnist')
        self.filters = [64, 128, 256]
        super()._build_model(self._generator, self._discriminator, lr=0.0002, beta1=0.9)


    def _discriminator(self, x, drop):        
        def conv_block(nn, conv_channels, dim_stride, name1, name2):
            nn = tf.nn.relu(nn)
            nn = tf.layers.batch_normalization(nn)
            nn = tf.layers.conv2d(nn, filters=conv_channels, kernel_size=[3, 3], 
                                  strides=[dim_stride, dim_stride],
                                  activation=None, padding='same', 
                                  name=name1)
            nn = tf.nn.relu(nn)
            nn = tf.layers.batch_normalization(nn)
            nn = tf.layers.conv2d(nn, filters=conv_channels, kernel_size=[3, 3], 
                                  strides=[1,1],
                                  activation=None, padding='same', 
                                  name=name2)
            nn = tf.nn.relu(nn)
            return tf.layers.batch_normalization(nn)

            
        def double_res_block(inputs, conv_channels, names):
            if conv_channels == self.filters[0]: dim_stride = 1
            else: dim_stride = 2

            nn = conv_block(inputs, conv_channels=conv_channels,
                             dim_stride=dim_stride,
                             name1=names[0], name2=names[1])

            if dim_stride == 2:
                inputs = tf.nn.avg_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
                chn_pad = int(conv_channels/2)
                inputs = tf.pad(inputs, paddings=[[0,0],[0,0],[0,0],[int(chn_pad/2),int(chn_pad/2)]], 
                                mode='CONSTANT', constant_values=0.)

            nn = tf.add(nn, inputs)
            nn = conv_block(inputs, conv_channels=conv_channels,
                             dim_stride=1,
                             name1=names[2], name2=names[3])
            nn = tf.add(nn, inputs)
            return nn

        net = tf.reshape(x, shape=[-1, self.in_height, self.in_width, 1])

        if self.in_height==28 and self.in_width==28:
            #reshape (adding zeros) to the more convenient 32x32 shape
            inputs = tf.pad(net, paddings=[[0,0],[2,2],[2,2],[0,0]], mode='CONSTANT', constant_values=0.)

        inputs = tf.layers.conv2d(inputs, filters=self.filters[0], kernel_size=[7, 7], 
                                  strides=[1, 1],
                                  activation=None, padding='same', 
                                  name='first_conv_layer')

        names = ('layer1_1', 'layer1_2', 'layer1_3', 'layer1_4')
        inputs = double_res_block(inputs, conv_channels=self.filters[0], names=names)
        names = ('layer2_1', 'layer2_2', 'layer2_3', 'layer2_4')
        inputs = double_res_block(inputs, conv_channels=self.filters[1], names=names)
        names = ('layer3_1', 'layer3_2', 'layer3_3', 'layer3_4')
        net = double_res_block(inputs, conv_channels=self.filters[2], names=names)

        final_shape = net.get_shape()
        assert final_shape[3] == self.filters[2]
        net = tf.reshape(net, shape=[-1,self.filters[2]*final_shape[1]*final_shape[2]])
        net = minibatch_discrimination(net, num_kernels=20, kernel_dim=10)
        net = tf.nn.relu(net)
        net = tf.layers.batch_normalization(net)
        net = tf.layers.dense(net, 1, activation=None, 
                              name='layer_dense')
        return net, tf.nn.sigmoid(net, name='logit')

    
    def _generator(self, noise, drop):
        initial_shape = 8
        final_shape = 32

        def conv_block(nn, conv_channels, dim_stride, name1, name2):
            nn = tf.nn.relu(nn)
            nn = tf.layers.batch_normalization(nn)
            nn = tf.layers.conv2d_transpose(nn, filters=conv_channels, kernel_size=[3,3], 
                                            strides=[dim_stride,dim_stride],
                                            activation=None, padding='same', name=name1)
            nn = tf.nn.relu(nn)
            nn = tf.layers.batch_normalization(nn)
            nn = tf.layers.conv2d_transpose(nn, filters=conv_channels, kernel_size=[3,3], 
                                            strides=[1,1],
                                            activation=None, padding='same', name=name2)
            nn = tf.nn.relu(nn)
            return tf.layers.batch_normalization(nn)

            
        def double_res_block(inputs, conv_channels, pad_input, names):
            if conv_channels == self.filters[-1]: dim_stride = 1
            else: dim_stride = 2

            nn = conv_block(inputs, conv_channels=conv_channels,
                            dim_stride=dim_stride,
                            name1=names[0], name2=names[1])

            if dim_stride == 2:
                inputs = tf.layers.conv2d_transpose(inputs, filters=int(int(inputs.shape[3])/2), 
                                                    kernel_size=[3,3], 
                                                    strides=[dim_stride,dim_stride],
                                                    activation=None, padding='same', 
                                                    name='input_conv'+names[2])
                """
                inputs = tf.pad(inputs, 
                                paddings=[[0,0], 
                                          [int(pad_input/2),int(pad_input/2)], 
                                          [int(pad_input/2),int(pad_input/2)], 
                                          [0,0]], 
                                mode='CONSTANT', constant_values=0.)
                inputs = tf.transpose(inputs, [0,1,3,2])
                inputs = tf.nn.avg_pool(inputs, ksize=[1,1,2,1], strides=[1,1,2,1], padding='VALID')
                inputs = tf.transpose(inputs, [0,1,3,2])
                """
            nn = tf.add(nn, inputs)
            nn = conv_block(inputs, conv_channels=conv_channels,
                             dim_stride=1,
                             name1=names[2], name2=names[3])
            return tf.add(nn, inputs)
        
        net = tf.reshape(noise, shape=[-1, noise.shape[1]])
        inputs = tf.layers.dense(noise, initial_shape*initial_shape, activation=None)
        inputs = tf.reshape(inputs, shape=[-1,initial_shape,initial_shape,1])

        names = ('layer1_1', 'layer1_2', 'layer1_3', 'layer1_4')
        inputs = double_res_block(inputs, conv_channels=self.filters[-1], pad_input=initial_shape/2, 
                                  names=names)
        names = ('layer2_1', 'layer2_2', 'layer2_3', 'layer2_4')
        inputs = double_res_block(inputs, conv_channels=self.filters[-2], pad_input=initial_shape, 
                                  names=names)
        names = ('layer3_1', 'layer3_2', 'layer3_3', 'layer3_4')
        net = double_res_block(inputs, conv_channels=self.filters[-3], pad_input=initial_shape*2, 
                               names=names)

        net = tf.nn.relu(net)
        net = tf.layers.batch_normalization(net)
        net = tf.layers.conv2d_transpose(net, filters=1, kernel_size=[7,7], 
                                         strides=[1,1],
                                         activation=None, padding='same', name='last_convolution')
        net = tf.nn.relu(net)
        net = tf.layers.batch_normalization(net)

        if self.in_height==28 and self.in_width==28:
            #crop the outer parts of the images to retrieve the original 28x28 size
            net = tf.slice(net, begin=[0,2,2,0], size=[-1,self.in_height,self.in_width,-1])

        net = tf.nn.tanh(net)
        return tf.reshape(net, shape=[-1, self.in_height, self.in_width], name='output')
