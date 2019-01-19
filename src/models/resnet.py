import os
import numpy as np
import tensorflow as tf
from src.utilities import log_tf_files
from src.utilities import PlotGenSamples
from src.models.architecture import minibatch_discrimination

def noise(m,n):
    return np.random.normal(loc=0.0, scale=1., size=[m,n])

class ResNetGAN():
    def __init__(self, sess, input_height=28, input_width=28, nchannels=1,
                 batch_size=128, noise_dim=100, 
                 checkpoint_dir='/fred/oz012/Bruno/checkpoints/0/'):
        self.sess = sess
        self.input_height = input_height
        self.input_width = input_width
        self.nchannels = nchannels
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.checkpoint_dir = checkpoint_dir

        def _load_data():
            (train_data, train_labels), _ = tf.keras.datasets.mnist.load_data()
            train_data = train_data / 255.
            dataset_size = len(train_data)
            dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
            self.dataset = dataset.shuffle(buffer_size=dataset_size).repeat(1).batch(self.batch_size)
            self.nbatches = int(np.ceil(dataset_size/self.batch_size))

        def _build_model():
            self.dropout_prob_ph = tf.placeholder(tf.float32, shape=(), 
                                             name='dropout_prob_ph')
            with tf.variable_scope('G'):
                self.gen_data_ph = tf.placeholder(tf.float32, shape=[None, self.noise_dim], 
                                             name='gen_data_ph')
                self.G_sample = self._generator(self.gen_data_ph, drop=self.dropout_prob_ph)

            with tf.variable_scope('D') as scope:
                self.data_ph = tf.placeholder(tf.float32, shape=[None, 28, 28], 
                                         name='data_ph')   
                self.D_real_logits, self.D_real = self._discriminator(self.G_sample, 
                                                                      drop=self.dropout_prob_ph)
            with tf.variable_scope('D', reuse=True):
                self.D_fake_logits, self.D_fake = self._discriminator(self.G_sample, 
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
            
            D_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
            G_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
            D_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D')
            G_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G')
            
            self.D_train_op = D_optimizer.minimize(self.D_loss, var_list=D_trainable_vars, 
                                                   name='D_train_op')
            self.G_train_op = G_optimizer.minimize(self.G_loss, var_list=G_trainable_vars, 
                                                   name='G_train_op')

        _load_data()
        _build_model()

    def _discriminator(self, x, drop):
        filters = [64, 128, 256]
        alpha = 0.2
        
        def conv_block(nn, conv_channels, dim_stride, name1, name2):
            nn = tf.layers.conv2d(nn, filters=conv_channels, kernel_size=[3, 3], 
                                  strides=[dim_stride, dim_stride],
                                  activation=None, padding='same', 
                                  name=name1)
            nn = tf.layers.batch_normalization(nn)
            nn = tf.nn.leaky_relu(nn, alpha=alpha)
            nn = tf.layers.conv2d(nn, filters=conv_channels, kernel_size=[3, 3], 
                                  strides=[1,1],
                                  activation=None, padding='same', 
                                  name=name2)
            return tf.layers.batch_normalization(nn)

            
        def double_res_block(inputs, conv_channels, names):
            if conv_channels == filters[0]: dim_stride = 1
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
            inputs = tf.nn.leaky_relu(nn, alpha=alpha)

            nn = conv_block(inputs, conv_channels=conv_channels,
                             dim_stride=1,
                             name1=names[2], name2=names[3])
            nn = tf.add(nn, inputs)
            return tf.nn.leaky_relu(nn, alpha=alpha)

        net = tf.reshape(x, shape=[-1, 28, 28, 1])
                
        #reshape (adding zeros) to the more convenient 32x32 shape
        inputs = tf.pad(net, paddings=[[0,0],[2,2],[2,2],[0,0]], mode='CONSTANT', constant_values=0.)
        inputs = tf.layers.conv2d(inputs, filters=filters[0], kernel_size=[7, 7], 
                                  strides=[1, 1],
                                  activation=None, padding='same', 
                                  name='first_conv_layer')

        names = ('layer1_1', 'layer1_2', 'layer1_3', 'layer1_4')
        inputs = double_res_block(inputs, conv_channels=filters[0], names=names)
        names = ('layer2_1', 'layer2_2', 'layer2_3', 'layer2_4')
        inputs = double_res_block(inputs, conv_channels=filters[1], names=names)
        names = ('layer3_1', 'layer3_2', 'layer3_3', 'layer3_4')
        net = double_res_block(inputs, conv_channels=filters[2], names=names)

        final_shape = net.get_shape()
        assert final_shape[3] == filters[2]
        net = tf.reshape(net, shape=[-1,filters[2]*final_shape[1]*final_shape[2]])
        net = minibatch_discrimination(net, num_kernels=20, kernel_dim=10)
        net = tf.layers.dense(net, 1, activation=None, 
                              name='layer_dense')
        return net, tf.nn.sigmoid(net, name='discriminator_logit')


    def _generator(self, noise, drop):
        filters = [256, 128, 64]
        initial_shape = 8
        final_shape = 32

        def conv_block(nn, conv_channels, dim_stride, name1, name2):
            nn = tf.layers.conv2d_transpose(nn, filters=conv_channels, kernel_size=[3,3], 
                                            strides=[dim_stride,dim_stride],
                                            activation=None, padding='same', name=name1)
            nn = tf.layers.batch_normalization(nn)
            nn = tf.nn.relu(nn)
            nn = tf.layers.conv2d_transpose(nn, filters=conv_channels, kernel_size=[3,3], 
                                            strides=[1,1],
                                            activation=None, padding='same', name=name2)
            return tf.layers.batch_normalization(nn)

            
        def double_res_block(inputs, conv_channels, pad_input, names):
            if conv_channels == filters[0]: dim_stride = 1
            else: dim_stride = 2

            nn = conv_block(inputs, conv_channels=conv_channels,
                            dim_stride=dim_stride,
                            name1=names[0], name2=names[1])

            if dim_stride == 2:
                inputs = tf.pad(inputs, 
                                paddings=[[0,0], 
                                          [int(pad_input/2),int(pad_input/2)], 
                                          [int(pad_input/2),int(pad_input/2)], 
                                          [0,0]], 
                                mode='CONSTANT', constant_values=0.)
                inputs = tf.transpose(inputs, [0,1,3,2])
                inputs = tf.nn.avg_pool(inputs, ksize=[1,1,2,1], strides=[1,1,2,1], padding='VALID')
                inputs = tf.transpose(inputs, [0,1,3,2])
            nn = tf.add(nn, inputs)
            inputs = tf.nn.relu(nn)

            nn = conv_block(inputs, conv_channels=conv_channels,
                             dim_stride=1,
                             name1=names[2], name2=names[3])
            nn = tf.add(nn, inputs)
            return tf.nn.relu(nn)
        
        net = tf.reshape(noise, shape=[-1, noise.shape[1]])
        inputs = tf.layers.dense(noise, initial_shape*initial_shape, activation=None)
        inputs = tf.reshape(inputs, shape=[-1,initial_shape,initial_shape,1])
        names = ('layer1_1', 'layer1_2', 'layer1_3', 'layer1_4')
        inputs = double_res_block(inputs, conv_channels=filters[0], pad_input=initial_shape/2, 
                                  names=names)
        names = ('layer2_1', 'layer2_2', 'layer2_3', 'layer2_4')
        inputs = double_res_block(inputs, conv_channels=filters[1], pad_input=initial_shape, 
                                  names=names)
        names = ('layer3_1', 'layer3_2', 'layer3_3', 'layer3_4')
        net = double_res_block(inputs, conv_channels=filters[2], pad_input=initial_shape*2, 
                               names=names)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.relu(net)
        net = tf.layers.conv2d_transpose(net, filters=1, kernel_size=[7,7], 
                                         strides=[1,1],
                                         activation=None, padding='same', name='last_conv_layer')
        #net = tf.layers.dense(net, filters[2]*final_shape*final_shape, activation=None)
        #crop the outer parts of the images to retrieve the original 28x28 size
        net = tf.slice(net, begin=[0,2,2,0], size=[-1,28,28,-1])
        net = tf.nn.tanh(net)
        return tf.reshape(net, shape=[-1, 28, 28], name='generator_output')


    def train(self, nepochs):
        summary = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
        plot = PlotGenSamples()

        iterator = self.dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        flip_prob = 0.0 #label flipping
        flip_arr = np.random.binomial(n=1, p=flip_prob, size=(nepochs, self.nbatches))
        minval = .85 #smoothing

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
            plot.plot_mnist(sample[:36], 'mnist_gen'+str(epoch))
            plot.plot_mnist(inputs[:36], 'mnist_data'+str(epoch))



    def _summaries(self, loss_d, loss_g):
        tf.summary.scalar('loss_d', loss_d)
        tf.summary.scalar('loss_g', loss_g)
    
        #log_tf_files(num_layers=3, loss=loss_g, player='G')
        #log_tf_files(num_layers=3, loss=loss_d, player='D')
