import os
import numpy as np
import tensorflow as tf
from src.utilities import log_tf_files
from src.utilities import PlotGenSamples

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

        self._load_data()
        self._build_model()

    def _load_data():
        (train_data, train_labels), _ = tensorflow.keras.datasets.mnist.load_data()
        train_data = train_data / 255.
        dataset_size = len(train_data)
        dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
        self.dataset = dataset.shuffle(buffer_size=dataset_size).repeat(1).batch(self.batch_size)
        self.nbatches = int(np.ceil(dataset_size/self.batch_size))

    def _build_model():
        dropout_prob_ph = tf.placeholder(tf.float32, shape=(), 
                                         name='dropout_prob_ph')
        dropout_prob_D = 0.7
        dropout_prob_G = 0.3

        with tf.variable_scope('G'):
            gen_data_ph = tf.placeholder(tf.float32, shape=[None, noise_size], 
                                         name='gen_data_ph')
            G_sample = self._generator(gen_data_ph, prob=dropout_prob_ph)

        with tf.variable_scope('D') as scope:
            data_ph = tf.placeholder(tf.float32, shape=[None, 28, 28], 
                                     name='data_ph')   
            D_real_logits, D_real = self._discriminator(G_sample, prob=dropout_prob_ph)
        with tf.variable_scope('D', reuse=True):
            D_fake_logits, D_fake = self._discriminator(G_sample, prob=dropout_prob_ph)
        
        flip_prob = 0.0 #label flipping
        flip_arr = np.random.binomial(n=1, p=flip_prob, size=(nepochs, nbatches))
        minval = .85 #smoothing
        batch_size_ph = tf.placeholder(tf.int32, shape=[], 
                                       name='batch_size_ph') 
        real_labels_ph = tf.placeholder(tf.float32, 
                                        name='real_labels_ph')
        fake_labels_ph = tf.placeholder(tf.float32, 
                                        name='fake_labels_ph')
    
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits
        D_loss_real = cross_entropy(logits=D_real_logits, labels=real_labels_ph)
        D_loss_fake = cross_entropy(logits=D_fake_logits, labels=fake_labels_ph)
        D_loss = tf.reduce_mean(D_loss_real + D_loss_fake) 
        G_loss = tf.reduce_mean(cross_entropy(logits=D_fake_logits, labels=real_labels_ph)) 

        self.summaries()
    
        D_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
        G_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
        D_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D')
        G_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G')
    
        D_train_op = D_optimizer.minimize(D_loss, var_list=D_trainable_vars, name='D_train_op')
        G_train_op = G_optimizer.minimize(G_loss, var_list=G_trainable_vars, name='G_train_op')


    def _discriminator():
        filters = [16, 32, 64]
        alpha = 0.2
        
        def conv_block(nn, conv_channels, stride_last, name1, name2):
            nn = tf.layers.conv2d(nn, filters=conv_channels, kernel_size=[3, 3], 
                                  strides=[1, 1],
                                  activation=None, padding='same', 
                                  name=name1)
            nn = tf.layers.batch_normalization(nn)
            nn = tf.nn.leaky_relu(nn, alpha=alpha)
            nn = tf.layers.conv2d(nn, filters=conv_channels, kernel_size=[3, 3], 
                                  strides=[stride_last, stride_last],
                                  activation=None, padding='same', 
                                  name=name2)
            return tf.layers.batch_normalization(nn)

            
        def res_block(inputs, conv_channels, names):
            nn = conv_block(inputs, conv_channels=conv_channels,
                             stride_last=1,
                             name1=names[0], name2=names[1])
            nn = tf.add(nn, inputs)
            inputs = tf.nn.leaky_relu(nn, alpha=alpha)

            nn = conv_block(inputs, conv_channels=conv_channels,
                             stride_last=2,
                             name1=names[2], name2=names[3])

            inputs = tf.nn.avg_pool(inputs, ksize=[1,2,2,1], strides=[1,2,2,1], padding='valid')
            inputs = tf.pad(inputs, paddings=[[0,0],[0,0],[0,0],[conv_channels/2,conv_channels/2]], 
                            mode='CONSTANT', constant_values=0.)
            nn = tf.add(nn, inputs)
            return tf.nn.leaky_relu(nn, alpha=alpha)

        net = tf.reshape(x, shape=[-1, 28, 28, 1])
                
        #reshape (adding zeros) to the more convenient 32x32 shape
        inputs = tf.pad(net, paddings=[[0,0],[2,2],[2,2],[0,0]], mode='CONSTANT', constant_values=0.)

        names = ('layer1_1', 'layer1_2', 'layer1_3', 'layer1_4')
        inputs = res_block(inputs, conv_channels=filters[0], names=names)
        names = ('layer2_1', 'layer2_2', 'layer2_3', 'layer2_4')
        inputs = res_block(inputs, conv_channels=filters[1], names=names)
        names = ('layer3_1', 'layer3_2', 'layer3_3', 'layer3_4')
        inputs = res_block(inputs, conv_channels=filters[2], names=names)

        net = tf.reshape(net, shape=[-1,filters[2]*4*4])            
        net = tf.layers.dense(net, 1, activation=None, name='layer_dense')
        return net, tf.nn.sigmoid(net, name='discriminator_logit')


    def _generator():

        #copiar arquitectura anterior para verificar que o discriminador funciona

    def train(nepochs):
        summary = tf.summary.merge_all()
        saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)
        plot = PlotGenSamples()

        iterator = self.dataset.make_initializable_iterator()
        next_element = iterator.get_next()

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
                _, D_loss_c, D_real_c, D_fake_c = self.sess.run([D_train_op, D_loss, D_real, D_fake],
                              feed_dict={data_ph: inputs, gen_data_ph: noise_D,
                                         dropout_prob_ph: dropout_prob_D,
                                         batch_size_ph: len(inputs),
                                         real_labels_ph: real, fake_labels_ph: fake})
                
                #train generator
                noise_G = noise(len(inputs), self.noise_dim)
                _, G_loss_c  = self.sess.run([G_train_op, G_loss],
                                      feed_dict={data_ph: inputs, gen_data_ph: noise_G,
                                                 dropout_prob_ph: dropout_prob_G,
                                                 batch_size_ph: len(inputs),
                                                 real_labels_ph: real, fake_labels_ph: fake})

            summ = sess.run(summary, 
                            feed_dict={data_ph: inputs, gen_data_ph: noise_G,
                                       dropout_prob_ph: dropout_prob_G,
                                       batch_size_ph: len(inputs),
                                       real_labels_ph: real, fake_labels_ph: fake})
        
            train_writer.add_summary(summ, epoch*nbatches+(batch+1))
            saver.save(sess, checkpoint)

            #print generated samples
            sample = sess.run(G_sample, feed_dict={gen_data_ph: noise_G,
                                                   dropout_prob_ph: dropout_prob_G,
                                                   batch_size_ph: batch_size,
                                                   real_labels_ph: real, fake_labels_ph: fake})
            plot.plot_mnist(sample[:36], 'mnist_gen'+str(epoch))
            plot.plot_mnist(inputs[:36], 'mnist_data'+str(epoch))



    def _summaries(loss_d, loss_g):
        tf.summary.scalar('loss_d', loss_d)
        tf.summary.scalar('loss_g', loss_g)
    
        log_tf_files(num_layers=3, loss=loss_g, player='G')
        log_tf_files(num_layers=3, loss=loss_d, player='D')
