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

    def generate(self, N, n_per_plot, name, fits=False):
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
            gen_samples = self.sess.run(G_output, 
                                        feed_dict={gen_data_ph: noise(n_per_plot,self.noise_dim),
                                                   dropout_prob_ph: 0., 
                                                   batch_size_ph: n_per_plot})
            self._plot(gen_samples, params, name+str(i), n=n_per_plot)
            if fits:
                for j in range(n_per_plot):
                    to_fits(params[0][j], gen_samples[j].reshape(self.in_height), 
                            'wavelength', 'flux', 'data_'+str(i)+'_'+str(j)+'.fits')

    def save_features(self, ninputs, save_path, 
                      additional_data_name=None, additional_data_path=None, 
                      additional_files_name=None, additional_ninputs=None):
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
        ->additional_data_name: 
           name of a dataset. This allows the user to save the features for more than one dataset.
           In order to see the discriminator response, it is useful to be able to run data on a model
           that was trained using different data.
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
        D_feats = tf.get_default_graph().get_tensor_by_name('D/layer4/BiasAdd:0')
        D_feats = tf.reshape(D_feats, shape=[-1,7*1*1024])
        D_noise = noise(ninputs, self.noise_dim)
            
        data_ph = tf.get_default_graph().get_tensor_by_name('D/data_ph:0')
        dropout_prob_ph = tf.get_default_graph().get_tensor_by_name('dropout_prob_ph:0')
        batch_size_ph = tf.get_default_graph().get_tensor_by_name('batch_size_ph:0')
        batch_norm_ph = tf.get_default_graph().get_tensor_by_name('batch_norm_ph:0')

        iterator = self.dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        self.sess.run(iterator.initializer)

        if additional_ninputs == None:
            additional_ninputs = self.dataset_size
        if additional_data_name != None:
            if additional_files_name == None:
                additional_files_name = self.files_name
            dataset2, nbatches2 = self._load_data(ret=True, 
                                                  data_name=additional_data_name,
                                                  data_path=additional_data_path,
                                                  files_name=additional_files_name,
                                                  data_size=additional_ninputs)
            iterator2 = dataset2.make_initializable_iterator()
            next_element2 = iterator2.get_next()
            self.sess.run(iterator2.initializer)

        saver.restore(self.sess, latest_checkpoint)

        save_start = 0
        M = D_feats.get_shape()[1]
        
        with h5py.File(save_path+'.hdf5', 'w') as f:
            
            group = f.create_group('data')
            dset = group.create_dataset(self.data_name, (ninputs, M), dtype=np.float32) 

            for b in range(self.nbatches):
                inputs, *params = self.sess.run(next_element)
                inputs = self._pre_process(inputs, params)

                #Makes sure that the number of input data is the one the user wants
                if (b+1)*self.batch_size > ninputs:
                    inputs = inputs[:ninputs - b*self.batch_size]

                feats = self.sess.run(D_feats,
                                      feed_dict={data_ph: inputs, #gen_data_ph: D_noise,
                                                 dropout_prob_ph: 0.,
                                                 batch_norm_ph: False})
                dset[save_start:save_start+len(feats), :] = feats
                save_start += len(feats)

                #Stopping criterion
                if len(inputs) != self.batch_size:
                    break

            save_start = 0
            count=0
            if additional_data_name != None:
                dset = group.create_dataset(additional_data_name+'_additional', 
                                            (additional_ninputs, M), dtype=np.float32) 
                for b in range(nbatches2):
                    try:
                        inputs, *params = self.sess.run(next_element2)
                        inputs = self._pre_process(inputs, params)
                    
                        #Makes sure that the number of input data is the one the user wants
                        if (b+1)*self.batch_size > additional_ninputs:
                            inputs = inputs[:additional_ninputs - b*self.batch_size]

                        count += len(inputs)
                        feats = self.sess.run(D_feats,
                                              feed_dict={data_ph: inputs, #gen_data_ph: D_noise,
                                                         dropout_prob_ph: 0.,
                                                         batch_norm_ph: False})
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
    def __init__(self, sess, checkpoint_dir, tensorboard_dir, in_height=28, in_width=28, nchannels=1,
                 batch_size=128, noise_dim=100, dataset_size=None,
                 data_name='mnist', mode='original', opt_pars=(0.0001, 0.5, 0.999), 
                 pics_save_names=(None,None), d_iters=1, files_path=None, files_name=None):
        super().__init__(sess=sess, in_height=in_height, in_width=in_width, nchannels=nchannels,
                         batch_size=batch_size, noise_dim=noise_dim, dataset_size=dataset_size, 
                         checkpoint_dir=checkpoint_dir, tensorboard_dir=tensorboard_dir,
                         mode=mode, opt_pars=opt_pars, pics_save_names=pics_save_names,
                         d_iters=d_iters, data_name=data_name, files_path=files_path, 
                         files_name=files_name)
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
    def __init__(self, sesss, checkpoint_dir, tensorboard_dir, in_height=28, in_width=28, nchannels=1,
                 batch_size=128, noise_dim=100, dataset_size=None,
                 data_name='mnist', mode='original', opt_pars=(0.00005, 0.9, 0.999),
                 pics_save_names=(None,None), d_iters=1, files_path=None, files_name=None):
        super().__init__(sess=sess, in_height=in_height, in_width=in_width, nchannels=nchannels,
                         batch_size=batch_size, noise_dim=noise_dim, dataset_size=dataset_size,
                         data_name=data_name, mode=mode,
                         checkpoint_dir=checkpoint_dir, tensorboard_dir=tensorboard_dir,
                         opt_pars=opt_pars, pics_save_names=pics_save_names,
                         d_iters=d_iters, files_path=files_path, files_name=files_name)
        super()._load_data()
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
