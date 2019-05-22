import numpy as np
import tensorflow as tf
from tensorflow import layers

def minibatch_discrimination(inputs, num_kernels=5, kernel_dim=3):
    with tf.variable_scope('minibatch_discrimination'):
        T = tf.get_variable('T', shape=[inputs.get_shape()[1], num_kernels*kernel_dim], 
                            initializer=tf.random_normal_initializer(stddev=0.02))
        M = tf.reshape(tf.matmul(inputs,T), (-1,num_kernels,kernel_dim))
        diffs = tf.expand_dims(M, 3) - tf.expand_dims(tf.transpose(M, [1,2,0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([inputs, minibatch_features], 1)

def tikhonov_regularizer(D_real_logits, D_real_arg, D_fake_logits, D_fake_arg, batch_size):
    D1 = tf.nn.sigmoid(D_real_logits)
    D2 = tf.nn.sigmoid(D_fake_logits)
    grad_D1_logits = tf.gradients(D_real_logits, D_real_arg)[0]
    grad_D2_logits = tf.gradients(D_fake_logits, D_fake_arg)[0]
    grad_D1_logits_norm = tf.norm(tf.reshape(grad_D1_logits, [batch_size,-1]), axis=1, keepdims=True)
    grad_D2_logits_norm = tf.norm(tf.reshape(grad_D2_logits, [batch_size,-1]), axis=1, keepdims=True)

    reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
    reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
    disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
    return disc_regularizer

#try tf.orthogonal_initializer
def dcgan_discriminator_spectra(x, prob=0.):
    filters = [128, 256, 512, 1024]
    alpha = 0.2

    net = tf.reshape(x, shape=[-1, x.shape[1], 1, 1])
    net = tf.layers.conv2d(net, filters=filters[0], kernel_size=[10,1], strides=[5,1],
                   activation=None, padding='same', name='layer1')
    net = tf.nn.leaky_relu(net, alpha=alpha)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net,rate=prob)

    net = tf.layers.conv2d(net, filters=filters[1], kernel_size=[10,1], strides=[5,1],
                   activation=None, padding='same', name='layer2')
    net = tf.nn.leaky_relu(net, alpha=alpha)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net,rate=prob)

    net = tf.layers.conv2d(net, filters=filters[2], kernel_size=[10,1], strides=[5,1],
                   activation=None, padding='same', name='layer3')
    net = tf.nn.leaky_relu(net, alpha=alpha)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net,rate=prob)

    net = tf.layers.conv2d(net, filters=filters[3], kernel_size=[10,1], strides=[4,1],
                   activation=None, padding='same', name='layer4')
    net = tf.nn.leaky_relu(net, alpha=alpha)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net,rate=prob)

    net = tf.reshape(net, shape=[-1, filters[3]*7*1])
    #net = tf.layers.dense(net, 512, activation=None, name='layer5')
    net = tf.nn.leaky_relu(net, alpha=alpha)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net,rate=prob)
    
    net = minibatch_discrimination(net, num_kernels=30, kernel_dim=20)

    net = tf.layers.dense(net, 1, activation=None, name='layer6')    

    return net, tf.nn.sigmoid(net, name='discriminator_logit')

def dcgan_generator_spectra(noise, data_size, prob=0.):
    filters = [1024, 512, 256, 128]

    net = tf.reshape(noise, shape=[-1, noise.shape[1]])
    net = tf.layers.dense(net, filters[0]*7, activation=None)
    #net = tf.layers.dropout(net,rate=0.5)
    net = tf.reshape(net, shape=[-1,7,1,filters[0]])

    net = tf.layers.conv2d_transpose(net, filters=filters[1], kernel_size=[10,1], strides=[4,1],
                             activation=None, padding='same', name='layer1')
    net = tf.nn.relu(net)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net,rate=prob)

    net = tf.layers.conv2d_transpose(net, filters=filters[2], kernel_size=[10,1], strides=[5,1],
                             activation=None, padding='same', name='layer2')
    net = tf.nn.relu(net)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net,rate=prob)

    net = tf.layers.conv2d_transpose(net, filters=filters[3], kernel_size=[10,1], strides=[5,1],
                             activation=None, padding='same', name='layer3')
    net = tf.nn.relu(net)
    net = tf.layers.batch_normalization(net)
    net = tf.layers.dropout(net,rate=prob)

    net = tf.layers.conv2d_transpose(net, filters=1, kernel_size=[10,1], strides=[5,1],
                             activation=None, padding='same', name='layer4')
    net = tf.layers.batch_normalization(net)
    net = tf.nn.tanh(net)
    net = tf.layers.dropout(net,rate=prob)
    net = tf.reshape(net, shape=[-1, data_size], name='generator_output')
    return net
    
def dcgan_discriminator_mnist(x, y=None, prob=0.):
    filters = [64, 128, 256]
    alpha = 0.2
    net = tf.reshape(x, shape=[-1, 28, 28, 1])

    #reshape (adding zeros) to the more convenient 32x32 shape
    net = tf.pad(net, paddings=[[0,0],[2,2],[2,2],[0,0]], mode='CONSTANT', constant_values=0.)

    #condition concatenation for cGAN   
    if y != None:
        y = tf.reshape(y, shape=[-1, 1]) #(batch_size, 1)
        y = tf.tile(y, multiples=[1, 32*32])
        y = tf.reshape(y, shape=[-1, 32, 32, 1])
        net = tf.concat([net, y], axis=3)

    net = tf.layers.conv2d(net, filters=filters[0], kernel_size=[5,5], strides=[2,2],
                   activation=None, padding='same', name='layer1')
    net = tf.layers.batch_normalization(net)
    net = tf.nn.leaky_relu(net, alpha=alpha)
    net = tf.layers.dropout(net,rate=prob)

    net = tf.layers.conv2d(net, filters=filters[1], kernel_size=[5,5], strides=[2,2],
                   activation=None, padding='same', name='layer2')
    net = tf.layers.batch_normalization(net)
    net = tf.nn.leaky_relu(net, alpha=alpha)
    net = tf.layers.dropout(net,rate=prob)

    net = tf.layers.conv2d(net, filters=filters[2], kernel_size=[5,5], strides=[2,2],
                           activation=None, padding='same', name='layer3')
    net = tf.layers.batch_normalization(net)
    net = tf.nn.leaky_relu(net, alpha=alpha)
    net = tf.layers.dropout(net,rate=prob)

    net = tf.reshape(net, shape=[-1,filters[2]*4*4])    
    net = minibatch_discrimination(net, num_kernels=20, kernel_dim=10)

    #return the second to last layer to allow the implementation of feature discrimination
    features = net

    net = tf.layers.dense(net, 1, activation=None, name='layer_dense')
    return net, tf.nn.sigmoid(net, name='logit'), features

def dcgan_generator_mnist(noise, y=None, prob=0.):
    filters = [256, 128, 64]

    net = tf.reshape(noise, shape=[-1, noise.shape[1]])
    
    if y != None:
        y = tf.reshape(y, shape=[-1, 1])
        net = tf.concat([noise, y], axis=1)

    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tf.layers.dense(net, filters[0]*4*4, activation=None)
    net = tf.layers.dropout(net,rate=prob)
    net = tf.reshape(net, shape=[-1,4,4,filters[0]])
    
    net = tf.layers.conv2d_transpose(net, filters=filters[1], kernel_size=[5,5], strides=[2,2],
                                     activation=None, padding='same', name='layer1')
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net,rate=0.2)
    
    net = tf.layers.conv2d_transpose(net, filters=filters[2], kernel_size=[5,5], strides=[2,2],
                                     activation=None, padding='same', name='layer2')
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net,rate=0.2)

    net = tf.layers.conv2d_transpose(net, filters=1, kernel_size=[5,5], strides=[2,2],
                                         activation=None, padding='same', name='layer3')

    #crop the outer parts of the images to retrieve the original 28x28 size
    net = tf.slice(net, begin=[0,2,2,0], size=[-1,28,28,-1])
    net = tf.nn.tanh(net)
    net = tf.layers.dropout(net,rate=0.2)
    
    return tf.reshape(net, shape=[-1, 28, 28], name='output')

def dcgan_discriminator_cifar10(x, prob=0.):
    filters = [64, 128, 256]
    alpha = 0.2
    assert x.shape[1]==32
    assert x.shape[2]==32
    net = tf.reshape(x, shape=[-1, 32, 32, 3])
   
    net = tf.layers.conv2d(net, filters=filters[0], kernel_size=[5,5], strides=[2,2],
                   activation=None, padding='same', name='layer1')
    net = tf.layers.batch_normalization(net)
    net = tf.nn.leaky_relu(net, alpha=alpha)
    net = tf.layers.dropout(net,rate=prob)

    net = tf.layers.conv2d(net, filters=filters[1], kernel_size=[5,5], strides=[2,2],
                   activation=None, padding='same', name='layer2')
    net = tf.layers.batch_normalization(net)
    net = tf.nn.leaky_relu(net, alpha=alpha)
    net = tf.layers.dropout(net,rate=prob)

    net = tf.layers.conv2d(net, filters=filters[2], kernel_size=[5,5], strides=[2,2],
                           activation=None, padding='same', name='layer3')
    net = tf.layers.batch_normalization(net)
    net = tf.nn.leaky_relu(net, alpha=alpha)
    net = tf.layers.dropout(net,rate=prob)

    net = tf.reshape(net, shape=[-1,filters[2]*4*4])    
    net = minibatch_discrimination(net, num_kernels=20, kernel_dim=10)
    
    net = tf.layers.dense(net, 1, activation=None, name='layer4')
    return net, tf.nn.sigmoid(net, name='discriminator_logit')

def dcgan_generator_cifar10(noise, prob=0.):
    filters = [256, 128, 64]

    net = tf.reshape(noise, shape=[-1, noise.shape[1]])
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tf.layers.dense(net, filters[0]*4*4, activation=None)
    net = tf.layers.dropout(net,rate=prob)
    net = tf.reshape(net, shape=[-1,4,4,filters[0]])
    
    net = tf.layers.conv2d_transpose(net, filters=filters[1], kernel_size=[5,5], strides=[2,2],
                                     activation=None, padding='same', name='layer1')
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net,rate=0.2)

    net = tf.layers.conv2d_transpose(net, filters=filters[2], kernel_size=[5,5], strides=[2,2],
                                     activation=None, padding='same', name='layer2')
    net = tf.layers.batch_normalization(net)
    net = tf.nn.relu(net)
    net = tf.layers.dropout(net,rate=0.2)

    net = tf.layers.conv2d_transpose(net, filters=3, kernel_size=[5,5], strides=[2,2],
                                         activation=None, padding='same', name='layer3')

    net = tf.nn.tanh(net)
    net = tf.layers.dropout(net,rate=0.2)
    return tf.reshape(net, shape=[-1, 32, 32, 3], name='generator_output')
