import numpy as np
import tensorflow as tf

def noise(m,n):
    return np.random.normal(loc=0.0, scale=1., size=[m,n])

def linear_regression_layer(x, name):
    w = tf.Variable([1.0])
    b = tf.Variable([0.0])
    nb = name + '/bias'
    nw = name + '/kernel'
    return tf.math.add( tf.math.multiply(x, w, name=nb), b, name=nw)

def minibatch_discrimination(inputs, num_kernels=5, kernel_dim=3, name=''):
    with tf.variable_scope('minibatch_discrimination'):
        T = tf.get_variable('T', shape=[inputs.get_shape()[1], num_kernels*kernel_dim], 
                            initializer=tf.random_normal_initializer(stddev=0.02))
        M = tf.reshape(tf.matmul(inputs,T), (-1,num_kernels,kernel_dim))
        diffs = tf.expand_dims(M, 3) - tf.expand_dims(tf.transpose(M, [1,2,0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    return tf.concat([inputs, minibatch_features], 1, name=name)

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
