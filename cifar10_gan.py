import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.datasets.fashion_mnist import load_data as load_fashion
from src.architecture import dcgan_discriminator_cifar10, dcgan_generator_cifar10
from src.architecture import tikhonov_regularizer
from src import argparser
from src.utilities import write_to_file, log_tf_files
from src.utilities import PlotGenSamples

def noise(m,n):
    return np.random.normal(loc=0.0, scale=1., size=[m,n])

def generate(checkpoint, noise_size, size=(6,6)):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint)
    saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
    G_output = tf.get_default_graph().get_tensor_by_name('G/generator_output:0')

    gen_data_ph = tf.get_default_graph().get_tensor_by_name('G/gen_data_ph:0')
    dropout_prob_ph = tf.get_default_graph().get_tensor_by_name('dropout_prob_ph:0')
    batch_size_ph = tf.get_default_graph().get_tensor_by_name('batch_size_ph:0')

    with tf.Session() as sess:
        saver.restore(sess, latest_checkpoint)
        gen_samples = sess.run(G_output, feed_dict={gen_data_ph: noise(size[0]*size[1],noise_size),
                                                    dropout_prob_ph: 0., 
                                                    batch_size_ph: size[0]*size[1]})

    plot = PlotGenSamples()
    plot.plot_cifar10(gen_samples, 'generate')


def predict(checkpoint, noise_size, n_predictions=10):
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint)
    saver = tf.train.import_meta_graph(latest_checkpoint + '.meta')
    D_logit = tf.get_default_graph().get_tensor_by_name('D/discriminator_logit:0')

    data_ph = tf.get_default_graph().get_tensor_by_name('D/data_ph:0')
    dropout_prob_ph = tf.get_default_graph().get_tensor_by_name('dropout_prob_ph:0')
    batch_size_ph = tf.get_default_graph().get_tensor_by_name('batch_size_ph:0')

    (train_data, train_labels), _ = load_fashion()
    train_data = train_data[:n_predictions]
    train_data = train_data / 255.

    (real_train_data, real_train_labels), _ = load_data()
    real_train_data = real_train_data[:n_predictions]
    real_train_data = real_train_data / 255.
    
    with tf.Session() as sess:
        saver.restore(sess, latest_checkpoint)
        predictions = sess.run(D_logit, feed_dict={data_ph: train_data,
                                                   dropout_prob_ph: 0., 
                                                   batch_size_ph: 1})
        real_predictions = sess.run(D_logit, feed_dict={data_ph: real_train_data,
                                                   dropout_prob_ph: 0., 
                                                   batch_size_ph: 1})
    print()
    print("##Fake samples predictions##")
    for i,pred in enumerate(predictions):
        print('Prediction {}: {}'.format(i, pred[0]))
    print()
    print("##Real samples predictions##")
    for i,pred in enumerate(real_predictions):
        print('Prediction {}: {}'.format(i, pred[0]))

def train(nepochs, batch_size, noise_size, checkpoint):

    training_dataset, nbatches = train_data(batch_size)
    iterator = training_dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    
    dropout_prob_ph = tf.placeholder_with_default(0.0, shape=(), name='dropout_prob_ph')
    dropout_prob_D = 0.7
    dropout_prob_G = 0.3
    
    with tf.variable_scope('G'):
        gen_data_ph = tf.placeholder(tf.float32, shape=[None, noise_size], name='gen_data_ph')
        G_sample = dcgan_generator_cifar10(gen_data_ph, dropout_prob_ph)

    with tf.variable_scope('D') as scope:
        data_ph = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='data_ph')   
        D_real_logits, D_real = dcgan_discriminator_cifar10(data_ph, dropout_prob_ph)
    with tf.variable_scope('D', reuse=True):
        D_fake_logits, D_fake = dcgan_discriminator_cifar10(G_sample, dropout_prob_ph)
        
    flip_prob = 0.333 #label flipping
    flip_arr = np.random.binomial(n=1, p=flip_prob, size=(nepochs, nbatches))
    minval = .85 #smoothing
    batch_size_ph = tf.placeholder(tf.int32, shape=[], name='batch_size_ph') 
    real_labels_ph = tf.placeholder(tf.float32, name='real_labels_ph')
    fake_labels_ph = tf.placeholder(tf.float32, name='fake_labels_ph')
    
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits
    D_loss_real = cross_entropy(logits=D_real_logits, labels=real_labels_ph)
    D_loss_fake = cross_entropy(logits=D_fake_logits, labels=fake_labels_ph)
    gamma_ph = tf.placeholder(tf.float32, shape=[], name='gamma_ph')
    D_loss_reg = tikhonov_regularizer(D_real_logits, data_ph, D_fake_logits, gen_data_ph, batch_size_ph)
    D_loss = tf.reduce_mean(D_loss_real + D_loss_fake) #+ (gamma_ph/2.)*D_loss_reg
    G_loss = tf.reduce_mean(cross_entropy(logits=D_fake_logits, labels=real_labels_ph))
    
    tf.summary.scalar('D_loss', D_loss)
    tf.summary.scalar('G_loss', G_loss)
    
    log_tf_files(num_layers=3, loss=G_loss, player='G')
    log_tf_files(num_layers=3, loss=D_loss, player='D')
    
    D_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5)
    G_optimizer = tf.train.AdamOptimizer(0.0002, beta1=0.5)
    D_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'D')
    G_trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'G')
    
    D_train_op = D_optimizer.minimize(D_loss, var_list=D_trainable_vars, name='D_train_op')
    G_train_op = G_optimizer.minimize(G_loss, var_list=G_trainable_vars, name='G_train_op')
    
    #accuracy
    pred_classes_real = tf.round(D_real)
    labels_real = tf.ones_like(real_labels_ph, dtype=tf.float32)
    pred_classes_fake = tf.round(D_fake)
    labels_fake = tf.zeros_like(fake_labels_ph, dtype=tf.float32)
    pred_classes_tot = tf.concat([tf.round(D_real),tf.round(D_fake)], axis=0)
    labels_tot = tf.concat([labels_real, labels_fake], axis=0)
    with tf.name_scope('acc'):
        with tf.name_scope('acc_tot'):
            D_acc_tot, D_acc_tot_op = tf.metrics.accuracy(labels=labels_real, 
                                                          predictions=pred_classes_real)    
        with tf.name_scope('acc_real'):
            D_acc_real, D_acc_real_op = tf.metrics.accuracy(labels=labels_fake, 
                                                            predictions=pred_classes_fake)    
        with tf.name_scope('acc_fake'):
            D_acc_fake, D_acc_fake_op = tf.metrics.accuracy(labels=labels_tot, 
                                                            predictions=pred_classes_tot)    
    tf.summary.scalar('D_accuracy_real', D_acc_real)
    tf.summary.scalar('D_accuracy_fake', D_acc_fake)
    tf.summary.scalar('D_accuracy_tot', D_acc_tot)
    summary = tf.summary.merge_all()
    vars_train_reset = [v for v in tf.global_variables() if 'acc/' in v.name]

    plot = PlotGenSamples()    
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1)

    with tf.Session() as sess:
        init = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())
        sess.run(init)
        train_writer = tf.summary.FileWriter('tensorboard/'+str(np.random.randint(0,99999)), 
                                             sess.graph)

        for epoch in range(nepochs):
            print("Epoch: {}".format(epoch), flush=True)
            sess.run(iterator.initializer)

            #very slow: keep outside inner loop!
            #con: the accuracy between batches will be wrong
            sess.run(tf.variables_initializer(vars_train_reset)) 

            #tikhonov regularizer simulated annealing
            gamma = 2. * np.power(0.01, epoch/nepochs) 

            for batch in range(nbatches):
                inputs, _ = sess.run(next_element)

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
                noise_D = noise(len(inputs),noise_size)
                _, D_loss_c, D_real_c, D_fake_c = sess.run([D_train_op, D_loss, D_real, D_fake],
                              feed_dict={data_ph: inputs, gen_data_ph: noise_D,
                                         dropout_prob_ph: dropout_prob_D,
                                         batch_size_ph: len(inputs),
                                         real_labels_ph: real, fake_labels_ph: fake,
                                         gamma_ph: gamma})
                
                #train generator
                noise_G = noise(len(inputs),noise_size)
                _, G_loss_c  = sess.run([G_train_op, G_loss],
                                      feed_dict={data_ph: inputs, gen_data_ph: noise_G,
                                                 dropout_prob_ph: dropout_prob_G,
                                                 batch_size_ph: len(inputs),
                                                 real_labels_ph: real, fake_labels_ph: fake,
                                                 gamma_ph: gamma})

                D_acc_real_c, D_acc_fake_c, D_acc_tot_c = sess.run([D_acc_real, D_acc_fake, D_acc_tot],
                         feed_dict={data_ph: inputs, gen_data_ph: noise_G,
                                    dropout_prob_ph: dropout_prob_G,
                                    batch_size_ph: len(inputs),
                                    real_labels_ph: real, fake_labels_ph: fake,
                                    gamma_ph: gamma})
                         
                write_to_file('metrics_cifar10_gan.txt', [epoch*nbatches+(batch+1)],
                              [G_loss_c], [D_loss_c], [np.mean(D_real_c)], [np.mean(D_fake_c)])

            summ = sess.run(summary, 
                            feed_dict={data_ph: inputs, gen_data_ph: noise_G,
                                       dropout_prob_ph: dropout_prob_G,
                                       batch_size_ph: len(inputs),
                                       real_labels_ph: real, fake_labels_ph: fake,
                                       gamma_ph: gamma})
            train_writer.add_summary(summ, epoch*nbatches+(batch+1))    
      
            #save meta graph for later used
            saver.save(sess, checkpoint)

            #print generated samples
            sample = sess.run(G_sample, feed_dict={gen_data_ph: noise_G,
                                                   dropout_prob_ph: dropout_prob_G,
                                                   batch_size_ph: batch_size,
                                                   real_labels_ph: real, fake_labels_ph: fake,
                                                   gamma_ph: gamma})
            plot.plot_cifar10(sample[:36], 'cifar10_gen'+str(epoch))
            plot.plot_cifar10(inputs[:36], 'cifar10_data'+str(epoch))
            

def main(argv=None):
    nepochs = 1000
    batch_size = 128
    noise_size = 100 

    checkpoint = '/fred/oz012/Bruno/checkpoints/'+str(FLAGS.checkpoint)+'/'
    if FLAGS.mode == 'train':
        train(nepochs, batch_size, noise_size, checkpoint)
    elif FLAGS.mode == 'predict':
        predict(checkpoint, noise_size)
    elif FLAGS.mode == 'generate':
        generate(checkpoint, noise_size)
    else:
        raise ValueError('The specified mode is not valid.')

def train_data(batch_size):
    (train_data, train_labels), _ = load_data()
    train_data = train_data / 255.
    dataset_size = len(train_data)
    nbatches = int(np.ceil(dataset_size/batch_size))
    dataset = tf.data.Dataset.from_tensor_slices((train_data, train_labels))
    return dataset.shuffle(buffer_size=dataset_size).repeat(1).batch(batch_size), nbatches

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    FLAGS, _ = argparser.add_args(parser)
    tf.logging.set_verbosity(tf.logging.DEBUG)
    tf.app.run()
