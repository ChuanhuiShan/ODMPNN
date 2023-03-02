#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#By @Kevin Xu
#kevin28520@gmail.com
#Youtube: https://www.youtube.com/channel/UCVCSn4qQXTDAtGWpWAe4Plw
#

#The aim of this project is to use TensorFlow to process our own data.
#    - input_data.py:  read in data and generate batches
#    - model: build the model architecture
#    - training: train

# data: cats vs. dogs from Kaggle
# Download link: https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
# data size: ~540M

# How to run?
# 1. run the training.py once
# 2. call the run_training() in the console to train the model.


#%%

import tensorflow as tf

num_kernel = 16
#%%

def inference(images, batch_size, n_classes):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]
    images = tf.reshape(images, [-1,50176,3])
    with tf.variable_scope('conv1_lrn') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [28,3, num_kernel],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[num_kernel],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv1d(images, weights, 1, "VALID")
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
    
    #pool1   
    with tf.variable_scope('pooling1') as scope:
        pool1 = tf.nn.pool(input=conv1, window_shape=[2], pooling_type="MAX", padding="VALID", strides=[2])#池化层
        # pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1],strides=[1,2,2,1],
        #                        padding='VALID', name='pooling1')
        # norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                               # beta=0.75,name='norm1')
    print(pool1.get_shape().as_list()) # pool1.shape = [None, 500, 6]

    # #conv2
    # with tf.variable_scope('conv2') as scope:
        # weights = tf.get_variable('weights',
                                  # shape=[5,5,96,256],
                                  # dtype=tf.float32,
                                  # initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        # biases = tf.get_variable('biases',
                                 # shape=[256], 
                                 # dtype=tf.float32,
                                 # initializer=tf.constant_initializer(0.1))
        # conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='VALID')
        # pre_activation = tf.nn.bias_add(conv, biases)
        # conv2 = tf.nn.relu(pre_activation, name='conv2')
    
    
    # #norm2 and pool2
    # with tf.variable_scope('pooling2_lrn') as scope:
        # pool2 = tf.nn.max_pool(conv2, ksize=[1,3,3,1], strides=[1,2,2,1],
                               # padding='VALID',name='pooling2')
        # norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                          # beta=0.75,name='norm2')
    # #conv3
    # with tf.variable_scope('conv3') as scope:
        # weights = tf.get_variable('weights',
                                  # shape=[3,3,256,384],
                                  # dtype=tf.float32,
                                  # initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        # biases = tf.get_variable('biases',
                                 # shape=[384], 
                                 # dtype=tf.float32,
                                 # initializer=tf.constant_initializer(0.1))
        # conv = tf.nn.conv2d(norm2, weights, strides=[1,1,1,1],padding='SAME')
        # pre_activation = tf.nn.bias_add(conv, biases)
        # conv3 = tf.nn.relu(pre_activation, name='conv3')

    # #conv4
    # with tf.variable_scope('conv4') as scope:
        # weights = tf.get_variable('weights',
                                  # shape=[3,3,384,384],
                                  # dtype=tf.float32,
                                  # initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        # biases = tf.get_variable('biases',
                                 # shape=[384], 
                                 # dtype=tf.float32,
                                 # initializer=tf.constant_initializer(0.1))
        # conv = tf.nn.conv2d(conv3, weights, strides=[1,1,1,1],padding='SAME')
        # pre_activation = tf.nn.bias_add(conv, biases)
        # conv4 = tf.nn.relu(pre_activation, name='conv4')


    # #conv5
    # with tf.variable_scope('conv5') as scope:
        # weights = tf.get_variable('weights',
                                  # shape=[3,3,384,256],
                                  # dtype=tf.float32,
                                  # initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        # biases = tf.get_variable('biases',
                                 # shape=[256], 
                                 # dtype=tf.float32,
                                 # initializer=tf.constant_initializer(0.1))
        # conv = tf.nn.conv2d(conv4, weights, strides=[1,1,1,1],padding='SAME')
        # pre_activation = tf.nn.bias_add(conv, biases)
        # conv5 = tf.nn.relu(pre_activation, name='conv5')

    
    
    # #pool6
    # with tf.variable_scope('pooling6') as scope:
# ##        norm2 = tf.nn.lrn(conv5, depth_radius=4, bias=1.0, alpha=0.001/9.0,
# ##                          beta=0.75,name='norm2')
        # pool6 = tf.nn.max_pool(conv5, ksize=[1,3,3,1], strides=[1,2,2,1],
                               # padding='VALID',name='pooling6')

    
    #local7
    with tf.variable_scope('local7') as scope:
        reshape = tf.reshape(pool1, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,4096],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[4096],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        local7 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)    
        local7 = tf.nn.dropout(local7, keep_prob=0.5)

        
    #local8
    with tf.variable_scope('local8') as scope:
        weights = tf.get_variable('weights',
                                  shape=[4096,4096],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[4096],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        local8 = tf.nn.relu(tf.matmul(local7, weights) + biases, name='local8')
        local8 = tf.nn.dropout(local8, keep_prob=0.5)


    # softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[4096, n_classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[n_classes],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(local8, weights), biases, name='softmax_linear')
    
    return softmax_linear

#%%
def losses(logits, labels):
    '''Compute loss from logits and labels
    Args:
        logits: logits tensor, float, [batch_size, n_classes]
        labels: label tensor, tf.int32, [batch_size]
        
    Returns:
        loss tensor of float type
    '''
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits\
                        (logits=logits, labels=labels, name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name+'/loss', loss)
    return loss

#%%
def trainning(loss, learning_rate):
    '''Training ops, the Op returned by this function is what must be passed to 
        'sess.run()' call to cause the model to train.
        
    Args:
        loss: loss tensor, from losses()
        
    Returns:
        train_op: The op for trainning
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op

#%%
def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy

#%%




