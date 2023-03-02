
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
SEED = 66478

def matr_prod2(xx, WW):  # xx.shape = [-1,8,8,3], WW.shape = (8,5,3,1)
    y = []
    k, m, n, l = xx.get_shape()
    WW = tf.squeeze(WW, 3) # W.shape = (8,5,3), WW.shape = (8,5,3,1)
    xx = tf.split(xx, 3, 3, name='split')  # xx.shape = [-1,8,8,3], xx[0].shape = [-1,8,8,1]
    WW = tf.split(WW, 3, 2, name='split')  # WW.shape = [8,5,3], xx[0].shape = [8,5,1]
    for i,num in enumerate(range(l)):
        xx[i] = tf.squeeze(xx[i], 3) # xx[i].shape = (-1,8,8)
        y.append(tf.einsum('ibh,hnd->ibnd', xx[i], WW[i])) # y[i].shape = (-1,8,5,1)
    return tf.reduce_sum(y, axis=0) # y.shape = (-1,8,5,1)

def matrix_product(x, W1, W2):
    x1, x2 = tf.split(x, 2, 2, name='split') # x.shape = (batch_size, 32, 16, 3), x1 or x2 = (batch_size, 224, 28, 3)
    x11 = matr_prod2(x1, W1) # x11.shape = (-1,28,5,1)
    x12 = matr_prod2(x2, W2) # x11.shape = (-1,28,5,1)  
    mpro = tf.concat([x11, x12], 2) # mpro.shape = (?,224,10,1)
    return mpro
    
def matrix_product_1w(xx, WW):
  xx = tf.split(xx, 3, 3, name='split') # (?,4,4,1)
  WW = tf.split(WW, 3, 2, name='split') # (4,3,1,1)
  x2 = 0
  for i in range(3):
    xx[i] = tf.squeeze(xx[i], 3) # x[0].shape = [-1,4,4]
    WW[i] = tf.squeeze(WW[i], 3) # W1.shape = (4,3,1)
    # tf.einsum('ibh,hnd->ibnd', left, right)
    x1 = tf.einsum('ibh,hnd->ibnd', xx[i], WW[i])  # x1.shape = (?,4,3,1)
    # print('x1_size:', x1.get_shape().as_list())
    x2 = tf.add(x1, x2)
  return x2
    
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)
  
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
  
#%%
def inference(images, batch_size, n_classes):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''
    #conv1, shape = [kernel size, kernel size, channels, kernel numbers]
    
    # with tf.variable_scope('conv1_lrn') as scope:
        # weights = tf.get_variable('weights', 
                                  # shape = [11,11,3, 32],
                                  # dtype = tf.float32, 
                                  # initializer=tf.truncated_normal_initializer(stddev=0.01,dtype=tf.float32))
        # biases = tf.get_variable('biases', 
                                 # shape=[32],
                                 # dtype=tf.float32,
                                 # initializer=tf.constant_initializer(0.1))
        # conv = tf.nn.conv2d(images, weights, strides=[1,2,2,1], padding='VALID')
        # pre_activation = tf.nn.bias_add(conv, biases)
        # conv1 = tf.nn.relu(pre_activation, name= scope.name) 
    x_image = tf.reshape(images, [-1,224,224,3]) 
    image = tf.split(x_image, 8, 1, name='split')  # the size of image11 is (batch_size, 32, 4, 3)
    imgs = []
    for v in range(8):
        im = tf.split(image[v], 8, 2, name='split')  # the size of img[i][j] is (batch_size, 4, 4, 3)
        for l in range(8):
            imgs.append(im[l])


    conv = []
    names = globals()
    for i,num in enumerate(range(num_kernel)):
        convv1 = []
        names['W_conv1' + str(i) ] = weight_variable([28, 10, 3, 1])
        names['W_conv2' + str(i) ] = weight_variable([28, 10, 3, 1])
        names['W_conv3' + str(i) ] = weight_variable([28, 10, 3, 1])
        # conv1_biases = tf.Variable(tf.zeros([1]))
        names['b_conv1' + str(i) ] = bias_variable([1]) 
  
        for j,num_iamge in enumerate(range(62)):
            h_conv1 = tf.nn.relu(matrix_product_1w(imgs[j], names['W_conv1' + str(i) ]) ) # h_conv1.shape = (batch_size, 4, 4, 3)
            convv1.append(h_conv1) 
            h_conv2 = tf.nn.relu(matrix_product_1w(imgs[j+1], names['W_conv2' + str(i) ]) )
            convv1.append(h_conv2)
            h_conv3 = tf.nn.relu(matrix_product_1w(imgs[j+2], names['W_conv3' + str(i) ]) )
            convv1.append(h_conv3)
      
        conv.append(tf.concat(convv1, 2))
    relu = tf.concat(conv, 3)  
    print('relu_size:', relu.get_shape().as_list())
    
    with tf.variable_scope('pooling1') as scope:
        pool1 = tf.nn.max_pool(relu, ksize=[1,3,3,1],strides=[1,2,2,1],
                               padding='VALID', name='pooling1')
        # norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                               # beta=0.75,name='norm1')
    
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




