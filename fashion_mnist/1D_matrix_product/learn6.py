#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 09:06:42 2018
@author: xy
"""
import tensorflow as tf 
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy

num_kernel = 32
all_epoch = []
test_all_epoch = []
all_loss = []
all_train_accuracy = []
all_test_accuracy = []


#下载minist数据，创建mnist_data文件夹，one_hot编码
mnist = input_data.read_data_sets("data", one_hot=True)    
x = tf.placeholder(tf.float32, [None, 784])                        
y_real = tf.placeholder(tf.float32, shape=[None, 10])
#初始化权重
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)#正态分布
  return tf.Variable(initial)
#初始化偏置
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)#偏置初始化为0.1
  return tf.Variable(initial)
#构建卷积层
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')#卷积步长为1,不足补0
#构建池化层
def max_pool(x):
    #大小2*2,步长为2,不足补0
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def split_rank(x):
  x1, x2 = tf.split(x, 2, 2, name='split')
  x11, x12 = tf.split(x1, 2, 1, name='split')
  x21, x22 = tf.split(x2, 2, 1, name='split')
  x = [x11, x12, x21, x22]
  return x

def matmul_3D_2D(A, B):
  "A is 3D tensor, A.shape=(k, m, n), B is 2D tensor, B.shape=(n, p)"
  k, m, n = A.get_shape()
  n2, p = B.get_shape()
  A = tf.reshape(A,[k*m,n])
  C = tf.matmul(A, B)   # C.shape = AB.shape = (k*m,p)
  return tf.reshape(C,[k,m,p])

def matmul_4D_4D(A, B):
  "A is 4D tensor, A.shape=(k, m, n, 1), B is 4D tensor, B.shape=(n, p, 1, 1)"
  A = tf.squeeze(A, 3) # A.shape=(k, m, n)
  B = tf.squeeze(B, [2, 3])  # B.shape=(n, p)
  k, m, n = A.get_shape() 
  n2, p = B.get_shape()
  A = tf.reshape(A,[k*m,n])
  C = tf.matmul(A, B)   # C.shape = AB.shape = (k*m,p)
  C = tf.reshape(C,[k,m,p]) # C.shape = AB.shape = (k, m, p)
  return tf.expand_dims(C, 3)  # C.shape = AB.shape = (k, m, p, 1)
  
def two_d_matrix_product(x, W):
  x = split_rank(x)
  W = tf.squeeze(W, 3) # W.shape = (7,5,1)
  x[0] = tf.squeeze(x[0], 3) # x[0].shape = [-1,7,7]
  # tf.einsum('ibh,hnd->ibnd', left, right)
  x11 = tf.einsum('ibh,hnd->ibnd', x[0], W)  # x11.shape = (?,7,5,1)
  x[1] = tf.squeeze(x[1], 3) # x[1].shape = [-1,7,7]
  x12 = tf.einsum('ibh,hnd->ibnd', x[1], W)  # x12.shape = (?,7,5,1)
  x[2] = tf.squeeze(x[2], 3) # x[2].shape = [-1,7,7]
  x21 = tf.einsum('ibh,hnd->ibnd', x[2], W)  # x21.shape = (?,7,5,1)
  x[3] = tf.squeeze(x[3], 3) # x[3].shape = [-1,7,7]
  x22 = tf.einsum('ibh,hnd->ibnd', x[3], W)  # x22.shape = (?,7,5,1)
  mpro_1 = tf.concat([x11, x12], 1)
  mpro_2 = tf.concat([x21, x22], 1)
  mpro = tf.concat([mpro_1, mpro_2], 2)   # mpro.shape = (?,14,10,1)
  return mpro

def matrix_product(x, W1, W2, W3):
  x1, x2, x3 = tf.split(x, 3, 2, name='split') # x.shape = (batch_size, 28, 21, 1), x1 or x2 = (batch_size, 28, 7, 1)
  W1 = tf.squeeze(W1, 3) # W1.shape = (7,5,1)
  W2 = tf.squeeze(W2, 3) # W2.shape = (7,5,1)
  W3 = tf.squeeze(W3, 3) # W2.shape = (7,5,1)
  x1 = tf.squeeze(x1, 3) # x[0].shape = [-1,28,7]
  # tf.einsum('ibh,hnd->ibnd', left, right)
  x11 = tf.einsum('ibh,hnd->ibnd', x1, W1)  # x11.shape = (?,28,5,1)
  x2 = tf.squeeze(x2, 3) # x[1].shape = [-1,28,7]
  x12 = tf.einsum('ibh,hnd->ibnd', x2, W2)  # x12.shape = (?,28,5,1) 
  x3 = tf.squeeze(x3, 3) # x[1].shape = [-1,28,7]
  x13 = tf.einsum('ibh,hnd->ibnd', x3, W3)  # x12.shape = (?,28,5,1)   
  mpro = tf.concat([x11, x12, x13], 2) # mpro.shape = (?,28,10,1)
  return mpro

def matrix_product_1w(x, W):
  x = tf.squeeze(x, 3) # x[0].shape = [-1,28,1]
  W = tf.squeeze(W, 3) # W1.shape = (1,7,1)
  # tf.einsum('ibh,hnd->ibnd', left, right)
  x1 = tf.einsum('ibh,hnd->ibnd', x, W)  # x11.shape = (?,28,7,1)
  return x1
  
x_image = tf.reshape(x, [-1,28,28,1]) 
image = tf.split(x_image, 7, 1, name='split')  # the size of image[i] is (batch_size, 4, 28, 1)
imgs = []
for v in range(7):
  im = tf.split(image[v], 7, 2, name='split')  # the size of img[i][j] is (batch_size, 4, 4, 1)
  for l in range(7):
    imgs.append(im[l])

conv = []
names = globals()
for i,num in enumerate(range(num_kernel)):
  convv1 = []
  names['W_conv1' + str(i) ] = weight_variable([4, 3, 1, 1])
  names['W_conv2' + str(i) ] = weight_variable([4, 3, 1, 1])
  names['W_conv3' + str(i) ] = weight_variable([4, 3, 1, 1])
  # conv1_biases = tf.Variable(tf.zeros([1]))
  names['b_conv1' + str(i) ] = bias_variable([1]) 
  
  for j,num_iamge in enumerate(range(47)):
    h_conv1 = tf.nn.relu(matrix_product_1w(imgs[j], names['W_conv1' + str(i) ]) ) # h_conv1.shape = (batch_size, 28, 7, 1)
    convv1.append(h_conv1) 
    h_conv2 = tf.nn.relu(matrix_product_1w(imgs[j+1], names['W_conv2' + str(i) ]) )
    convv1.append(h_conv2)
    h_conv3 = tf.nn.relu(matrix_product_1w(imgs[j+2], names['W_conv3' + str(i) ]) )
    convv1.append(h_conv3)
  # convv1[0] = convv1[0]
  # convv1[1] = convv1[1] + convv1[3]
  # for ii in range(2, 47):
    # convv1[ii] = convv1[3*ii-4]+convv1[3*ii-2]+convv1[3*ii]
  # convv1[47] = convv1[139]+convv1[137]
  # convv1[48] = convv1[140]
  conv.append(tf.concat(convv1, 2))
relu = tf.concat(conv, 3)  
print('relu_size:', relu.get_shape().as_list())
h_pool1 = max_pool(relu)#池化层
[p, wi, hi, q] = h_pool1.get_shape().as_list()
# #第一层
# x_image = tf.reshape(x, [-1,28,28,1])         
# W_conv1 = weight_variable([5, 5, 1, 32])      
# b_conv1 = bias_variable([32])       
# h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)#卷积层
# h_pool1 = max_pool(h_conv1)#池化层
# #第二层
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      
# h_pool2 = max_pool(h_conv2)
#密集连接层
W_fc1 = weight_variable([wi * hi * num_kernel, 1024])
b_fc1 = bias_variable([1024])
h_pool1_flat = tf.reshape(h_pool1, [-1, wi * hi * num_kernel])              
h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)    
#dropout
keep_prob = tf.placeholder("float") 
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)                 
#输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_predict=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   
#模型训练评估
cross_entropy = -tf.reduce_sum(y_real*tf.log(y_predict))    
train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy) 
logs_loss=-tf.reduce_sum(tf.reduce_mean(y_real*tf.log(y_predict)))  
correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_real,1))    
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))                
sess=tf.InteractiveSession()                          
sess.run(tf.global_variables_initializer())
for i in range(50001):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:#训练100次
    train_loss = logs_loss.eval(feed_dict={x:batch[0], y_real: batch[1], keep_prob: 1.0})
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_real: batch[1], keep_prob: 1.0})
    print('step %d,training loss %g, training accuracy %g'%(i, train_loss, train_accuracy))
    all_epoch.append(i)
    all_loss.append(train_loss)
    all_train_accuracy.append(train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_real: batch[1], keep_prob: 0.5})
  if i%500 ==0:
    test_all_epoch.append(i)
    test_accuracy=accuracy.eval(feed_dict={x: mnist.test.images, y_real: mnist.test.labels, keep_prob: 1.0})
    all_test_accuracy.append(test_accuracy)
    print('step %d, test accuracy %g'%(i, test_accuracy))
    
numpy.save('./all_epoch.npy',all_epoch)
numpy.save('./all_loss.npy',all_loss)
numpy.save('./all_train_accuracy.npy',all_train_accuracy)
numpy.save('./test_all_epoch.npy',test_all_epoch)
numpy.save('./all_test_accuracy.npy',all_test_accuracy)

test_accuracy=accuracy.eval(feed_dict={x: mnist.test.images, y_real: mnist.test.labels, keep_prob: 1.0})
print("test accuracy",test_accuracy)