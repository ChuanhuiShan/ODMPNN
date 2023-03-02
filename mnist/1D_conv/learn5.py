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
def conv1d(x, W):
  return tf.nn.conv1d(x, W, 1, "VALID")#卷积步长为1,不足补0
#构建池化层
# def max_pool(x):
    # #大小2*2,步长为2,不足补0
  # return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],strides=[1, 1, 2, 1], padding='SAME')
#第一层
x_image = tf.reshape(x, [-1,784,1])         
W_conv1 = weight_variable([4, 1, num_kernel])      
b_conv1 = bias_variable([num_kernel])       
h_conv1 = tf.nn.relu(conv1d(x_image, W_conv1) + b_conv1)#卷积层
# print(h_conv1.get_shape().as_list())
h_pool1 = tf.nn.pool(input=h_conv1, window_shape=[2], pooling_type="MAX", padding="VALID", strides=[2])#池化层
# print(h_pool1.get_shape().as_list())
[p, wi, q] = h_pool1.get_shape().as_list()
# #第二层
# W_conv2 = weight_variable([5, 5, 32, 64])
# b_conv2 = bias_variable([64])
# h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)      
# h_pool2 = max_pool(h_conv2)
#密集连接层
W_fc1 = weight_variable([wi * num_kernel, 1024])
b_fc1 = bias_variable([1024])
h_pool1_flat = tf.reshape(h_pool1, [-1, wi*num_kernel])              
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
  # print(batch[0].shape)
  # print(batch[1].shape)
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