# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 08:35:13 2019

@author: KemenczkyP
"""

import tensorflow as tf

class net3(tf.keras.Model):

  def __init__(self):
    super(net3, self).__init__()
    
    self.conv_1 = tf.keras.layers.Conv3D(filters = 8,
                                         activation=tf.nn.leaky_relu,
                                         kernel_size = [7,7,7],
                                         padding='same')
    
    
    self.conv_2_1 = tf.keras.layers.Conv3D(filters = 32,
                                           activation=tf.nn.leaky_relu,
                                           kernel_size = [1,1,1],
                                           padding='same')
    self.conv_2_2 = tf.keras.layers.Conv3D(filters = 32,
                                           activation=tf.nn.leaky_relu,
                                           kernel_size = [3,3,3],
                                           padding='same')
    self.conv_2_3 = tf.keras.layers.Conv3D(filters = 32,
                                           activation=tf.nn.leaky_relu,
                                           kernel_size = [5,5,5],
                                           padding='same')
    self.deconv_2_4 = tf.keras.layers.Conv3DTranspose(filters=16, #depth
                                                      kernel_size=[5, 5, 5],
                                                      strides = (2,2,2),
                                                      padding="same",
                                                      activation=tf.nn.leaky_relu)
    
    self.conv_3_1 = tf.keras.layers.Conv3D(filters = 128,
                                           activation=tf.nn.leaky_relu,
                                           kernel_size = [1,1,1],
                                           padding='same')
    
    self.conv_3_2_1 = tf.keras.layers.Conv3D(filters = 128,
                                           activation=tf.nn.leaky_relu,
                                           kernel_size = [1,1,1],
                                           padding='same')
    self.conv_3_2_2 = tf.keras.layers.Conv3D(filters = 128,
                                             activation=tf.nn.leaky_relu,
                                             kernel_size=[3, 3, 3],
                                             strides = (2,2,2),
                                             padding='same')
    
    self.conv_3_3_1 = tf.keras.layers.Conv3D(filters = 128,
                                           activation=tf.nn.leaky_relu,
                                           kernel_size = [1,1,1],
                                           padding='same')
    self.conv_3_3_2 = tf.keras.layers.Conv3D(filters = 128,
                                             activation=tf.nn.leaky_relu,
                                             kernel_size=[5,5,5],
                                             strides = (2,2,2),
                                             padding='same')
            
    self.conv_3_4 = tf.keras.layers.Conv3D(filters = 64,
                                           activation=tf.nn.leaky_relu,
                                           kernel_size=[3,3,3],
                                           strides = (2,2,2),
                                           padding='same')
    
    self.conv_4_1_1 = tf.keras.layers.Conv3D(filters = 512,
                                           activation=tf.nn.leaky_relu,
                                           kernel_size = [3,3,3],
                                           padding='same')
    self.conv_4_1_2 = tf.keras.layers.Conv3D(filters = 1024,
                                           activation=tf.nn.leaky_relu,
                                           kernel_size = [3,3,3],
                                           padding='same')
    
    self.conv_4_2_1 = tf.keras.layers.Conv3D(filters = 256,
                                             activation=tf.nn.leaky_relu,
                                             kernel_size=[1,1,1],
                                             strides = (2,2,2),
                                             padding='same')
    self.conv_4_2_2 = tf.keras.layers.Conv3D(filters = 1024,
                                             activation=tf.nn.leaky_relu,
                                             kernel_size=[5,5,5],
                                             strides = (2,2,2),
                                             padding='same')
    
    self.dense_1 = tf.keras.layers.Dense(units=128,
                                        activation=tf.nn.leaky_relu)
    
    self.dense_out = tf.keras.layers.Dense(units=1)
    
    self.max_pool = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2),
                                              strides = 2,
                                              padding='same')
    self.batch_norm = tf.keras.layers.BatchNormalization()
    

  def call(self, inputs, sex, dropout_rate):
    
    # INPUT CONVOLUTION
    x = self.conv_1(inputs)
    x = self.max_pool(x)
    x = tf.keras.backend.dropout(x, dropout_rate)
    
    # INCEPTION1
    x_2_1 =  self.conv_2_1(x)
    x_2_1 = tf.keras.backend.dropout(x_2_1, dropout_rate)
    
    x_2_2 =  self.conv_2_2(x)
    x_2_2 = tf.keras.backend.dropout(x_2_2, dropout_rate)
    
    x_2_3 =  self.conv_2_3(x)
    x_2_3 = tf.keras.backend.dropout(x_2_3, dropout_rate)
    
    x_2_4 = self.max_pool(x)
    x_2_4 = self.deconv_2_4(x_2_4)
    
    x = tf.keras.backend.concatenate([x_2_1,x_2_2,x_2_3, x_2_4], 4)
    
    # INCEPTION2
    x_3_1 = self.conv_3_1(x)
    x_3_1 = tf.keras.backend.dropout(x_3_1, dropout_rate)
    x_3_1 = self.max_pool(x_3_1)
    x_3_1 = tf.keras.backend.dropout(x_3_1, dropout_rate)
    
    x_3_2 = self.conv_3_2_1(x)
    x_3_2 = tf.keras.backend.dropout(x_3_2, dropout_rate)
    x_3_2 = self.conv_3_2_2(x_3_2)
    x_3_2 = tf.keras.backend.dropout(x_3_2, dropout_rate)
    
    x_3_3 = self.conv_3_3_1(x)
    x_3_3 = tf.keras.backend.dropout(x_3_3, dropout_rate)
    x_3_3 = self.conv_3_3_2(x_3_3)
    x_3_3 = tf.keras.backend.dropout(x_3_3, dropout_rate)
    
    x_3_4 = self.conv_3_4(x)
    x_3_4 = tf.keras.backend.dropout(x_3_4, dropout_rate)
    
    x = tf.keras.backend.concatenate([x_3_1,x_3_2,x_3_3, x_3_4], 4)
    x = self.batch_norm(x)

    # INCEPTION3
    x_4_1 = self.conv_4_1_1(x)
    x_4_1 = tf.keras.backend.dropout(x_4_1, dropout_rate)
    x_4_1 = self.max_pool(x_4_1)
    x_4_1 = self.conv_4_1_2(x_4_1)
    x_4_1 = tf.keras.backend.dropout(x_4_1, dropout_rate)
    x_4_1 = self.max_pool(x_4_1)
    
    x_4_2 = self.conv_4_2_1(x)
    x_4_2 = tf.keras.backend.dropout(x_4_2, dropout_rate)
    x_4_2 = self.conv_4_2_2(x_4_2)
    x_4_2 = tf.keras.backend.dropout(x_4_2, dropout_rate)
    
    x = tf.keras.backend.concatenate([x_4_1,x_4_2], 4)
    
    #FLATTEN
    x = tf.keras.backend.reshape(x, [-1, x.shape[1] *x.shape[2]*x.shape[3] * 2048])
    
    #CONCAT GENRE
    x = tf.keras.backend.concatenate([x,sex], -1)
    x = tf.keras.backend.dropout(x, dropout_rate)
    
    #OUTPUT DENSE LAYERS
    x = self.dense_1(x)
    x = self.dense_out(x)
    
    return x
