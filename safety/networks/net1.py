# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:32:54 2019

@author: latlab
"""

def net1(input_layer, num_class, drop_out_rate=0):
    import tensorflow as tf
    # conv_1 = [batch, 39,47,39,1]
    with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
        conv_1 = tf.layers.conv3d( 
                inputs=input_layer,
                filters=1, #depth
                kernel_size=[9,9,9],
                strides = (2,2,2),
                padding="same",
                activation=tf.nn.leaky_relu)
        conv_1 = tf.layers.dropout(conv_1,rate=drop_out_rate)
    
    # conv_2_1 = [batch, 39,47,39,4]
    conv_2_1 = tf.layers.conv3d( 
            inputs=conv_1,
            filters=20, #depth
            kernel_size=[1, 1, 1],
            strides = (1,1,1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name = 'conv_2_1')
    conv_2_1 = tf.layers.dropout(conv_2_1,rate=drop_out_rate,
            name = 'conv_2_1d')
    
    # conv_2_2 = [batch, 39,47,39,4]
    conv_2_2 = tf.layers.conv3d( 
            inputs=conv_1,
            filters=20, #depth
            kernel_size=[3, 3, 3],
            strides = (1,1,1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name = 'conv_2_2')
    conv_2_2 = tf.layers.dropout(conv_2_2,rate=drop_out_rate,
            name = 'conv_2_2d')
    
    # conv_2_3 = [batch, 39,47,39,4]
    conv_2_3 = tf.layers.conv3d( 
            inputs=conv_1,
            filters=20, #depth
            kernel_size=[5, 5, 5],
            strides = (1,1,1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name = 'conv_2_3')
    conv_2_3 = tf.layers.dropout(conv_2_3,rate=drop_out_rate,
            name = 'conv_2_3d')

    # concat_4 = [batch, 39,47,39,16]
    concat_4 = tf.concat([conv_2_1,conv_2_2,conv_2_3],4,
            name = 'concat_4')
    
    # conv_5_1 = [batch, 39,47,39,2]
    conv_5_1 = tf.layers.conv3d( 
            inputs=concat_4,
            filters=64, #depth
            kernel_size=[1, 1, 1],
            strides = (1,1,1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name = 'conv_5_1')
    conv_5_1 = tf.layers.dropout(conv_5_1,rate=drop_out_rate,
            name = 'conv_5_1d')
    
    # conv_5_2 = [batch, 39,47,39,2]
    conv_5_2 = tf.layers.conv3d( 
            inputs=concat_4,
            filters=64, #depth
            kernel_size=[1, 1, 1],
            strides = (1,1,1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name = 'conv_5_2')
    conv_5_2 = tf.layers.dropout(conv_5_2,rate=drop_out_rate,
            name = 'conv_5_2d')
    
    # conv_5_3 = [batch, 39,47,39,2]
    conv_5_3 = tf.layers.conv3d( 
            inputs=concat_4,
            filters=64, #depth
            kernel_size=[1, 1, 1],
            strides = (1,1,1),
            padding="same",
            activation=tf.nn.leaky_relu,
            name = 'conv_5_3')
    conv_5_3 = tf.layers.dropout(conv_5_3,rate=drop_out_rate,
            name = 'conv_5_3d')
    
    # pool_6_1 = [batch, 20,24,20,2]
    pool_6_1 = tf.layers.max_pooling3d(inputs=conv_5_1, pool_size=[2, 2, 2], strides=2, padding="same",
            name = 'pool_6_1')
    # conv_6_2 = [batch, 20,24,20,16]
    pool_6_1 = tf.layers.dropout(pool_6_1,rate=drop_out_rate,
            name = 'pool_6_1d')
    
    conv_6_2 = tf.layers.conv3d( 
            inputs=conv_5_2,
            filters=128, #depth
            kernel_size=[3, 3, 3],
            strides = (2,2,2),
            padding="same",
            activation=tf.nn.leaky_relu,
            name = 'conv_6_2')
    conv_6_2 = tf.layers.dropout(conv_6_2,rate=drop_out_rate,
            name = 'conv_6_2d')
    
    # conv_6_3 = [batch, 20,24,20,8]
    conv_6_3 = tf.layers.conv3d( 
            inputs=conv_5_3,
            filters=64, #depth
            kernel_size=[5,5,5],
            strides = (2,2,2),
            padding="same",
            activation=tf.nn.leaky_relu,
            name = 'conv_6_3')
    conv_6_3 = tf.layers.dropout(conv_6_3,rate=drop_out_rate,
            name = 'conv_6_3d')

     # concat_7 = [batch, 19,23,19,58]
    concat_7 = tf.concat([pool_6_1,conv_6_2,conv_6_3],4,
            name = 'concat_7')
    
    conv_8 = tf.layers.conv3d(
            inputs=concat_7,
            filters=256,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.leaky_relu,
            name = 'conv_8')
    conv_8 = tf.layers.dropout(conv_8,rate=0.2,
            name = 'conv_8d')


    # pool_9 = [batch, 9,11,9,58]
    pool9 = tf.layers.max_pooling3d(inputs=conv_8, pool_size=[2,2,2], strides=2,
            name = 'pool_9')
    
    conv_10 = tf.layers.conv3d(
            inputs=pool9,
            filters=256,
            kernel_size=[3, 3, 3],
            padding="same",
            activation=tf.nn.leaky_relu,
            name = 'conv_10')
    conv_10 = tf.layers.dropout(conv_10,rate=drop_out_rate,
            name = 'conv_10d')

    
    # pool_9 = [batch, 4,5,4,64]
    pool11 = tf.layers.max_pooling3d(inputs=conv_10, pool_size=[2,2,2], strides=2,
            name = 'pool_11')
    
    pool12_flat = tf.reshape(pool11, [-1, pool11.shape[1] *pool11.shape[2]*pool11.shape[3] * 256],
            name = 'pool_12_flat')
    
    dense1 = tf.layers.dense(inputs=pool12_flat, units=128, activation=tf.nn.leaky_relu,
            name = 'dense_1')    
    dense1 = tf.layers.dropout(dense1,rate=drop_out_rate,
            name = 'dense_1d')

    logits = tf.layers.dense(inputs=dense1, units=num_class,
            name = 'logits_0')
        
    return logits, conv_10