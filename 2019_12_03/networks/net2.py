# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 09:32:54 2019

@author: KemyPeti
"""
def net(input_layer, sex, num_class, drop_out_rate=0, training = 1):
    import tensorflow as tf
    

    # conv_1 = [batch, 39,47,39,1]
    with tf.variable_scope("conv1", reuse=tf.AUTO_REUSE):
        conv_1 = tf.layers.conv3d( 
                inputs=input_layer,
                filters=8, #depth
                kernel_size=[7,7,7],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu)
        pool_2 = tf.layers.max_pooling3d(inputs=conv_1,
                                           pool_size=[2, 2, 2],
                                           strides=2,
                                           padding="same",
                                           name = 'pool_2')
        dropout_3 = tf.layers.dropout(pool_2,rate=drop_out_rate)
    
    with tf.variable_scope("inception1", reuse=tf.AUTO_REUSE):
        # conv_2_1 = [batch, 39,47,39,4]
        conv_4_1 = tf.layers.conv3d( 
                inputs=dropout_3,
                filters=32, #depth
                kernel_size=[1, 1, 1],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_4_1')
        dropout_4_1 = tf.layers.dropout(conv_4_1,rate=drop_out_rate,
                name = 'dropout_4_1')
        
        # conv_2_2 = [batch, 39,47,39,4]
        conv_4_2 = tf.layers.conv3d( 
                inputs=dropout_3,
                filters=32, #depth
                kernel_size=[3, 3, 3],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_4_2')
        dropout_4_2 = tf.layers.dropout(conv_4_2,rate=drop_out_rate,
                name = 'dropout_4_2')
        
        # conv_2_3 = [batch, 39,47,39,4]
        conv_4_3 = tf.layers.conv3d( 
                inputs=dropout_3,
                filters=32, #depth
                kernel_size=[5, 5, 5],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_4_3')
        dropout_4_3 = tf.layers.dropout(conv_4_3,rate=drop_out_rate,
                name = 'dropout_4_3')
        
        maxpooling_4_4 = tf.layers.max_pooling3d(inputs=dropout_3,
                                           pool_size=[2, 2, 2],
                                           strides=2,
                                           padding="same",
                                           name = 'maxpooling_4_4')
        deconv_4_4_2 = tf.layers.conv3d_transpose(inputs=maxpooling_4_4,
                                                  filters=16, #depth
                                                  kernel_size=[5, 5, 5],
                                                  strides = (2,2,2),
                                                  padding="same",
                                                  activation=tf.nn.leaky_relu,
                                                  name = 'deconv_4_4_2') 
        # concat_4 = [batch, 39,47,39,16]
        batchnorm_6 = tf.concat([dropout_4_1,dropout_4_2,dropout_4_3, deconv_4_4_2],4,
                name = 'batchnorm_6')
        
    with tf.variable_scope("conv3", reuse=tf.AUTO_REUSE):
        conv_7 = tf.layers.conv3d( 
                inputs=batchnorm_6,
                filters=256, #depth
                kernel_size=[3,3,3],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu)
        pool_8 = tf.layers.max_pooling3d(inputs=conv_7,
                                           pool_size=[2, 2, 2],
                                           strides=2,
                                           padding="same",
                                           name = 'pool_8')
        dropout_9 = tf.layers.dropout(pool_8,rate=drop_out_rate)    
    
    with tf.variable_scope("conv4", reuse=tf.AUTO_REUSE):
        conv_10 = tf.layers.conv3d( 
                inputs=dropout_9,
                filters=512, #depth
                kernel_size=[3,3,3],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu)
        pool_11 = tf.layers.max_pooling3d(inputs=conv_10,
                                           pool_size=[2, 2, 2],
                                           strides=2,
                                           padding="same",
                                           name = 'pool_11')
        dropout_12 = tf.layers.dropout(pool_11,rate=drop_out_rate)    

    
    pool_13_flat = tf.reshape(dropout_12, [-1, dropout_12.shape[1] *dropout_12.shape[2]*dropout_12.shape[3] * 512],
            name = 'pool_13_flat')
    
    concat_s = tf.concat([pool_13_flat,sex],-1,name = 'concat_s')
    
    
    dense1 = tf.layers.dense(inputs=concat_s, units=128, activation=tf.nn.leaky_relu,
            name = 'dense_1')    
    dense1 = tf.layers.dropout(dense1,rate=drop_out_rate,
            name = 'dense_1d')

    logits = tf.layers.dense(inputs=dense1, units=num_class,
            name = 'logits_0')

    return logits

