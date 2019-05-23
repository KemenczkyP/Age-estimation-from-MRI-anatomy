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
    
    
    with tf.variable_scope("inception2", reuse=tf.AUTO_REUSE):
        # conv_5_1 = [batch, 39,47,39,2]
        conv_7_1 = tf.layers.conv3d( 
                inputs=dropout_3,
                filters=128, #depth
                kernel_size=[1, 1, 1],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_7_1')
        dropout_7_1 = tf.layers.dropout(conv_7_1,rate=drop_out_rate,
                name = 'dropout_7_1')
        # pool_6_1 = [batch, 20,24,20,2]
        pool_8_1 = tf.layers.max_pooling3d(inputs=dropout_7_1, pool_size=[2, 2, 2], strides=2, padding="same",
                name = 'pool_8_1')
        # conv_6_2 = [batch, 20,24,20,16]
        dropout_8_1 = tf.layers.dropout(pool_8_1,rate=drop_out_rate,
                name = 'dropout_8_1')
        
        
        # conv_5_2 = [batch, 39,47,39,2]
        conv_7_2 = tf.layers.conv3d( 
                inputs=dropout_3,
                filters=128, #depth
                kernel_size=[1, 1, 1],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_7_2')
        dropout_7_2 = tf.layers.dropout(conv_7_2,rate=drop_out_rate,
                name = 'dropout_7_2')
        conv_8_2 = tf.layers.conv3d( 
            inputs=dropout_7_2,
            filters=128, #depth
            kernel_size=[3, 3, 3],
            strides = (2,2,2),
            padding="same",
            activation=tf.nn.leaky_relu,
            name = 'conv_8_2')
        dropout_8_2 = tf.layers.dropout(conv_8_2,rate=drop_out_rate,
                name = 'dropout_8_2')
            
        
        # conv_5_3 = [batch, 39,47,39,2]
        conv_7_3 = tf.layers.conv3d( 
                inputs=dropout_3,
                filters=128, #depth
                kernel_size=[1, 1, 1],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_7_3')
        dropout_7_3 = tf.layers.dropout(conv_7_3,rate=drop_out_rate,
                name = 'dropout_7_3')
        # conv_6_3 = [batch, 20,24,20,8]
        conv_8_3 = tf.layers.conv3d( 
                inputs=dropout_7_3,
                filters=128, #depth
                kernel_size=[5,5,5],
                strides = (2,2,2),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_8_3')
        dropout_8_3 = tf.layers.dropout(conv_8_3,rate=drop_out_rate,
                name = 'dropout_8_3')
        
        conv_78_4 = tf.layers.conv3d( 
                inputs=dropout_3,
                filters=64, #depth
                kernel_size=[3,3,3],
                strides = (2,2,2),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_78_4')
        
        batchnorm_10 = tf.concat([dropout_8_1,dropout_8_2,dropout_8_3, conv_78_4],4,name = 'batchnorm_10')
        batchnorm_10 = tf.layers.batch_normalization(batchnorm_10, training)
    
    with tf.variable_scope("convout", reuse=tf.AUTO_REUSE):
        conv_last = tf.layers.conv3d( 
                    inputs=batchnorm_10,
                    filters=512, #depth
                    kernel_size=[3,3,3],
                    strides = (1,1,1),
                    padding="same",
                    activation=tf.nn.leaky_relu)
        pool_last = tf.layers.max_pooling3d(inputs=conv_last,
                                            pool_size=[2, 2, 2],
                                            strides=2,
                                            padding="same",
                                            name = 'pool_last')
        dropout_last = tf.layers.dropout(pool_last,rate=drop_out_rate)

    pool_25_flat = tf.reshape(dropout_last, [-1, dropout_last.shape[1] *dropout_last.shape[2]*dropout_last.shape[3] * 512],
            name = 'pool_12_flat')
    
    concat_s = tf.concat([pool_25_flat,sex],-1,name = 'concat_s')
    
    
    dense1 = tf.layers.dense(inputs=concat_s, units=128, activation=tf.nn.leaky_relu,
            name = 'dense_1')    
    dense1 = tf.layers.dropout(dense1,rate=drop_out_rate,
            name = 'dense_1d')

    logits = tf.layers.dense(inputs=dense1, units=num_class,
            name = 'logits_0')
        
    return logits