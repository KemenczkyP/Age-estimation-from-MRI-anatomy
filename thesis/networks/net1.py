import tensorflow as tf

def net(input_layer, sex, num_class, drop_out_rate=0, training = 1):
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
    
    with tf.variable_scope("conv2", reuse=tf.AUTO_REUSE):
        conv_4 = tf.layers.conv3d( 
                inputs=dropout_3,
                filters=64, #depth
                kernel_size=[5,5,5],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu)
        pool_5 = tf.layers.max_pooling3d(inputs=conv_4,
                                           pool_size=[2, 2, 2],
                                           strides=2,
                                           padding="same",
                                           name = 'pool_5')
        dropout_6 = tf.layers.dropout(pool_5,rate=drop_out_rate)
        
    with tf.variable_scope("conv3", reuse=tf.AUTO_REUSE):
        conv_7 = tf.layers.conv3d( 
                inputs=dropout_6,
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
