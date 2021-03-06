import tensorflow as tf

def net(input_layer, sex, num_class, drop_out_rate=0, training = 1):

    with tf.variable_scope("conv_group_1", reuse=tf.AUTO_REUSE):
        conv_1 = tf.layers.conv3d( 
                inputs=input_layer,
                filters=8, #depth
                kernel_size=[7,7,7],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv1')
        pool_2 = tf.layers.max_pooling3d(inputs=conv_1,
                                           pool_size=[2, 2, 2],
                                           strides=2,
                                           padding="same",
                                           name = 'pool1')
        dropout_3 = tf.layers.dropout(pool_2,rate=drop_out_rate, name = 'do1')
    
        
        
    with tf.variable_scope("inception1", reuse=tf.AUTO_REUSE):
        conv_4_1 = tf.layers.conv3d( 
                inputs=dropout_3,
                filters=32, #depth
                kernel_size=[1, 1, 1],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'inc-conv2_1')
        dropout_4_1 = tf.layers.dropout(conv_4_1,rate=drop_out_rate,
                name = 'do2_1')

        conv_4_2 = tf.layers.conv3d( 
                inputs=dropout_3,
                filters=32, #depth
                kernel_size=[3, 3, 3],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'inc-conv2_2')
        dropout_4_2 = tf.layers.dropout(conv_4_2,rate=drop_out_rate,
                name = 'do2_2')

        conv_4_3 = tf.layers.conv3d( 
                inputs=dropout_3,
                filters=32, #depth
                kernel_size=[5, 5, 5],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'inc-conv2_3')
        dropout_4_3 = tf.layers.dropout(conv_4_3,rate=drop_out_rate,
                name = 'do2_3')
        
        maxpooling_4_4 = tf.layers.max_pooling3d(inputs=dropout_3,
                                           pool_size=[2, 2, 2],
                                           strides=2,
                                           padding="same",
                                           name = 'pool2_4')
        deconv_4_4_2 = tf.layers.conv3d_transpose(inputs=maxpooling_4_4,
                                                  filters=16, #depth
                                                  kernel_size=[5, 5, 5],
                                                  strides = (2,2,2),
                                                  padding="same",
                                                  activation=tf.nn.leaky_relu,
                                                  name = 'inc-deconv2_4')
        batchnorm_6 = tf.concat([dropout_4_1,dropout_4_2,dropout_4_3, deconv_4_4_2],4,
                name = 'concat2')
    
    with tf.variable_scope("inception2", reuse=tf.AUTO_REUSE):
        conv_7_1 = tf.layers.conv3d( 
                inputs=batchnorm_6,
                filters=128, #depth
                kernel_size=[1, 1, 1],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_7_1')
        dropout_7_1 = tf.layers.dropout(conv_7_1,rate=drop_out_rate,
                name = 'dropout_7_1')
        pool_8_1 = tf.layers.max_pooling3d(inputs=dropout_7_1, pool_size=[2, 2, 2], strides=2, padding="same",
                name = 'pool_8_1')
        dropout_8_1 = tf.layers.dropout(pool_8_1,rate=drop_out_rate,
                name = 'dropout_8_1')
        

        conv_7_2 = tf.layers.conv3d( 
                inputs=batchnorm_6,
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
            

        conv_7_3 = tf.layers.conv3d( 
                inputs=batchnorm_6,
                filters=128, #depth
                kernel_size=[1, 1, 1],
                strides = (1,1,1),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_7_3')
        dropout_7_3 = tf.layers.dropout(conv_7_3,rate=drop_out_rate,
                name = 'dropout_7_3')

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
                inputs=batchnorm_6,
                filters=64, #depth
                kernel_size=[3,3,3],
                strides = (2,2,2),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_78_4')
        
        batchnorm_10 = tf.concat([dropout_8_1,dropout_8_2,dropout_8_3, conv_78_4],4,name = 'batchnorm_10')
        batchnorm_10 = tf.layers.batch_normalization(batchnorm_10, training)
    
    with tf.variable_scope("node1_convs_w_pooling", reuse=tf.AUTO_REUSE):
        conv_11 = tf.layers.conv3d(
                inputs=batchnorm_10,
                filters=512,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_11')
        dropout_12 = tf.layers.dropout(conv_11,rate=drop_out_rate,
                name = 'dropout_12')

        pool_13 = tf.layers.max_pooling3d(inputs=dropout_12, pool_size=[2,2,2], strides=2,
                name = 'pool_13')
        
        conv_14 = tf.layers.conv3d(
                inputs=pool_13,
                filters=1024,
                kernel_size=[3, 3, 3],
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_14')
        dropout_15 = tf.layers.dropout(conv_14,rate=drop_out_rate,
                name = 'dropout_15')

        pool_16 = tf.layers.max_pooling3d(inputs=dropout_15, pool_size=[2,2,2], strides=2,
                name = 'pool_16')
    
    with tf.variable_scope("node2_convolutions", reuse=tf.AUTO_REUSE):
        conv_17_2 = tf.layers.conv3d(
                inputs=batchnorm_10,
                filters=256,
                kernel_size=[1,1,1],
                strides=(2,2,2),
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_17_2')
        dropout_18_2 = tf.layers.dropout(conv_17_2,rate=drop_out_rate,
                name = 'dropout_18_2')
        conv_19_2 = tf.layers.conv3d(
                inputs=dropout_18_2,
                filters=1024,
                strides=(2,2,2),
                kernel_size=[5,5,5],
                padding="same",
                activation=tf.nn.leaky_relu,
                name = 'conv_19_2')
        dropout_20_2 = tf.layers.dropout(conv_19_2,rate=drop_out_rate,
                name = 'dropout_20_2')
    
    concat_21 = tf.concat([pool_16,dropout_20_2],4,name = 'concat_21')

    
    pool_25_flat = tf.reshape(concat_21, [-1, concat_21.shape[1] *concat_21.shape[2]*concat_21.shape[3] * 2048],
            name = 'flatten')
    
    concat_s = tf.concat([pool_25_flat,sex],-1,name = 'concat')
    
    
    dense1 = tf.layers.dense(inputs=concat_s, units=128, activation=tf.nn.leaky_relu,
            name = 'dense1')    
    dense1 = tf.layers.dropout(dense1,rate=drop_out_rate,
            name = 'dod1')

    logits = tf.layers.dense(inputs=dense1, units=num_class,
            name = 'denseo')
        
    return logits
