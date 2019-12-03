import tensorflow as tf

print("Tensorflow version: {}".format(tf.__version__))

def generate_model(drop_out_rate):
    
    inputs = tf.keras.Input(shape=(79,95,79,1))
    sexs = tf.keras.Input(shape=(1))
    
    ############################### CONV 1 ####################################
    
    conv_1 = tf.keras.layers.Conv3D(filters = 8,
                                    kernel_size = (7,7,7),
                                    strides = (1,1,1),
                                    padding='same',
                                    activation = tf.nn.leaky_relu)(inputs)
    pool_2 = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),
                                       strides=(2,2,2),
                                       padding='same')(conv_1)
    dropout_3 = tf.keras.layers.Dropout(rate = drop_out_rate)(pool_2)
    
    ############################ INCEPTION 1 ##################################
    conv_4_1 = tf.keras.layers.Conv3D(filters = 32,
                                    kernel_size = (1,1,1),
                                    strides = (1,1,1),
                                    padding='same',
                                    activation = tf.nn.leaky_relu)(dropout_3)
    dropout_4_1 = tf.keras.layers.Dropout(rate = drop_out_rate)(conv_4_1)
    
    
    conv_4_2 = tf.keras.layers.Conv3D(filters = 32,
                                    kernel_size = (3,3,3),
                                    strides = (1,1,1),
                                    padding='same',
                                    activation = tf.nn.leaky_relu)(dropout_3)
    dropout_4_2 = tf.keras.layers.Dropout(rate = drop_out_rate)(conv_4_2)
    
    
    conv_4_3 = tf.keras.layers.Conv3D(filters = 32,
                                    kernel_size = (5,5,5),
                                    strides = (1,1,1),
                                    padding='same',
                                    activation = tf.nn.leaky_relu)(dropout_3)
    dropout_4_3 = tf.keras.layers.Dropout(rate = drop_out_rate)(conv_4_3)
    
    
    maxpooling_4_4 = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),
                                               strides=(2,2,2),
                                               padding='same')(dropout_3)
    
    deconv_4_4_2 = tf.keras.layers.Conv3DTranspose(filters=16,
                                                   kernel_size = (5,5,5),
                                                   strides=(2,2,2),
                                                   padding='same',
                                                   activation = tf.nn.leaky_relu)(maxpooling_4_4)
    
    batchnorm_6 = tf.keras.layers.Concatenate(axis = -1)([dropout_4_1,dropout_4_2,dropout_4_3, deconv_4_4_2])
        
    ############################ INCEPTION 2 ##################################
    conv_7_1 = tf.keras.layers.Conv3D(filters = 128,
                                    kernel_size = (1,1,1),
                                    strides = (1,1,1),
                                    padding='same',
                                    activation = tf.nn.leaky_relu)(batchnorm_6)
    dropout_7_1 = tf.keras.layers.Dropout(rate = drop_out_rate)(conv_7_1)
    pool_8_1 = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),
                                               strides=(2,2,2),
                                               padding='same')(dropout_7_1)
    dropout_8_1 = tf.keras.layers.Dropout(rate = drop_out_rate)(pool_8_1)
    
    
    conv_7_2 = tf.keras.layers.Conv3D(filters = 128,
                                    kernel_size = (1,1,1),
                                    strides = (1,1,1),
                                    padding='same',
                                    activation = tf.nn.leaky_relu)(batchnorm_6)
    dropout_7_2 = tf.keras.layers.Dropout(rate = drop_out_rate)(conv_7_2)
    conv_8_2 = tf.keras.layers.Conv3D(filters = 128,
                                    kernel_size = (3,3,3),
                                    strides = (2,2,2),
                                    padding='same',
                                    activation = tf.nn.leaky_relu)(dropout_7_2)
    dropout_8_2 = tf.keras.layers.Dropout(rate = drop_out_rate)(conv_8_2)

    conv_7_3 = tf.keras.layers.Conv3D(filters = 128,
                                kernel_size = (1,1,1),
                                strides = (1,1,1),
                                padding='same',
                                activation = tf.nn.leaky_relu)(batchnorm_6)
    dropout_7_3 = tf.keras.layers.Dropout(rate = drop_out_rate)(conv_7_3)
            
    conv_8_3 = tf.keras.layers.Conv3D(filters = 128,
                                kernel_size = (5,5,5),
                                strides = (2,2,2),
                                padding='same',
                                activation = tf.nn.leaky_relu)(dropout_7_3)
    dropout_8_3 = tf.keras.layers.Dropout(rate = drop_out_rate)(conv_8_3)
    
    conv_78_4 = tf.keras.layers.Conv3D(filters = 64,
                                kernel_size = (3,3,3),
                                strides = (2,2,2),
                                padding='same',
                                activation = tf.nn.leaky_relu)(batchnorm_6)
    
    batchnorm_10 = tf.keras.layers.Concatenate(axis = -1)([dropout_8_1,dropout_8_2,dropout_8_3, conv_78_4])
    batchnorm_10 = tf.keras.layers.BatchNormalization()(batchnorm_10)
        
    ################################ NODE 1 ###################################
    conv_11 = tf.keras.layers.Conv3D(filters = 512,
                                kernel_size = (3,3,3),
                                padding='same',
                                activation = tf.nn.leaky_relu)(batchnorm_10)
    dropout_12 = tf.keras.layers.Dropout(rate = drop_out_rate)(conv_11)
    
    pool_13 = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),
                                        strides=(2,2,2),
                                        padding='same')(dropout_12)
    conv_14 = tf.keras.layers.Conv3D(filters = 1024,
                                kernel_size = (3,3,3),
                                padding='same',
                                activation = tf.nn.leaky_relu)(pool_13)
    dropout_15 = tf.keras.layers.Dropout(rate = drop_out_rate)(conv_14)
    pool_16 = tf.keras.layers.MaxPool3D(pool_size=(2,2,2),
                                        strides=(2,2,2),
                                        padding='same')(dropout_15)   

    ################################ NODE 2 ###################################
    conv_17_2 = tf.keras.layers.Conv3D(filters = 256,
                                kernel_size = (1,1,1),
                                strides=(2,2,2),
                                padding='same',
                                activation = tf.nn.leaky_relu)(batchnorm_10)
    dropout_18_2 = tf.keras.layers.Dropout(rate = drop_out_rate)(conv_17_2)
    conv_19_2 = tf.keras.layers.Conv3D(filters = 1024,
                                kernel_size = (5,5,5),
                                strides=(2,2,2),
                                padding='same',
                                activation = tf.nn.leaky_relu)(dropout_18_2)
    dropout_20_2 = tf.keras.layers.Dropout(rate = drop_out_rate)(conv_19_2)
    
    
    ################################### END ###################################
    concat_21 = tf.keras.layers.Concatenate(axis = -1)([pool_16,dropout_20_2])
    
    
    avgp = tf.keras.layers.AveragePooling3D(padding = 'same',
                                            pool_size=(concat_21.shape[1],
                                                       concat_21.shape[2],
                                                       concat_21.shape[3]))(concat_21)
        
    fl = tf.keras.layers.Flatten()(avgp)
        
    de = tf.keras.layers.Dense(1, activation=tf.nn.leaky_relu, use_bias = False)(fl)

        
    model = tf.keras.Model(inputs=[inputs, sexs], outputs=de)
    return model

'''
model = generate_model(0.4)
model.build([(95,79,95,1),(1)])

t = model.trainable_weights
a = []
for idx in range(36):
    a.append(t[idx]g.numpy())
'''