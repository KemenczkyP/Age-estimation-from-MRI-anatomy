import tensorflow as tf

print("Tensorflow version: {}".format(tf.__version__))


def generate_model(drop_out_rate):
    inputs = tf.keras.Input(shape=(79, 95, 79, 1))
    sexs = tf.keras.Input(shape=(1))

    ############################### CONV 1 ####################################

    conv_1 = tf.keras.layers.Conv3D(filters=8,
                                    kernel_size=(7, 7, 7),
                                    strides=(1, 1, 1),
                                    padding='same',
                                    activation=tf.nn.leaky_relu)(inputs)
    pool_2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2),
                                       strides=(2, 2, 2),
                                       padding='same')(conv_1)
    dropout_3 = tf.keras.layers.Dropout(rate=drop_out_rate)(pool_2)

    ############################ INCEPTION 1 ##################################
    conv_4_1 = tf.keras.layers.Conv3D(filters=32,
                                      kernel_size=(1, 1, 1),
                                      strides=(1, 1, 1),
                                      padding='same',
                                      activation=tf.nn.leaky_relu)(dropout_3)
    dropout_4_1 = tf.keras.layers.Dropout(rate=drop_out_rate)(conv_4_1)

    conv_4_2 = tf.keras.layers.Conv3D(filters=32,
                                      kernel_size=(3, 3, 3),
                                      strides=(1, 1, 1),
                                      padding='same',
                                      activation=tf.nn.leaky_relu)(dropout_3)
    dropout_4_2 = tf.keras.layers.Dropout(rate=drop_out_rate)(conv_4_2)

    conv_4_3 = tf.keras.layers.Conv3D(filters=32,
                                      kernel_size=(5, 5, 5),
                                      strides=(1, 1, 1),
                                      padding='same',
                                      activation=tf.nn.leaky_relu)(dropout_3)
    dropout_4_3 = tf.keras.layers.Dropout(rate=drop_out_rate)(conv_4_3)

    maxpooling_4_4 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2),
                                               strides=(2, 2, 2),
                                               padding='same')(dropout_3)

    deconv_4_4_2 = tf.keras.layers.Conv3DTranspose(filters=16,
                                                   kernel_size=(5, 5, 5),
                                                   strides=(2, 2, 2),
                                                   padding='same',
                                                   activation=tf.nn.leaky_relu)(maxpooling_4_4)

    batchnorm_6 = tf.keras.layers.Concatenate(axis=-1)([dropout_4_1, dropout_4_2, dropout_4_3, deconv_4_4_2])

    ############################ INCEPTION 2 ##################################
    conv_7_1 = tf.keras.layers.Conv3D(filters=128,
                                      kernel_size=(1, 1, 1),
                                      strides=(1, 1, 1),
                                      padding='same',
                                      activation=tf.nn.leaky_relu)(batchnorm_6)
    dropout_7_1 = tf.keras.layers.Dropout(rate=drop_out_rate)(conv_7_1)
    pool_8_1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2),
                                         strides=(2, 2, 2),
                                         padding='same')(dropout_7_1)
    dropout_8_1 = tf.keras.layers.Dropout(rate=drop_out_rate)(pool_8_1)

    conv_7_2 = tf.keras.layers.Conv3D(filters=128,
                                      kernel_size=(1, 1, 1),
                                      strides=(1, 1, 1),
                                      padding='same',
                                      activation=tf.nn.leaky_relu)(batchnorm_6)
    dropout_7_2 = tf.keras.layers.Dropout(rate=drop_out_rate)(conv_7_2)
    conv_8_2 = tf.keras.layers.Conv3D(filters=128,
                                      kernel_size=(3, 3, 3),
                                      strides=(2, 2, 2),
                                      padding='same',
                                      activation=tf.nn.leaky_relu)(dropout_7_2)
    dropout_8_2 = tf.keras.layers.Dropout(rate=drop_out_rate)(conv_8_2)

    conv_7_3 = tf.keras.layers.Conv3D(filters=128,
                                      kernel_size=(1, 1, 1),
                                      strides=(1, 1, 1),
                                      padding='same',
                                      activation=tf.nn.leaky_relu)(batchnorm_6)
    dropout_7_3 = tf.keras.layers.Dropout(rate=drop_out_rate)(conv_7_3)

    conv_8_3 = tf.keras.layers.Conv3D(filters=128,
                                      kernel_size=(5, 5, 5),
                                      strides=(2, 2, 2),
                                      padding='same',
                                      activation=tf.nn.leaky_relu)(dropout_7_3)
    dropout_8_3 = tf.keras.layers.Dropout(rate=drop_out_rate)(conv_8_3)

    conv_78_4 = tf.keras.layers.Conv3D(filters=64,
                                       kernel_size=(3, 3, 3),
                                       strides=(2, 2, 2),
                                       padding='same',
                                       activation=tf.nn.leaky_relu)(batchnorm_6)

    batchnorm_10 = tf.keras.layers.Concatenate(axis=-1)([dropout_8_1, dropout_8_2, dropout_8_3, conv_78_4])
    batchnorm_10 = tf.keras.layers.BatchNormalization()(batchnorm_10)

    avgp = tf.keras.layers.AveragePooling3D(padding='same',
                                            pool_size=(batchnorm_10.shape[1],
                                                       batchnorm_10.shape[2],
                                                       batchnorm_10.shape[3]))(batchnorm_10)

    fl = tf.keras.layers.Flatten()(avgp)

    de = tf.keras.layers.Dense(1, activation=tf.nn.leaky_relu, use_bias=False)(fl)

    model = tf.keras.Model(inputs=[inputs, sexs], outputs=de)
    return model


