import tensorflow as tf
import os
import matplotlib.pyplot as plt

def get_dataset_ready(path,
                      batch = 5,
                      shuffle_buffer = 100,
                      epochs = 70,
                      ):
    

    keys_to_features = {
                    'image/encoded': tf.io.FixedLenFeature((), tf.string, default_value=''),
                    'image/label': tf.io.FixedLenFeature([], dtype=tf.float32, default_value=-1),
                    'image/width': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
                    'image/height': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
                    'image/sex': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
                    'image/depth': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1),
                    'image/channels': tf.io.FixedLenFeature([], dtype=tf.int64, default_value=-1) 
            }
    
    
    dataset = tf.data.TFRecordDataset([path])
    dataset = dataset.repeat(epochs)         #epoch_size
    #dataset = dataset.shuffle(shuffle_buffer)    #shuffle_buffer_size
    dataset = dataset.batch(batch)          #batch size
    dataset = dataset.prefetch(1)       #buffer_size
    return dataset, keys_to_features

'''
dset, keys = get_dataset_ready(os.getcwd() + '\\Mig_N_data',
                               batch = 5,
                               shuffle_buffer = 100,
                               epochs = 70,
                               )

for serialized_examples in dset:
    parsed = tf.io.parse_example(serialized_examples,
                                 keys)
    image =  tf.io.decode_raw(parsed['image/encoded'],
                              out_type=tf.float32, #the decode type have to be the same as the input type!!!
                              little_endian=True)
    image = tf.reshape(image, (-1,79,95,79))
'''