# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 15:01:03 2019

@author: KemenczkyP
"""
from __future__ import absolute_import, division, print_function
    
import numpy as np
import tensorflow as tf
tf.random.set_seed(0)

import sys
import os

sys.path.append(os.getcwd()+'\\code\\')

import TFFuncLib as TFEC
import argparse
import network 
import printer_funs as prtf
#%%
parser = argparse.ArgumentParser()

parser.add_argument('--MRI_volume_size', default = [79,95,79])
parser.add_argument('--TFR_folder', default = "D:\\Peti\\Aging\\TEMP_DATA\\tf_record_folder", type = str)
parser.add_argument('--checkpoint_dir', default = "D:\\Peti\\Aging\\tf2\\TensorflowImageToTFRecords-master\\TensorflowImageToTFRecords-master\\saved_model\\net3_08_01", type = str)
parser.add_argument('--checkpoint_name', default = "/cp-{step:08d}.ckpt", type = str)

parser.add_argument('--TRAIN_TFR', default = "PUBLIC_TRAIN", type = str)
parser.add_argument('--TEST_TFR', default = "PUBLIC_TEST", type = str)

parser.add_argument('--batch_size', default = 3)
parser.add_argument('--max_epochs', default = 120)
parser.add_argument('--learning_rate', default = 0.0001)
parser.add_argument('--dropout_rate', default = 0.4)

parser.add_argument('--optimizer', default = 'adam', type = str)
parser.add_argument('--loss', default = 'mse', type = str)

#dataset manip
parser.add_argument('--shuffle_buffer_size', default = 100)
parser.add_argument('--prefetch_buffer_size', default = 1)



args = parser.parse_args()

#%%
model = network.net3()

if(args.loss == 'mse'):
    loss = tf.keras.losses.mse
if(args.optimizer == 'adam'):
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)



TFRecReader = TFEC.TFRecordReader.read_keys(path = args.TFR_folder,
                                            filename = args.TRAIN_TFR)
dataset = TFRecReader.get_dataset()

TFRecReader = TFEC.TFRecordReader.read_keys(path = args.TFR_folder,
                                            filename = args.TEST_TFR)
dataset_test = TFRecReader.get_dataset()


#------------------------CALL THE EXAMPLES FROM THE DATASET-------------------#

dataset = dataset.repeat(args.max_epochs)
dataset = dataset.shuffle(args.shuffle_buffer_size)
dataset = dataset.batch(args.batch_size)
dataset = dataset.prefetch(args.prefetch_buffer_size)

dataset_test = dataset_test.batch(args.batch_size)
#%%
current_step = 0
for serialized in dataset:
    parsed = TFEC.TFRecordReader.parse_examples(serialized)
    image =  tf.io.decode_raw(parsed['image/encoded'],
                              out_type=tf.float32, #the decode type have to be the same as the input type!!!
                              little_endian=True)
    image = tf.keras.backend.reshape(image, (image.shape[0],
                                             args.MRI_volume_size[0],
                                             args.MRI_volume_size[1],
                                             args.MRI_volume_size[2],
                                             1))
    ## BACKPROPAGATION
    with tf.GradientTape() as tape:
        net_out = model(inputs = image,
                        sex = tf.keras.backend.reshape(tf.keras.backend.cast(parsed['sex'], tf.float32),(args.batch_size,1)),
                        dropout_rate = args.dropout_rate)
        train_loss = tf.keras.backend.mean(loss(net_out, parsed['age']))
    grads = tape.gradient(train_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
    #PRINT Model summary in the first step
    if(current_step == 0):
        model.summary()
    
    #PRINT validation data in every 1000th step
    if(current_step % 1000 == 0):
        net_out = model(inputs = image,
                        sex = tf.keras.backend.reshape(tf.keras.backend.cast(parsed['sex'], tf.float32),(parsed['sex'].shape[0],1)),
                        dropout_rate = args.dropout_rate)
        train_loss = tf.keras.backend.mean(loss(net_out, parsed['age']))
        
        test_loss = []
        for serialized_t in dataset_test:
            parsed_t = TFEC.TFRecordReader.parse_examples(serialized_t)
            image_t =  tf.io.decode_raw(parsed_t['image/encoded'],
                                        out_type=tf.float32, #the decode type have to be the same as the input type!!!
                                        little_endian=True)
            image_t = tf.keras.backend.reshape(image_t, (image_t.shape[0],
                                                         args.MRI_volume_size[0],
                                                         args.MRI_volume_size[1],
                                                         args.MRI_volume_size[2],
                                                         1))
            net_out = model(inputs = image_t,
                        sex = tf.keras.backend.reshape(tf.keras.backend.cast(parsed_t['sex'], tf.float32),(parsed_t['sex'].shape[0],1)),
                        dropout_rate = 0)
            test_loss.append(loss(net_out, parsed['age']).numpy())
    
        t_l = np.array([])
        for idx in range(len(test_loss)):
            for jdx in range(len(test_loss[idx])):
                t_l = np.append(t_l,test_loss[idx][jdx])
            
        print("\nTEST--Step:{0:7f}; Train loss:{1:9f}; Test loss:{2:9f}".format(current_step, train_loss.numpy(),
                                                                              np.mean(t_l)))
        
        model.save_weights(args.checkpoint_dir + args.checkpoint_name.format(step=current_step))

    prtf.percent_printer(iteration = int(current_step /1000),
                         actual_step = current_step,
                         all_step = 1000,
                         string_ = 'TRAIN')
    current_step += 1

