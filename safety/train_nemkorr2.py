# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 2018
@author: MRegina, CTO at OrthoPred Ltd
# TFRecords reader and interpreter code for Numpy array data (specifically for 3D arrays with an additional channel dimension).
# The source code is partially based on TensorFlow Authors' imagenet_main.py:
# https://github.com/tensorflow/models/blob/r1.4.0/official/resnet/imagenet_main.py
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import sys
import threading

import os

tf.logging.set_verbosity(tf.logging.INFO)

class Age_estim_CNN():
    def __init__(self,
                 NETWORK_STRUCTURE,
                 MRI_volume_size = [79,95,79],
                 TFR_DATA_DIR = 'output',
                 MODEL_DATA_DIR = 'train_model',
                 batch_size = 4,
                 num_class = 1,#regression
                 TFRec_train_name = 'NNtrain_TRAIN_data',
                 TFRec_valid_name = 'NNtrain_VALID_data',
                 TFRec_test_name = 'NNtrain_TEST_data',
                 CHECKPOINT_save_checkpoints_secs = 3600,
                 CHECKPOINT_keep_checkpoint_max = 20,
                 DROPOUT_RATE = 0.25,
                 LEARNING_RATE_init_lr = 0.005,
                 LEARNING_RATE_decay_steps = 100000,
                 LEARNING_RATE_decay_rate = 0.96):
        '''
        LOSS: MSE
        OPTIMIZER: MomentumOptimizer
        LEARNING RATE: exponential_decay
        
        
        '''
        self.NETWORK_STRUCTURE = NETWORK_STRUCTURE #neural_network
        
        self.var_MRI_volume_size = MRI_volume_size
        self.var_TFR_DATA_DIR = TFR_DATA_DIR
        self.var_MODEL_DATA_DIR = MODEL_DATA_DIR
        self.var_batch_size = batch_size
        self.var_num_class = num_class
        self.var_DROPOUT_RATE = DROPOUT_RATE
        self.var_LEARNING_RATE_init_lr = LEARNING_RATE_init_lr
        self.var_LEARNING_RATE_decay_steps = LEARNING_RATE_decay_steps
        self.var_LEARNING_RATE_decay_rate = LEARNING_RATE_decay_rate
        
        self.var_TFRec_train_name = TFRec_train_name
        self.var_TFRec_valid_name = TFRec_valid_name
        self.var_TFRec_test_name = TFRec_test_name
        
        #GPU config
        self.config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8 
        
        self.my_checkpointing_config = tf.estimator.RunConfig(save_checkpoints_secs = CHECKPOINT_save_checkpoints_secs, #saves checkpoints in every 'CHECKPOINT_save_checkpoints_secs' seconds
                                                              keep_checkpoint_max = CHECKPOINT_keep_checkpoint_max) #keep the last 'CHECKPOINT_keep_checkpoint_max' checkpoints
                                   
        print('\nTensorboard client will be ready after starting the TRAIN phase! USE: http://localhost:6006\n')
    def TRAIN(self,
              epoch = 1,
              valid_by_epoch = 1 # monitoring valid results in every 'valid_by_epoch' epochs
              ):
        
        
        with tf.Session(config=self.config) as sess:
            
            self.train_epoch = int(epoch/valid_by_epoch) # for loop limit
            self.train_input_epoch = valid_by_epoch      # input fn epoch
            
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            
            '''
            images, labels = input_valid_fn()
            print("returned")
            #evaluate image tensors to get actual numbers to show
            imgs = images['x'].eval(session = sess)
            sexs = images['s'].eval(session = sess)
            lbls = labels.eval(session = sess)
            '''
            mnist_classifier = tf.estimator.Estimator(model_fn=self._optimizer_,
                                                      model_dir=os.getcwd() + self.var_MODEL_DATA_DIR,
                                                      config=self.my_checkpointing_config)
            
            logging_hook_train = tf.train.LoggingTensorHook(tensors={"loss": "loss"},
                                                            every_n_iter=20)

            logging_hook_valid = tf.train.LoggingTensorHook(tensors={"labels": "labs",
                                                                     "predictions": "pred",
                                                                     "sexs": "sexs"},
                                                            every_n_iter=1)
            
            for idx in range(self.train_epoch):
                
                mnist_classifier.train(input_fn=self.input_fn,            
                                       hooks=[logging_hook_train])
                
                mnist_classifier.evaluate(input_fn=self.input_valid_fn,            
                                          hooks=[logging_hook_valid])
                
                
                
    def _optimizer_(self, features, labels, mode):
        """Model function for CNN."""
        with tf.device('/gpu:0'):
            
            labels = labels
            sexs = tf.cast(features["s"],dtype=tf.float32,name='sexs')
            
            input_layer = tf.reshape(features["x"], [-1,                          #batch
                                                     self.var_MRI_volume_size[0], #79
                                                     self.var_MRI_volume_size[1], #95
                                                     self.var_MRI_volume_size[2], #79
                                                     1])                          #depth
            
            if mode == tf.estimator.ModeKeys.TRAIN:
                drop_out_rate = self.var_DROPOUT_RATE
            else:
                drop_out_rate = 0
            
            logits = self.NETWORK_STRUCTURE(input_layer, sexs, self.var_num_class, drop_out_rate, training = 1)
        
            tf.identity(tf.transpose(logits),name = "pred")
            tf.identity(tf.transpose(labels),name = "labs")
            
            labels = tf.cast(labels,dtype=tf.float64,name='labels')
            logits = tf.cast(logits,dtype=tf.float64,name='logits')
            
            loss = tf.losses.mean_squared_error(labels = labels, predictions=logits)
            
            tf.identity(loss,name = "loss")                     
                
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode=mode, 
                                                  loss=loss)
            
            if mode == tf.estimator.ModeKeys.TRAIN:
                
                global_step=tf.Variable(0,trainable=False)
                learning_rate = tf.train.exponential_decay(self.var_LEARNING_RATE_init_lr,
                                                           global_step,
                                                           self.var_LEARNING_RATE_decay_steps,
                                                           self.var_LEARNING_RATE_decay_rate,
                                                           staircase=True)
                optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                                       momentum = 0.8)
                
                train_op = optimizer.minimize(
                        loss=loss,
                        global_step=tf.train.get_global_step())
                
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        
        
        
            return tf.estimator.EstimatorSpec(
                    mode=mode, loss=loss)

    def input_fn(self):
        """Input function which provides batches."""
        dataset = tf.data.Dataset.from_tensor_slices(self._filenames_([True, False, False], self.var_TFR_DATA_DIR))
        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda value: self._record_parser_(value), num_parallel_calls=5)
        dataset = dataset.shuffle(buffer_size = self.var_batch_size*100)
        dataset = dataset.prefetch(self.var_batch_size)
        
        dataset = dataset.repeat(self.train_input_epoch)
        dataset = dataset.batch(self.var_batch_size)
        iterator = dataset.make_one_shot_iterator()
        images, sex, labels = iterator.get_next()
    
        print("iterator created")
        return {'x': images, 's': sex}, labels/100
        
    
        
    def input_valid_fn(self, num_epochs=1):
        """Input function which provides batches."""
        dataset = tf.data.Dataset.from_tensor_slices(self, self._filenames_([False,False,True], self.var_TFR_DATA_DIR))
        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.map(lambda value: self._record_parser_(value), num_parallel_calls=5)
        dataset = dataset.shuffle(buffer_size = self.var_batch_size*100)
        dataset = dataset.prefetch(self.var_batch_size)
        
        dataset = dataset.repeat(1)
        dataset = dataset.batch(self.var_batch_size)
        iterator = dataset.make_one_shot_iterator()
        images, sex, labels = iterator.get_next()
    
        print("iterator created")
        return {'x': images, 's': sex}, labels/100
        
    def _filenames_(self, TRAIN_VALID_TEST, data_dir):
        """Return filenames for dataset."""
        if TRAIN_VALID_TEST[0]:
          return [
              os.path.join(data_dir, self.var_TFRec_train_name)
              for i in range(1)]
        if TRAIN_VALID_TEST[2]:
            return [
              os.path.join(data_dir, self.var_TFRec_test_name)
              for i in range(1)]
        if TRAIN_VALID_TEST[1]:
          return [
              os.path.join(data_dir, self.var_TFRec_valid_name)
              for i in range(1)]
        else:
            raise('Each input is "False" !')
    
    def _record_parser_(self, value):
        """Parse a TFRecord file from `value`."""
     
        keys_to_features = {
                'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
                'image/label': tf.FixedLenFeature([], dtype=tf.float32, default_value=-1),
                'image/width': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
                'image/height': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
                'image/sex': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
                'image/depth': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
                'image/channels': tf.FixedLenFeature([], dtype=tf.int64, default_value=-1) 
        }
    
        parsed = tf.parse_single_example(value, keys_to_features)
        print("parsed one example")
        
        #decode label
        label = tf.cast(tf.reshape(parsed['image/label'], shape=[-1]),dtype=tf.float32)
        sex = tf.cast(tf.reshape(parsed['image/sex'], shape=[-1]),dtype=tf.int32)
        
        #decode the array shape
        width = tf.cast(tf.reshape(parsed['image/width'], shape=[]),dtype=tf.int32)
        height = tf.cast(tf.reshape(parsed['image/height'], shape=[]),dtype=tf.int32)
        depth = tf.cast(tf.reshape(parsed['image/depth'], shape=[]),dtype=tf.int32)
        channels = tf.cast(tf.reshape(parsed['image/channels'], shape=[]),dtype=tf.int32)
        
        #decode 3D array data (important to check data type and byte order)
        image = tf.reshape(tf.decode_raw(parsed['image/encoded'],out_type=tf.float32,little_endian=True),shape=[height,width,depth,channels])
            
        print("image decoded")
        
        return image, sex, label
            
    def launchTensorBoard(self):
        import os
        
        os.system('python -m tensorboard.main --logdir=' + os.getcwd() + self.var_MODEL_DATA_DIR)
        return
        
        
sys.path
sys.path.append(os.getcwd() +'\\networks')
import net3 as ANN
        
AE_CNN = Age_estim_CNN(NETWORK_STRUCTURE = ANN.net,
                       MRI_volume_size = [79,95,79],
                       TFR_DATA_DIR = 'TEMP_DATA/tf_record_folder',
                       MODEL_DATA_DIR = '\\tmp\\mnist_convnet_model\\model_2019_04_01',
                       batch_size = 4,
                       num_class = 1,
                       TFRec_train_name = 'NNtrain_TRAIN_data',
                       TFRec_valid_name = 'NNtrain_VALID_data',
                       TFRec_test_name = 'NNtrain_TEST_data',
                       CHECKPOINT_save_checkpoints_secs = 3600,
                       CHECKPOINT_keep_checkpoint_max = 20,
                       DROPOUT_RATE = 0.25,
                       LEARNING_RATE_init_lr = 0.001,
                       LEARNING_RATE_decay_steps = 100000,
                       LEARNING_RATE_decay_rate = 0.96)

AE_CNN.TRAIN()
