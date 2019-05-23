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

import os

tf.logging.set_verbosity(tf.logging.INFO)


_DATA_DIR= '.\\output\\'
batch_size=4
num_class = 1

num_epochs=0

# train data: is_training=True is_testing=False
# validation data: is_training=False is_testing=False
# test data: is_training=False is_testing=True

#%%
#{
#}
def cnn_model_fn(features, labels, mode):
    import tensorflow as tf
    """Model function for CNN."""
    with tf.device('/gpu:0'):
        labels = labels

        sexs = tf.cast(features["s"],dtype=tf.float32,name='sexs')
        input_layer = tf.reshape(features["x"], [-1, 79,95,79, 1])
        
        sys.path
        sys.path.append(os.getcwd() +'\\networks')
        import net2
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            drop_out_rate = 0.25
        else:
            drop_out_rate = 0
        
        logits = net2.net2(input_layer, sexs, num_class, drop_out_rate,training = 1)
    
        tf.identity(tf.transpose(logits),name = "pred")
        tf.identity(tf.transpose(labels),name = "labs")
        
        labels = tf.cast(labels,dtype=tf.float64,name='labels')
        logits = tf.cast(logits,dtype=tf.float64,name='logits')
        
        loss = tf.losses.mean_squared_error(labels = labels, predictions=logits)
        
        
        
        tf.identity(loss,name = "loss")        
        predictions = {
                "classes": tf.argmax(input=logits, axis=1,name="classes"),
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")}
    
          
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode=mode, 
                                              predictions=predictions,
                                              loss=loss)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            global_step=tf.Variable(0,trainable=False)
            learning_rate=tf.train.linear_cosine_decay(0.000001,global_step,5000)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            
            train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())
            
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    
    
        return tf.estimator.EstimatorSpec(
                mode=mode, loss=loss)

#%%
def record_parser(value):
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

#%%

def filenames(is_training, is_testing, data_dir):
    """Return filenames for dataset."""
    if is_training:
      return [
          os.path.join(data_dir, 'train_data')
          for i in range(1)]
    if is_testing:
        return [
          os.path.join(data_dir, 'test_data')
          for i in range(1)]
    else:
      return [
          os.path.join(data_dir, 'validation_data')
          for i in range(1)]

def input_fn(data_dir = _DATA_DIR, batch_size=batch_size, num_epochs=1):
    """Input function which provides batches."""
    dataset = tf.data.Dataset.from_tensor_slices(filenames(True,False, data_dir))
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(lambda value: record_parser(value), num_parallel_calls=5)
    dataset = dataset.shuffle(buffer_size = batch_size*100)
    dataset = dataset.prefetch(batch_size)
    
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, sex, labels = iterator.get_next()

    print("iterator created")
    return {'x': images, 's': sex}, labels/100
    
#%%
def input_valid_fn(data_dir = _DATA_DIR, batch_size=batch_size, num_epochs=1):
    """Input function which provides batches."""
    dataset = tf.data.Dataset.from_tensor_slices(filenames(False,True, data_dir))
    dataset = dataset.flat_map(tf.data.TFRecordDataset)
    dataset = dataset.map(lambda value: record_parser(value), num_parallel_calls=5)
    dataset = dataset.shuffle(buffer_size = batch_size*100)
    dataset = dataset.prefetch(batch_size)
    
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    images, sex, labels = iterator.get_next()

    print("iterator created")
    return {'x': images, 's': sex}, labels/100


#%%

# start a session to plot some data slices
config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.8
with tf.Session(config=config) as sess:
    
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    
    images, labels = input_valid_fn(_DATA_DIR,batch_size)
    print("returned")
    
    #evaluate image tensors to get actual numbers to show
    imgs = images['x'].eval(session = sess)
    sexs = images['s'].eval(session = sess)
    lbls = labels.eval(session = sess)
    
    #%%
    
    my_checkpointing_config = tf.estimator.RunConfig(
            save_checkpoints_secs = 3600, #saves checkpoints in every 2 seconds
            keep_checkpoint_max = 20)
    
    mnist_classifier = tf.estimator.Estimator(
            model_fn=cnn_model_fn,
            model_dir=os.getcwd() +"/tmp/mnist_convnet_model\\train_modifiedNET_6",
            config=my_checkpointing_config)
    
#%%   
    testing_eval = np.array([])
    for idx in range(120):
        
        tensors_to_log = {"loss": "loss"}
        
        logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=20)
        
        mnist_classifier.train(
                input_fn=input_fn,            
                hooks=[logging_hook])#         
                               
        logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=1)
        
        eval_results = mnist_classifier.evaluate(input_fn=input_valid_fn,            
                                                 hooks=[logging_hook])
        
        testing_eval = np.append(testing_eval,eval_results)
   
