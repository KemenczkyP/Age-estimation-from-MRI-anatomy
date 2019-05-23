    # -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 08:51:55 2019

@author: KemyPeti
"""
from __future__ import absolute_import, division, print_function

import tensorflow as tf
#tf.enable_eager_execution()

import sys
import os

import preprocess as PP
import training_class as TC
#%%
def main():
    '''
    # preprocess
    # RUN THIS 1*
    #read nifti files and their labels(age, sex) and save them as TFRecords
    preprocesser = PP.preprocess_(MRI_volume_size = [79,95,79],
                                  TARGET_DIR_NPY = 'TEMP_DATA/npy_folder',
                                  TARGET_DIR_TFR = 'TEMP_DATA/tf_record_folder/TENFOLD/',
                                  split_TRAIN_VALID = 0)
    
    
    
    
    preprocesser.PROCESS_1_CREATE_TFRECORD('valami')
    
    preprocesser.PROCESS_2_USE_DATA(connectomes1000_project = False,
                                    SALD_project = False,
                                    ADNI = False,
                                    MTA = True,
                                    MIGRAINE_True = False,
                                    MIGRAINE_False = False) #concat "True" datasets and save them into 1 TFRecord
    
    preprocesser.PROCESS_3_SAVE()
    preprocesser.PROCESS_4_CLOSE_TFRECORD()
    '''
    #%%
    sys.path
    sys.path.append(os.getcwd() +'\\networks')
    import net6 as ANN
    AE_CNN = TC.Age_estim_CNN(NETWORK_STRUCTURE = ANN.net,
                              MRI_volume_size = [79,95,79],
                              TFR_DATA_DIR = 'TEMP_DATA\\tf_record_folder\\TENFOLD\\',
                              MODEL_DATA_DIR = '\\tmp\\mnist_convnet_model\\2018_05_15\\PRETRAIN\\',
                              MODEL_NAME = 'PUBLIC',
                              batch_size = 6,
                              TFRec_train_name = 'ALL_PUBLIC_data',
                              TFRec_valid_name = 'MTA_ALL_retrain_data',
                              TFRec_test_name = 'MTA_ALL_retrain_data',
                              CHECKPOINT_save_checkpoints_secs = 3600,
                              CHECKPOINT_keep_checkpoint_max = 20,
                              DROPOUT_RATE = 0.25,
                              LEARNING_RATE_init_lr = 0.0001,
                              LEARNING_RATE_decay_steps = 100000,
                              LEARNING_RATE_decay_rate = 0.96)
    
    AE_CNN.TRAIN(epoch=120,valid_by_epoch = 5)# monitoring valid results in every 'valid_by_epoch' epochs
    #AE_CNN.TEST()# monitoring valid results in every 'valid_by_epoch' epochs
    
#%%
if __name__== "__main__":
    main()