# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 08:51:55 2019

@author: KemyPeti
"""
import sys
import os

import preprocess as PP
import training_class as TC

def main():
    #%% preprocess
    # RUN THIS 1*
    '''
    #read nifti files and their labels(age, sex) and save them as TFRecords
    preprocesser = PP.preprocess_(MRI_volume_size = [79,95,79],
                                  TARGET_DIR_NPY = 'TEMP_DATA/npy_folder',
                                  TARGET_DIR_TFR = 'TEMP_DATA/tf_record_folder',
                                  split_TRAIN_VALID = 0.88)
    preprocesser.PROCESS_1_CREATE_TFRECORD('NNtrain')
    preprocesser.PROCESS_2_USE_DATA(connectomes1000_project = True,
                                    SALD_project = True,
                                    MTA = False,
                                    MIGRAINE_True = False,
                                    MIGRAINE_False = False) #concat "True" datasets and save them into 1 TFRecord
    preprocesser.PROCESS_3_SAVE()
    preprocesser.PROCESS_4_CLOSE_TFRECORD()
    '''
    #%%
    sys.path
    sys.path.append(os.getcwd() +'\\networks')
    import net3 as ANN
            
    AE_CNN = TC.Age_estim_CNN(NETWORK_STRUCTURE = ANN.net,
                              MRI_volume_size = [79,95,79],
                              TFR_DATA_DIR = 'TEMP_DATA/tf_record_folder',
                              MODEL_DATA_DIR = '\\tmp\\mnist_convnet_model\\model_2019_04_01',
                              batch_size = 4,
                              TFRec_train_name = 'NNtrain_TRAIN_data',
                              TFRec_valid_name = 'NNtrain_VALID_data',
                              TFRec_test_name = 'NNtrain_TEST_data',
                              CHECKPOINT_save_checkpoints_secs = 3600,
                              CHECKPOINT_keep_checkpoint_max = 20,
                              DROPOUT_RATE = 0.25,
                              LEARNING_RATE_init_lr = 0.001,
                              LEARNING_RATE_decay_steps = 100000,
                              LEARNING_RATE_decay_rate = 0.96)
    
    AE_CNN.TRAIN(epoch = 200,
                 valid_by_epoch = 2)# monitoring valid results in every 'valid_by_epoch' epochs
    #%%
if __name__== "__main__":
  main()