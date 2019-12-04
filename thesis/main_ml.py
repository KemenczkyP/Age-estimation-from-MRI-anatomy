# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 08:51:55 2019

@author: KemenczkyP

The script uses TFRecord files converted with main_preprocess.py
"""
from __future__ import absolute_import, division, print_function


import sys
import os

sys.path.append(os.getcwd()+'\\machine_learning')
sys.path.append(os.getcwd()+'\\networks')

from machine_learning import training_class as TC
from networks import net3 as ANN

# %%
def main():
    AE_CNN = TC.Age_estim_CNN(NETWORK_STRUCTURE=ANN.net, # tensorflow neural network function
                              MRI_volume_size=[79, 95, 79], # default MRI volume size
                              TFR_DATA_DIR='DATA\\TFRecords\\', # TFRecord container dir.
                              MODEL_DATA_DIR='\\saved_models\\', # tf model save path
                              MODEL_NAME='net3_pre', # model name
                              batch_size=5,
                              TFRec_train_name='PUBLIC_TRAIN_data',
                              TFRec_valid_name='PUBLIC_VALID_data',
                              TFRec_test_name='PUBLIC_VALID_data',
                              CHECKPOINT_save_checkpoints_secs=3600, # checkpoint saver
                              CHECKPOINT_keep_checkpoint_max=20,
                              DROPOUT_RATE=0.4,
                              LEARNING_RATE_init_lr=0.000001, # initial learning rate
                              LEARNING_RATE_decay_steps=80000,
                              LEARNING_RATE_decay_rate=0.98)

    AE_CNN.TRAIN(epoch=70,
                 valid_by_epoch=1)# monitoring valid results in every 'valid_by_epoch' epochs
    AE_CNN.TEST()  # monitoring valid results in every 'valid_by_epoch' epochs

if __name__ == "__main__":
    main()