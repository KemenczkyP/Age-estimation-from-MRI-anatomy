# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 08:51:55 2019

@author: KemenczkyP
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
    sys.path
    sys.path.append(os.getcwd() + '\\networks')

    AE_CNN = TC.Age_estim_CNN(NETWORK_STRUCTURE=ANN.net,
                              MRI_volume_size=[79, 95, 79],
                              TFR_DATA_DIR='TEMP_DATA\\tf_record_folder\\2019_09_11\\',
                              MODEL_DATA_DIR='\\tmp\\mnist_convnet_model\\thesis\\',
                              MODEL_NAME='net3_retrain4_justCNNlFNN_10eps',
                              batch_size=5,
                              TFRec_train_name='MTA_data_TRAIN_data',
                              TFRec_valid_name='MTA_data_VALID_data',
                              TFRec_test_name='Mig_N_data',
                              CHECKPOINT_save_checkpoints_secs=3600,
                              CHECKPOINT_keep_checkpoint_max=20,
                              DROPOUT_RATE=0,
                              LEARNING_RATE_init_lr=0.000001,
                              LEARNING_RATE_decay_steps=80000,
                              LEARNING_RATE_decay_rate=0.98)

    AE_CNN.TRAIN(epoch=10,
                 valid_by_epoch=1,# monitoring valid results in every 'valid_by_epoch' epochs
                 init_again=0)
    AE_CNN.TEST()  # monitoring valid results in every 'valid_by_epoch' epochs


# %%
if __name__ == "__main__":
    main()