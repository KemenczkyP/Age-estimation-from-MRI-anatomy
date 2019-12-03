from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import sys
import random
import argparse

sys.path.append(os.getcwd() + '\\preprocessing')
from preprocessing.preprocessing import preprocess_

parser = argparse.ArgumentParser(description='Brain ageing preprocess flags')
parser.add_argument('--TARGET_DIR_NPY',
                    default= os.getcwd() + '/DATA/TEMP_DATA/npy_folder',
                    type=str,
                    help='Directory path storing numpy files temporarily')
parser.add_argument('--TARGET_DIR_TFR',
                    default= os.getcwd() + '/DATA/TFRecords',
                    type=str,
                    help='Directory path storing tfrecords')

args = parser.parse_args()
sys.path.append(os.getcwd() + '\\UTILS\\')

import numpy as np
import six
import tensorflow as tf
import nibabel as nib
import pandas as pd
import scipy
import time

import input_utils as IU
import path_utils as PU



# preprocess
# RUN THIS 1*
# read nifti files and their labels(age, sex) and save them as TFRecords
preprocesser = preprocess_(mri_volume_size=[79, 95, 79],
                           split_train_valid=[0.33, 0.66],
                           flags=args)  # [0.33, 0.66])

preprocesser.PROCESS_1_CREATE_TFRECORD('Test_N_2asfasfasdf')

l = preprocesser.PROCESS_2_USE_DATA(connectomes1000_project=True,
                                    SALD_project=True,
                                    ADNI=True,
                                    MTA=False,
                                    MIGRAINE_True=False,
                                    MIGRAINE_False=False,
                                    BIOBANK=False,
                                    MR_ART=False)  # concat "True" datasets and save them into 1 TFRecord

preprocesser.PROCESS_3_SAVE()
preprocesser.PROCESS_4_CLOSE_TFRECORD()