# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:04:29 2018
@author: Kemypeti

@reference: MRegina
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import sys
import random

os.chdir('D:\\Peti\\Aging\\')
sys.path
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

#%%

class preprocess_():
    def __init__(self,
                 MRI_volume_size = [79,95,79],
                 TARGET_DIR_NPY = 'input',
                 TARGET_DIR_TFR = 'output',
                 split_TRAIN_VALID = 0):
        '''
        Creates directory if it does not exist.
        \nINPUT:
            \n\t split_TRAIN_VALID: 0<=split_TRAIN_VALID<1: 
                if 0 no splitting, 
                if bigger, random generate a number, if it is bigger than split_TRAIN_VALID, volume goes to VALID, if not volume goes to TRAIN
        '''
        
        self.var_MRI_volume_size = MRI_volume_size;
        
        self.val_TARGET_DIR_NPY = TARGET_DIR_NPY;
        self._check_dir_existence_(self.val_TARGET_DIR_NPY)
        print('Saving npy files to %s' % (os.getcwd() + '\\' + self.val_TARGET_DIR_NPY + '\\'))
        
        self.var_TARGET_DIR_TFR = TARGET_DIR_TFR;
        self._check_dir_existence_(self.var_TARGET_DIR_TFR)
        print('Saving TFRecord files to %s' % (os.getcwd() + '\\' + self.var_TARGET_DIR_TFR + '\\'))
        
        self.var_split_TRAIN_VALID = split_TRAIN_VALID;
        
    def PROCESS_1_CREATE_TFRECORD(self, tfrecord_filename):
        if (self.var_split_TRAIN_VALID == 0):
            self.writer = self._generate_TFRecord_(tfrecord_filename)
        else:
            self.writer_TRAIN = self._generate_TFRecord_(tfrecord_filename + "_TRAIN")
            self.writer_VALID = self._generate_TFRecord_(tfrecord_filename + "_VALID")
    
    def PROCESS_2_USE_DATA(self, connectomes1000_project = True,
                           SALD_project = False,
                           ADNI = False,
                           MTA = False,
                           MIGRAINE_True = False,
                           MIGRAINE_False = False):
        
        self.var_data_usage = np.array([connectomes1000_project, 
                                        SALD_project,
                                        ADNI,
                                        MTA,
                                        MIGRAINE_True,
                                        MIGRAINE_False])
        
        read_file = np.where(self.var_data_usage == 1)[0]
        functions = [IU.manip.scan_1000_c,
                     IU.manip.scan_kinai,
                     IU.manip.scan_ADNI,
                     IU.manip.scan_MTA,
                     IU.manip.scan_migrene_True,
                     IU.manip.scan_migrene_False]
                              
        if(len(read_file)>1):
            arr = []
            for idx in range(len(read_file)):
                arr.append(functions[read_file[idx]]())
            self.labels = IU.manip.arange_dicts(arr)
        elif(len(read_file) == 0):
            print("Each label variable is marked by 'False'! The process will stop. PLEASE USE THE 'PROCESS_2_USE_DATA' FUNCTION AGAIN!")
        elif(len(read_file) == 1):
            self.labels = functions[read_file[0]]()
            
    def PROCESS_3_SAVE(self):
        start = time.time()
        num_of_saved_files = 0
        for idx in range(self.labels['age'].shape[0]):
            
            splited_file_name = self.labels['file_name'][idx].split('.')[0].split('\\')[-1] # split the filename
            
            img = nib.load(self.labels['file_name'][idx]) #get nifti file
            arr = np.array(img.dataobj)
            
            if(arr.shape != (self.var_MRI_volume_size[0],self.var_MRI_volume_size[1],self.var_MRI_volume_size[2])): #if the datasize is incorrect, skip
                string = ('UNABLE TO SAVE: ' + splited_file_name)   
                print(string)
                continue
            arr = self._connectomes1000_noise_correction_(arr)
            
            arr = np.subtract(arr,np.mean(arr)) # norm data
            arr = np.divide(arr,np.std(arr)) # norm data
            
            file_name = os.getcwd() + '\\' + self.val_TARGET_DIR_NPY + '\\'+splited_file_name
            label = self.labels['age'][idx]
            sex = self.labels['sex'][idx]
            np.save(file_name,arr)
            
            image_buffer, height, width, depth = self._process_image_(filename=file_name)
            
            example = self._convert_to_example_(file_name, image_buffer, label, np.int_(sex), height, width, depth)
            
            if(self.var_split_TRAIN_VALID == 0):
                self.writer.write(example.SerializeToString())
            else:
                which_set = random.random()
                if(self.var_split_TRAIN_VALID>=which_set):
                    self.writer_TRAIN.write(example.SerializeToString())
                else:
                    self.writer_VALID.write(example.SerializeToString())
    
            end = time.time()
            string = ('File saved: ' + splited_file_name + ';\t Time:' + str(end-start))
            print(string)
            os.remove(file_name +'.npy')
            num_of_saved_files += 1
        print("\n{0} files saved in TFRecords.".format(num_of_saved_files))
            
    def PROCESS_4_CLOSE_TFRECORD(self):
        if (self.var_split_TRAIN_VALID == 0):
            self.writer.close()
        else:
            self.writer_TRAIN.close()
            self.writer_VALID.close()
            
    def _check_dir_existence_(self, dir_name):
        '''
        Creates directory if it does not exist.
        \nINPUT:
            \n\t dir_name: check "current folder/dir_name/"
        '''
        PU.manip.create_dir_if_not_exists(os.getcwd() + '\\' + dir_name + '\\')
    
    def _generate_TFRecord_(self, filename):
        '''
        Creates directory if it does not exist.
        \nINPUT:
            \n\t dir_name: check "current folder/dir_name/"
        '''
        output_filename = '%s_data' % (filename)
        output_file = os.path.join(self.var_TARGET_DIR_TFR, output_filename)
        return tf.python_io.TFRecordWriter(output_file)
    
    def _connectomes1000_noise_correction_(self, arr):
        # NaN values in corners
     
        while True:
            nans = np.isnan(arr)
            if (np.sum(np.isnan(arr))==0):
                break
            nans_dilated = scipy.ndimage.binary_dilation(nans)
            nans_diff = np.logical_xor(nans_dilated,nans)
            nans_diff_indexes = np.where(nans_diff)
            for ldx in range(nans_diff_indexes[0].shape[0]):
                good_pixels = np.array([])
                for kdx in range(1,8):
                    for jdx in range(3):
                        good_pixels = np.append(good_pixels, nans_diff_indexes[jdx][ldx])
                    
                    if(np.mod(kdx,2) == 0):
                        chindex = -1
                    else:
                        chindex = 1
                    if (kdx>1):
                        if(good_pixels[-(kdx//2)] + chindex < 0 or good_pixels[-(kdx//2)] + chindex > self.var_MRI_volume_size[(kdx//2) -1]-1):
                            good_pixels = np.delete(good_pixels, range(good_pixels.size-3,good_pixels.size) ,0)
                            continue
                        good_pixels[-(kdx//2)] = good_pixels[-(kdx//2)] + chindex
                good_pixels = np.int_(np.reshape(good_pixels, (-1,3)))
                    
                add_=0
                num_=0
                nan_place = []
                for kdx in range(good_pixels.shape[0]):
                    p_v = arr[good_pixels[kdx,0],good_pixels[kdx,1],good_pixels[kdx,2]]
                    if(np.isnan(p_v)):
                        nan_place = np.append(nan_place, good_pixels[kdx,:])
                        continue
                    else:
                        add_ = add_ + p_v
                        num_ = num_ + 1
                if(len(nan_place)==3):
                    arr[np.int_(nan_place[0]),np.int_(nan_place[1]),np.int_(nan_place[2])] = add_/num_
        return arr
    
    def _process_image_(self, filename):
        """Process a single image file.
        Args:
            data_dir: string, root directory of images eg. './data/'
            filename: string, path to an image file in numpy format e.g., 'example.npy'
        Returns:
            image_buffer: string, encoded numpy array (should be 3D + a channel dimension)
            height: integer, image height in voxels
            width: integer, image width in voxels
            depth: integer, image depth in voxels
            num_channels: integer, number of color channels (last dimension of the numpy array)
        """
        
        # Read the numpy array.
        with tf.gfile.FastGFile(filename+'.npy', 'rb') as f:
            image_data = f.read()
        
        try:
                #extract numpy header information
                header_len,dt,index_order,np_array_shape = self._interpret_npy_header_(image_data)
                image = np.frombuffer(image_data, dtype=dt, offset=10+header_len)
                image = np.reshape(image,np_array_shape,order=index_order)
            
        except ValueError as err:
                
                print(err)
        
        # Check that the image is a 3D array + a channel dimension
        
        assert len(image.shape) == 3
        height = image.shape[0]
        width = image.shape[1]
        depth = image.shape[2]
        return image_data[10+header_len:], height, width, depth
    
    def _interpret_npy_header_(self, encoded_image_data):
        """Extracts numpy header information from byte encoded .npy files
        
        Args:
            encoded_image_data: string, (bytes) representation of a numpy array as the tf.gfile.FastGFile.read() method returns it
        Returns:
            header_len: integer, length of header information in bytes
            dt: numpy datatype with correct byte order (little or bigEndian)
            index_order: character 'C' for C-style indexing, 'F' for Fortran-style indexing
            np_array_shape: numpy array, original shape of the encoded numpy array
        """
        #Check if the encoded data is a numpy array or not
        numpy_prefix = b'\x93NUMPY'
        if encoded_image_data[:6] != numpy_prefix:
            raise ValueError('The encoded data is not a numpy array')
        
        #Check if the encoded data is not corrupted and long enough to hold the whole header information
        if len(encoded_image_data)>10:
            #header length in bytes is encoded in the 8-9th bytes of the data as an uint16 number
            header_len=np.frombuffer(encoded_image_data, dtype=np.uint16, offset=8,count=1)[0]
            #extract header data based on variable header length
            header_data=str(encoded_image_data[10:10+header_len])
            
            #extract data type information from the header
            dtypes_dict = {'u1': np.uint8, 'u2': np.uint16, 'u4' : np.uint32, 'u8': np.uint64,
                           'i1': np.int8, 'i2': np.int16, 'i4': np.int32, 'i8': np.int64,
                           'f4': np.float32, 'f8': np.float64, 'b1': np.bool}
    
            start_datatype = header_data.find("'descr': ")+10
            dt = dtypes_dict[header_data[start_datatype+1:start_datatype+3]]
            
            #both big and littleEndian byte order should be interpreted correctly
            if header_data[start_datatype:start_datatype+1] is '>':
                dt = dt.newbyteorder('>')
                
            
            #extract index ordering information from the header
            index_order='C'
            start_index_order=header_data.find("'fortran_order': ")+17
            if header_data[start_index_order:start_index_order+4]=='True':
                index_order='F'
            
            
            #extract array shape from the header
            start_shape = header_data.find("'shape': (")+10
            end_shape = start_shape + header_data[start_shape:].find(")")
            
            np_array_shape=np.fromstring(header_data[start_shape:end_shape],dtype=int, sep=',')
            
            return header_len,dt,index_order,np_array_shape
        else:
            raise ValueError('The encoded data length is not sufficient')
            
    def _int64_feature(self, value):
      """Wrapper for inserting int64 features into Example proto."""
      if not isinstance(value, list):
        value = [value]
      
      return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    
    
    def _float_feature(self, value):
      """Wrapper for inserting float features into Example proto."""
      if not isinstance(value, list):
        value = [value]
      return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    
    
    def _bytes_feature(self, value):
      """Wrapper for inserting bytes features into Example proto."""
      if isinstance(value, six.string_types):           
        value = six.binary_type(value, encoding='utf-8') 
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
     
    def _convert_to_example_(self, filename, image_buffer, label, sex, height, width, depth):
      """Build an Example proto for an example.
      Args:
        filename: string, path to an image file, e.g., '/path/to/example.npy'
        image_buffer: string, encoded 3D numpy array
        label: integer, identifier for the ground truth for the network
        height: integer, image height in voxels
        width: integer, image width in voxels
        depth: integer, image depth in voxels
        num_channels: integer, number of color channels (last dimension of the numpy array)
      Returns:
        Example proto
      """
      
      
      example = tf.train.Example(features=tf.train.Features(feature={
          'image/height': self._int64_feature(height),
          'image/width': self._int64_feature(width),
          'image/depth': self._int64_feature(depth),
          'image/label': self._float_feature(label),
          'image/sex': self._int64_feature(sex),
          'image/encoded': self._bytes_feature(image_buffer)}))
      return example
        
     

