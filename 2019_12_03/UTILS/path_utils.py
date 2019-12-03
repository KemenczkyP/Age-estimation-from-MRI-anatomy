# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 08:51:55 2019

@author: KemyPeti
"""


class manip():
    @staticmethod
    #-------------------------------------------------------------------------#
    def create_dir_if_not_exists(file_path):
        import os
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
            print("directory created")
        else:
            print("directory exists")
    #-------------------------------------------------------------------------#
    @staticmethod
    def find_in_subdirs(source_directory, file_fix_glob_notation):
        '''
        files = manip.find_in_subdirs('D:\\path_to_dir\\', 'pre*') \n
        "files" contains each file name begins with 'pre'
        '''
        import glob
        files = []
        for filename in glob.iglob(source_directory + '/**/'+file_fix_glob_notation, recursive=True):
            files.append(filename)
        return files
    #-------------------------------------------------------------------------#    
    @staticmethod
    def copy_(source, destination):
        
        from shutil import copyfile
        
        copyfile(source, destination)
        
    @staticmethod
    def copytree(src, dst, number_of_copies = 1, symlinks=False, ignore=None):
        '''
        The filename in the destination folder will get a number to the end if 
        there is more copies:
            if(number_of_copies != 1):
                -> .\\filename -> (.\\filename0;.\\filename1;...)
        '''
        import os
        import shutil
        dstfolders = []
        if(number_of_copies>1):
            for idx in range(number_of_copies):
                dstfolders.append(dst + str(idx))
        else:
            dstfolders.append(dst)
            
        for dest in dstfolders:
            if not os.path.exists(dest):
                os.makedirs(dest)
            for item in os.listdir(src):
                s = os.path.join(src, item)
                d = os.path.join(dest, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, symlinks, ignore)
                else:
                    if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                        shutil.copy2(s, d)
    

'''
src_path = 'D:\\Peti\\Aging\\data\\mig_v2_overusers\\'
files = manip.find_in_subdirs(src_path,'*.nii')

for idx in range(len(files)):
    spl = files[idx].split('\\')[-2]
    manip.copy_(files[idx], src_path + 'files\\' + spl + '.nii')

###############################################################################
manip.copytree('D:\\Peti\\Aging\\tmp\\mnist_convnet_model\\2018_06_17\\VALIDTRAIN\\net3\\tri1',
               'D:\\Peti\\Aging\\tmp\\mnist_convnet_model\\2018_06_17\\VALIDTRAIN\\net3_totalCNN\\net3_re',
               number_of_copies = 2)
'''