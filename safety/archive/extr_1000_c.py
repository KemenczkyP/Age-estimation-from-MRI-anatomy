# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 10:28:28 2019

@author: latlab
"""

'''
def get_data():
    from zipfile import ZipFile 
    import glob
    import pandas as pd
    import numpy as np
    import nibabel as nib
    
    #loop
    file_name = 'ds000003' 
    with ZipFile(file_name + '.zip', 'r') as myzip:
        
        # extracting all the files 
        print('Extracting all the files now...') 
        myzip.extractall(file_name) 
        print('Done!') 
    
    info = pd.read_csv('.\\' + file_name +'\\participants.tsv', sep = '\t')
    
    subs = glob.glob('.\\' + file_name +'\\sub*')
    
    for idx in range (len(subs)):
        spl = subs[idx].split('\\')[-1]
        index = np.where(info['participant_id'] == spl)[0][0]
        label = info['age'][index]
        label=int(label)
        
        data = glob.glob(subs[idx]+'\\anat\\*T1*')
        img = nib.load(data[0])  
        arr = np.array(img.dataobj)
        arr = np.subtract(arr,np.mean(arr))
        arr = np.divide(arr,np.std(arr))
    return label, arr
    

label, arr = get_data()
'''

import tarfile
import glob
import nibabel as nib
import shutil 
import os



tarnames = glob.glob("D:\\Peti\\Aging\\1000c\\*.tar")

for idx in tarnames[19:]:
    splt = idx.split(".")[0].split("\\")
    if not os.path.exists("D:\\Peti\\Aging\\1000c\\dottar\\"):
        os.makedirs("D:\\Peti\\Aging\\1000c\\dottar\\")
    tar = tarfile.open(idx, "r:")
    tar.extractall("D:\\Peti\\Aging\\1000c\\dottar\\")
    tar.close()
    anats = glob.glob("D:\\Peti\\Aging\\1000c\\dottar\\sub*")
    for jdx in anats:
        splt2 = jdx.split('\\')[-1]
        filenames = glob.glob(jdx+'\\anat\\*anon*')
        try:
            img = nib.load(filenames[0])
            nib.save(img, "D:\\Peti\\Aging\\1000c\\anat\\" + splt[-1] + '_' + splt2 + '.nii')
        except:
            continue
    txt_s = glob.glob("D:\\Peti\\Aging\\1000c\\dottar\\*.txt")
    for jdx in txt_s:
        shutil.copyfile(jdx, "D:\\Peti\\Aging\\1000c\\anat\\txt_s\\"+jdx.split('\\')[-1])

    shutil.rmtree('D:\\Peti\\Aging\\1000c\\dottar\\')
        
        
        

