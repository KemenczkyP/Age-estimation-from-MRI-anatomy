import pandas as pd
import numpy as np

def read_data(path, whichset):
    
    if(whichset == 'public'):
        train = pd.read_csv(path + 'public_train.csv')
        valid = pd.read_csv(path + 'public_valid.csv')
    elif(whichset == 'inhouse'):
        train = pd.read_csv(path + 'inhouse_train.csv')
        valid = pd.read_csv(path + 'inhouse_valid.csv')
    elif(whichset == 'migraine'):
        train = pd.read_csv(path + 'MIG.csv')
        valid = pd.read_csv(path + 'CTR.csv')
    
    return train, valid



class Data_Iterator:
    
    def __init__(self, 
                 data_dict,
                 batch_size = 4):
        
        self.datadict = data_dict
        self.batch_size = batch_size
        
        self.data_num = len(self.datadict['file_name'])
        if(self.data_num == 0):
            self.step_num = 0
        else:
            self.step_num = int((self.data_num-0.1)/self.batch_size) +1
        
        self.InitializeEpoch()
        self.current_pointer = 0
        
    def InitializeEpoch(self):
        self.current_shuffle = np.arange(0, len(self.datadict['file_name']))
        np.random.shuffle(self.current_shuffle)
        
    def GetData(self):
        
        if(self.current_pointer + self.batch_size >= len(self.current_shuffle)):
        
            #print(self.current_shuffle[self.current_pointer:])
        
            labels = self.datadict['age'].iloc[self.current_shuffle[self.current_pointer:]].to_numpy()
            sexs   = self.datadict['sex'].iloc[self.current_shuffle[self.current_pointer:]].to_numpy()
            images = self.ReadNumpy(self.datadict['file_name'].iloc[self.current_shuffle[self.current_pointer:]].to_numpy())
            
            self.current_pointer = 0
            self.InitializeEpoch()
        
        else:
            
            #print(self.current_shuffle[self.current_pointer: self.current_pointer + self.batch_size])
        
            labels = self.datadict['age'].iloc[self.current_shuffle[self.current_pointer: self.current_pointer + self.batch_size]].to_numpy()
            sexs   = self.datadict['sex'].iloc[self.current_shuffle[self.current_pointer: self.current_pointer + self.batch_size]].to_numpy()
            images = self.ReadNumpy(self.datadict['file_name'].iloc[self.current_shuffle[self.current_pointer: self.current_pointer + self.batch_size]].to_numpy())
        
            self.current_pointer = self.current_pointer + self.batch_size
        
        
        return labels, sexs, images
        
        
    @staticmethod
    def ReadNumpy(filenames):
        images = []
        for idx in range(len(filenames)):
            
            nii_name = filenames[idx]
            npy_name = nii_name.split('.')[0] + '.npy'
            
            images.append(np.load(npy_name))
        images = np.array(images)
        return images
        
    
''' 
import pickle

with open('D:\\Peti\\aging3\\DATA\\MIG.pickle', 'rb') as handle:
    mig = pickle.load(handle)

itera = Data_Iterator(mig)
labels, sexs, images = itera.GetData()

'''
