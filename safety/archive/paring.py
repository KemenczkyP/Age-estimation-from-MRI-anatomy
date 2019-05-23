# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 13:11:14 2019

@author: KemyPeti
"""
import pandas as pd
import numpy as np
import path_utils as PU

def scan_migrene(migrene):
    '''
    #scan migrene datas (1 if migrene, 0 if not) -> output dict file_name and age
    '''
    

    import glob
    from shutil import copyfile
    
    
    df = pd.read_excel('D:\\Peti\\Aging\\data\\migraine\\labels.xlsx')
    headers = list(df.columns.values)
    df_kontrol = df[[headers[0],headers[1],headers[2],headers[3]]] #ID, file name, age, sex
    df_mig = df[[headers[7],headers[8],headers[9],headers[10]]] #ID, file name, age, sex
    
    files = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\data\\migraine\\SALD_spm\\','anat\\wm*')
    
    d = dict()
    file_names = []
    age = []
    sex = [] # f=0, m=1
    name = []
    
    for file in files:
        spl = file.split('\\')[-1].split('.')[0].split('wm')[1]
        if(len(np.where(df_kontrol[headers[1]] == spl)[0])>0):
            if(migrene == 0):
                file_names.append(file)
                age.append(df_kontrol[headers[2]][np.where(df_kontrol[headers[1]] == spl)[0][0]])
                
                name.append(spl)
                if(df_kontrol[headers[3]][np.where(df_kontrol[headers[1]] == spl)[0][0]] == 1):
                    sex.append(1)
                elif(df_kontrol[headers[3]][np.where(df_kontrol[headers[1]] == spl)[0][0]] == 2):
                    sex.append(0)
                else:
                    print('SexNOTFOUND')
            
        elif(len(np.where(df_mig[headers[8]] == spl)[0])>0):
            if(migrene == 1):
                file_names.append(file)
                age.append(df_mig[headers[9]][np.where(df_mig[headers[8]] == spl)[0][0]])
                
                name.append(spl)
                if(df_mig[headers[10]][np.where(df_mig[headers[8]] == spl)[0][0]] == 1):
                    sex.append(1)
                elif(df_mig[headers[10]][np.where(df_mig[headers[8]] == spl)[0][0]] == 2):
                    sex.append(0)
                else:
                    print('SexNOTFOUND')
    
    
    d['age'] = np.array(age)
    d['file_name'] = np.array(file_names)
    d['sex'] = np.array(sex)
    d['name'] = np.array(name)
    return d





def pair(migraine, control,contact):
    import numpy as np
    migrene = []
    
    for idx in range(len(migraine['name'])):
        migrene.append(migraine['name'][idx])
        
        for jdx in range(len(control['file_name'])):
            if(migraine['sex'][idx] == control['sex'][jdx]):
                if(np.abs(migraine['age'][idx]-control['age'][jdx])<2):
                    contact[jdx,idx]= 1   
    return contact
    
def connect_fun(contact):
    connect = [] #mig, notmig
    
    while(True):
        
        next_ = np.argsort(np.sum(contact,0))
        next_val_ = np.sort(np.sum(contact,0))
        
        try:
            first_val = next_val_[np.where(next_val_>0)[0][0]]
            first_idx = next_[np.where(next_val_>0)[0][0]]
            
            if(first_val>0):
                oks = np.where(contact[:,first_idx] ==1)[0][0]
                connect.append([first_idx,oks])
                contact[:,first_idx] = 0
                contact[oks,:] = 0
        except:
            break 
    return connect
    
def get_filenames_from_connect(connect, data_migraine, data_control):
    connect2 = []

    for idx in range(len(connect)):
        connect2.append([data_migraine['file_name'][connect[idx][0]],
                         data_control['file_name'][connect[idx][1]]])
    return connect2

data_control = scan_migrene(0)
data_migraine = scan_migrene(1)
contact = np.zeros((len(data_control['name']),len(data_migraine['name'])))
contact = pair(data_migraine,data_control,contact)
connect = connect_fun(contact)

connect_names = get_filenames_from_connect(connect, data_migraine, data_control)

#%%
# plus new data

def scan_new_test():
    
    import glob
    from shutil import copyfile
    
    label_file = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\data\\new_test\\','*.csv')
    
    df = pd.read_csv(label_file[0]).values
    df = df[(0,1,4,6,7,8),:]
    
    return df


def add_ConnectNames(connect_names, new_data):
    cc2 = connect_names
    for jdx in range(len(new_data)):
        for idx in range(len(connect_names)):
            mig_name = connect_names[idx][0].split('\\')[-1].split('.')[0]
            if('wm'+new_data[jdx,0] == mig_name):
                print("Problem: {0}".format(new_data[jdx,0]))
        mig = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\data\\migraine\\SALD_spm\\','anat\\*'+'wm'+ new_data[jdx,0]+'*')
        new_control = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\data\\mig_extension\\MTA_controls_raw\\SALD_spm\\','anat\\wm*'+ str(new_data[jdx,1]) + '.nii')
        
        cc2.append([mig[0], new_control[0]])
    return cc2

new_data = scan_new_test()
cc2 = add_ConnectNames(connect_names,new_data)

#%%
from shutil import copyfile

def copy_files(connect_names):
    for idx in range(len(connect_names)):
        file_name = connect_names[idx][0].split('\\')[-1].split('.')[0]
        copyfile(connect_names[idx][0], 'D:\\Peti\\Aging\\data\\new_test\\migraine\\' + str(idx) + '_' + file_name + '.nii')
        
        file_name = connect_names[idx][1].split('\\')[-1].split('.')[0]
        copyfile(connect_names[idx][1], 'D:\\Peti\\Aging\\data\\new_test\\controls\\' + str(idx) + '_' + file_name + '.nii')
    
copy_files(cc2)

#%%
    
mig = data_migraine['file_name']
for idx in range(len(mig)):
    mig[idx] = mig[idx].split('\\')[-1].split('.')[0]
    
migNOT = data_control['file_name']
for idx in range(len(migNOT)):
    migNOT[idx] = migNOT[idx].split('\\')[-1].split('.')[0]

#%%
try:
    del(contact,first_val,  idx, jdx, kdx, oks)
except:
    pass
pairs_by_names = list()
for idx in range(len(connect)):
    for jdx in range(len(mig)):
        if(connect[idx][0] == int(mig[jdx].split('mig')[1])):
            for kdx in range(len(migNOT)):
                if(connect[idx][1] == int(migNOT[kdx].split('con')[1])):
                    pairs_by_names.append([data_migraine['file_name'][jdx],data_control['file_name'][kdx]])
    


hianyzo_mig_name = []
for idx in range(len(mig)):
    szum = 0
    for jdx in range(len(connect)):
        if(connect[jdx][0] == int(mig[idx].split('mig')[1])):
            szum = 1
            break
    if(szum == 0):
        hianyzo_mig_name.append(mig[idx])
        
del(idx,jdx,kdx,szum)

#%%
def hianyzo_to_dict(hianyzo_mig_name):
    
    d = dict()
    file_names = []
    age = []
    sex = [] # f=0, m=1
    name = []
        
    for idx in range(len(hianyzo_mig_name)):
        a = np.where(hianyzo_mig_name[idx] == data_migraine['file_name'])
        file_names.append(data_migraine['file_name'][a[0][0]])
        age.append(data_migraine['age'][a[0][0]])
        sex.append(data_migraine['sex'][a[0][0]])
        name.append(data_migraine['name'][a[0][0]])
        d['age'] = np.array(age)
        d['file_name'] = np.array(file_names)
        d['sex'] = np.array(sex)
        d['name'] = np.array(name)
    return d

d = hianyzo_to_dict(hianyzo_mig_name)
#%%
import input_utils as IU

MTA_files = IU.manip.scan_MTA()


class mappingo(object):
    class pairs():
        
        def __init__(self, first, second):
            self.first = first
            self.second = second
            
    def __init__(self, name_list):
        self.map = list()
        for idx in range(len(name_list)):
            self.map.append(self.pairs(idx+1,name_list[idx]))
        self.LENGTH = len(name_list)
            
    def __print__(self):
        for idx in range(len(self.map)):
            print("[ {0} \t;\t {1} ]".format(self.map[idx].first,self.map[idx].second))

        
        

contact = np.zeros((len(MTA_files['file_name']),len(d['name'])))
contact = pair(d,MTA_files,contact)

map_MTA = mappingo(MTA_files['file_name'])
map_mig = mappingo(hianyzo_mig_name)


connect = connect_fun(contact)


pairs_by_names2 = list()
for idx in range(len(connect)):
    for jdx in range(map_mig.LENGTH):
        if(connect[idx][0] == map_mig.map[jdx].first):
            for kdx in range(map_MTA.LENGTH):
                if(connect[idx][1] == int(map_MTA.map[kdx].first)):
                    pairs_by_names2.append([map_mig.map[jdx].second,
                                            map_MTA.map[kdx].second])

hianyzo_mig_name2 = []
for idx in range(len(hianyzo_mig_name)):
    szum = 0
    for jdx in range(len(pairs_by_names2)):
        if(hianyzo_mig_name[idx] == pairs_by_names2[jdx][0]):
            szum = 1
            break
    if(szum == 0):
        hianyzo_mig_name2.append(hianyzo_mig_name[idx])
        
del(idx,jdx,kdx,szum)
pairs_by_names = np.concatenate((pairs_by_names, pairs_by_names2))
d = hianyzo_to_dict(hianyzo_mig_name2)
#%%
from datetime import datetime

new_controls = pd.read_csv('MTA_control_subjects.csv')
dicto = new_controls.to_dict('series')
dicto['file_name'] = dicto['Control subject ID']

dicto['sex'] = []
dicto['age'] = []
for idx in range(len(dicto['Control subject sex'])):
    dicto['sex'].append(0 if dicto['Control subject sex'][idx] == 'female' else 1)
    
    datetime_object = datetime.strptime(dicto['Control subject measurement date'][idx], '%Y.%m.%d.')
    datetime_object2 = datetime.strptime(dicto['Control subject birth date'][idx], '%Y.%m.%d.')
    age = float(((datetime_object - datetime_object2)).total_seconds()/3600/24/365.25)
    dicto['age'].append(age)

pps = []
for idx in range(len(dicto['Control subject sex'])):
    pps.append([dicto['Migraine subject ID'][idx],dicto['Control subject ID'][idx]])
    
pairs_by_names = np.concatenate((pairs_by_names, pps))

dict_ = dict()
dict_['total_matching'] = pairs_by_names
dict_['migraine_data'] = data_migraine
dict_['control_data1'] = data_control
dict_['control_data2'] = MTA_files
dict_['control_data3'] = dicto

import pickle
f = open("matching.pkl","wb")
pickle.dump(dict_,f)
f.close()

#%%
#-----------------------------------------------------------------------------#

import pickle

with open('matching.pkl', 'rb') as f:
    data = pickle.load(f)

dict_ = dict()
age = []
sex = []
file_name = []
def append_dict(dict_,index):
    age.append(dict_['age'][index])
    sex.append(dict_['sex'][index])
    
    fn = PU.manip.find_in_subdirs('D:\Peti\Aging', '*'+ dict_['file_name'][index] + '*')
    if(len(fn) == 0):
        fn = dict_['file_name'][index] 
    print(fn)
    file_name.append(fn)
    

for idx in range(-10,len(data['total_matching'])):
    try:
        index_mig = np.where(data['total_matching'][idx][0] == data['migraine_data']['file_name'])[0][0]
    except:
        index_mig = np.where('wm' + data['total_matching'][idx][0] == data['migraine_data']['file_name'])[0][0]  
    
    append_dict(data['migraine_data'], index_mig)
    
    try:
        index_cont = np.where(data['total_matching'][idx][1] == data['control_data1']['file_name'])[0][0]
        append_dict(data['control_data1'], index_mig)
                             
    except:
        try:
            index_cont = np.where(data['total_matching'][idx][1] == data['control_data2']['file_name'])[0][0]
            append_dict(data['control_data2'], index_mig)
        except:
            try:
                index_cont = np.where(data['total_matching'][idx][1] == data['control_data3']['file_name'])[0][0]
                append_dict(data['control_data3'], index_mig)
            except:
                pass

#%%

    