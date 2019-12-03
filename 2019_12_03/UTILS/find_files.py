
import os
import sys
import ezodf
import numpy as np
import path_utils as PU
import pandas as pd
from shutil import copyfile

d = dict()

def _find_ID_in_path_(path_, ID):
    ID_s = str(ID)
    from_idx = path_.find(ID_s)
    return from_idx

#%%
#MTA
mtafiles = ezodf.opendoc('D:\\Peti\\Aging\\data\\MTA\\masolat_yd.ods')
sheet = mtafiles.sheets[0]
df_dict = {}
for i, row in enumerate(sheet.rows()):
    # row is a list of cells
    # assume the header is on the first row
    if i == 0:
        # columns as lists in a dictionary
        df_dict = {cell.value:[] for cell in row}
        # create index for the column headers
        col_index = {j:cell.value for j, cell in enumerate(row)}
        continue
    for j, cell in enumerate(row):
        # use header instead of column index
        df_dict[col_index[j]].append(cell.value)
# and convert to a DataFrame
d = dict()
file_names = []
kor = []
sex = []
wherefrom = []
isMigraine = []
szum = 0
for idx in range(len(df_dict['Kor'])):
    if(df_dict['Kor'][idx] != None):
        kor.append(df_dict['Kor'][idx]/365.25)
        file_names.append(df_dict['file names'][idx])
        if(df_dict['Nem'][idx] == 'nő'):
            sex.append(0)
        else:
            sex.append(1)
        wherefrom.append('MTA')
        isMigraine.append(0)
    else:
        szum +=1
        continue


#%%
#SALD
kinai = pd.read_excel('D:\\Peti\\Aging\\data\\SALD\\sub_information.xlsx')
for idx in range(len(kinai['Sub_ID'].values)):
    name = PU.manip.find_in_subdirs("D:\\Peti\\Aging\\data\\SALD\\","*" + str(kinai['Sub_ID'][idx]) + "*.nii")
    if(len(name) == 1):
        file_names.append(name[0])
        kor.append(kinai['Age'][idx])
        wherefrom.append('SALD')
        isMigraine.append(0)
        
        if(kinai['Sex'][idx] == 'F'):
            sex.append(0)
        else:
            sex.append(1)
            
#%%
            
demo_files = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\data\\1000c\\anat\\txt_s','*demog*')
anat_files = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\data\\1000c\\','wm*')
error = []

for txt in demo_files:
    place_name = txt.split('\\')[-1].split('_demog')[0]
    not_good_places = ['AnnArbor_a', 'AnnArbor_b', 'Bangor', 'Orangeburg', 'Oxford', 'Ontario']
    
    vane = False
    for not_good_place in not_good_places:
        if(place_name == not_good_place):
            vane = True
    if(vane):
        continue
            
  
    df = pd.read_csv(txt, sep="\t", header=None)
    
    #age = df[df[0]=='sub04619'][2].values
    for key,value in df.iterrows():
        matchers = [place_name, value[0]]
        try:
            
            matching = [s for s in anat_files if all(xs in s for xs in matchers)]
            value[2]
            
            file_names.append(matching[0])
            
            if(type(value[2]) == int or type(value[2]) == float):
                kor.append(value[2])
                if(value[3] == 'f'):
                    sex.append(0)
                else:
                    sex.append(1)
                    
            elif(type(value[3]) == int or type(value[3]) == float):
                kor.append(value[3])
                if(value[2] == 'f'):
                    sex.append(0)
                else:
                    sex.append(1)
            
            wherefrom.append('1000C')
            isMigraine.append(0)
                
        except:
            error.append(matchers)
            continue
#%%
#ADNI
        
source_path = "D:\\Peti\\Aging\\data\\ADNI\\";
csv_file = PU.manip.find_in_subdirs(source_path,'*.csv');
files = PU.manip.find_in_subdirs(source_path + "\\SIEMENS_to1p1_CN\\SALD_spm\\",'anat\\wm*')

df = pd.read_csv(csv_file[0])
headers = list(df.columns.values)

d = dict()

for file_idx in range (len(files)):
    subject = files[file_idx].split('\\')[-3].split('-')[0].split('s')[1]
    for jdx in range(len(df[headers[1]])):
        if(_find_ID_in_path_(subject, df[headers[1]][jdx]) != -1):
            break
    file_names.append(files[file_idx])
    kor.append(df[headers[4]][jdx])
    wherefrom.append('ADNI')
    isMigraine.append(0)
    if(df[headers[3]][jdx] == "F"):
        sex.append(0)
    elif(df[headers[3]][jdx] == "M"):
        sex.append(1)
    else:
        print('SexNOTFOUND')
            

#%%
#migraine
df = pd.read_excel('D:\\Peti\\Aging\\data\\migraine\\labels.xlsx')
headers = list(df.columns.values)
df_kontrol = df[[headers[0],headers[1],headers[2],headers[3]]] #ID, file name, age, sex
df_mig = df[[headers[7],headers[8],headers[9],headers[10]]] #ID, file name, age, sex

files = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\data\\migraine\\','anat\\wm*')
for migrene in range(2):
    
    for file in files:
        spl = file.split('\\')[-1].split('.')[0].split('wm')[1]
        
        if(_find_ID_in_path_(spl, 'MPRAGE_GASER') != -1 and migrene == 0):
            label_file = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\data\\new_test\\','labels.csv')
            
            df = pd.read_csv(label_file[0])
            
            for idx in range(len(df['file_name'])):
                if(_find_ID_in_path_(df['file_name'][idx], spl) != -1):
                    file_names.append(df['file_name'][idx])
                    kor.append(df['age'][idx])
                    sex.append(df['sex'][idx])
                    wherefrom.append('Migraine_measurement')
                    isMigraine.append(migrene)
        
        if(len(np.where(df_kontrol[headers[1]] == spl)[0])>0):
            if(migrene == 0):
                file_names.append(file)
                kor.append(df_kontrol[headers[2]][np.where(df_kontrol[headers[1]] == spl)[0][0]])
                wherefrom.append('Migraine_measurement')
                isMigraine.append(migrene)
                if(df_kontrol[headers[3]][np.where(df_kontrol[headers[1]] == spl)[0][0]] == 1):
                    sex.append(1)
                elif(df_kontrol[headers[3]][np.where(df_kontrol[headers[1]] == spl)[0][0]] == 2):
                    sex.append(0)
                else:
                    print('SexNOTFOUND')
                    
            
        elif(len(np.where(df_mig[headers[8]] == spl)[0])>0):
            if(migrene == 1):
                file_names.append(file)
                kor.append(df_mig[headers[9]][np.where(df_mig[headers[8]] == spl)[0][0]])
                wherefrom.append('Migraine_measurement')
                isMigraine.append(migrene)
                if(df_mig[headers[10]][np.where(df_mig[headers[8]] == spl)[0][0]] == 1):
                    sex.append(1)
                elif(df_mig[headers[10]][np.where(df_mig[headers[8]] == spl)[0][0]] == 2):
                    sex.append(0)
                else:
                    print('SexNOTFOUND')

#%%

#mig_uverusers     
               
source_path = "D:\\Peti\\Aging\\data\\mig_v2_overusers\\";
csv_file = PU.manip.find_in_subdirs(source_path,'*.xlsx');
files = PU.manip.find_in_subdirs(source_path + "\\SALD_spm\\",'anat\\wm*')

df = pd.read_excel(csv_file[0])
headers = list(df.columns.values)

filenameslist = df[headers[15]]
ageslist = df[headers[16]]
sexslist = df[headers[17]]

d = dict()

for file_idx in range (len(files)):
    subject = files[file_idx].split('\\')[-3]
    for jdx in range(len(filenameslist)):
        if(_find_ID_in_path_(subject, filenameslist[jdx]) != -1):
            break
    wherefrom.append('Migraine_measurement')
    isMigraine.append(2)
    file_names.append(files[file_idx])
    kor.append(ageslist[jdx])
    if(sexslist[jdx] == 2):
        sex.append(0)
    elif(sexslist[jdx] == 1):
        sex.append(1)
    else:
        print('SexNOTFOUND')
    

d['age'] = np.array(kor)
d['file_name'] = np.array(file_names)
d['sex'] = np.array(sex)
d['wherefrom'] = np.array(wherefrom)
d['migraine (0-no, 1-yes, 2-overuser)'] = np.array(isMigraine)


#%%
# copy files, change path and save the csv
dst_folder = 'D:\\Peti\\Aging\\data_mni\\'

for key, val in d.items():
    if(key == 'file_name'):
        new_f_names = []
        for fname in val:
            nifti = fname.split('\\')[-1]
            new_file_name = dst_folder + nifti
            copyfile(fname, new_file_name)
            new_f_names.append(new_file_name)

d['file_name'] = np.array(new_f_names)

df = pd.DataFrame(d)
df.to_csv(dst_folder + 'info_file.csv')
