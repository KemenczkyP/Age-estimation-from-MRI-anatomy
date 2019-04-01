# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 08:51:55 2019

@author: KemyPeti
"""


import sys
import os
sys.path
sys.path.append(os.getcwd() + '\\UTILS\\')
import path_utils as PU

class manip():
    @staticmethod
    def scan_1000_c():
        '''
        #scan 1000c datas -> output dict file_name and age
        '''
        import pandas as pd
        
        demo_files = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\data\\1000c\\anat\\txt_s','*demog*')
        anat_files = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\data\\1000c\\','wm*')
        new_df = dict()
        new_df['file_name']=[]
        new_df['age']=[]
        new_df['sex']=[]
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
                    new_df['file_name'].append(matching[0])
                    if(type(value[2]) == int or type(value[2]) == float):
                        new_df['age'].append(value[2])
                        if(value[3] == 'f'):
                            new_df['sex'].append(0)
                        else:
                            new_df['sex'].append(1)
                            
                    elif(type(value[3]) == int or type(value[3]) == float):
                        new_df['age'].append(value[3])
                        if(value[2] == 'f'):
                            new_df['sex'].append(0)
                        else:
                            new_df['sex'].append(1)
                        
                except:
                    error.append(matchers)
                    continue
        return new_df
        
    @staticmethod
    def scan_kinai():
        '''
        scan Southern Un. datas -> output dict file_name and age
        '''
        import pandas as pd
        
        kinai = pd.read_excel('D:\\Peti\\Aging\\data\\SALD\\sub_information.xlsx')
        names = []
        labels = []
        sex  = []
        for idx in range(len(kinai['Sub_ID'].values)):
            name = PU.manip.find_in_subdirs("D:\\Peti\\Aging\\data\\SALD\\","*" + str(kinai['Sub_ID'][idx]) + "*.nii")
            if(len(name) == 1):
                names.append(name)
                labels.append(kinai['Age'][idx])
                if(kinai['Sex'][idx] == 'F'):
                    sex.append(0)
                else:
                    sex.append(1)
                
        new_df = dict()
        new_df['age'] = labels
        new_df['file_name'] = names
        new_df['sex'] = sex
        return new_df
     
    @staticmethod
    def scan_MTA():
        '''
       #scan MTA datas -> output dict file_name and age
        '''
        
        import ezodf
        import numpy as np
        
        doc = ezodf.opendoc('D:\\Peti\\Aging\\data\\MTA\\masolat_yd.ods')
        sheet = doc.sheets[0]
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
        szum = 0
        for idx in range(len(df_dict['Kor'])):
            if(df_dict['Kor'][idx] != None):
                kor.append(df_dict['Kor'][idx]/365.25)
                file_names.append(df_dict['file names'][idx])
                if(df_dict['Nem'][idx] == 'nő'):
                    sex.append(0)
                else:
                    sex.append(1)
            else:
                szum +=1
                continue
        
        d['age'] = np.array(kor)
        d['file_name'] = np.array(file_names)
        d['sex'] = np.array(sex)
        return d
    
    @staticmethod
    def scan_migrene(migrene):
        '''
        #scan migrene datas (1 if migrene, 0 if not) -> output dict file_name and age
        '''
        
        import pandas as pd
        import numpy as np
        
        
        df = pd.read_excel('D:\\Peti\\Aging\\data\\migraine\\labels.xlsx')
        headers = list(df.columns.values)
        df_kontrol = df[[headers[0],headers[1],headers[2],headers[3]]] #ID, file name, age, sex
        df_mig = df[[headers[7],headers[8],headers[9],headers[10]]] #ID, file name, age, sex
        
        files = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\data\\migraine\\','anat\\wm*')
        
        d = dict()
        file_names = []
        age = []
        sex = [] # f=0, m=1
        
        for file in files:
            spl = file.split('\\')[-1].split('.')[0].split('wm')[1]
            
            if(len(np.where(df_kontrol[headers[1]] == spl)[0])>0):
                if(migrene == 0):
                    file_names.append(file)
                    age.append(df_kontrol[headers[2]][np.where(df_kontrol[headers[1]] == spl)[0][0]])
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
                    if(df_mig[headers[10]][np.where(df_mig[headers[8]] == spl)[0][0]] == 1):
                        sex.append(1)
                    elif(df_mig[headers[10]][np.where(df_mig[headers[8]] == spl)[0][0]] == 2):
                        sex.append(0)
                    else:
                        print('SexNOTFOUND')
        
        
        d['age'] = np.array(age)
        d['file_name'] = np.array(file_names)
        d['sex'] = np.array(sex)
        return d

    @staticmethod
    def scan_migrene_True():
        return manip.scan_migrene(1)
        
    @staticmethod
    def scan_migrene_False():
        return manip.scan_migrene(0)
    
    @staticmethod
    def scan_migcontrol_extension():
        import pandas as pd
        import numpy as np
        from datetime import datetime
        
        source_path = "D:\\Peti\\Aging\\data\\mig_extension\\";
        csv_file = PU.manip.find_in_subdirs(source_path,'*.csv');
        files = PU.manip.find_in_subdirs(source_path,'anat\\wm*')
        
        df = pd.read_csv(csv_file[0])
        headers = list(df.columns.values)
        
        d = dict()
        file_names = []
        age = []
        sex = [] # f=0, m=1
        
        for file_idx in range (len(files)):
            for jdx in range(len(df[headers[1]])):
                if(manip._find_ID_in_path_(files[file_idx], df[headers[1]][jdx]) != -1):
                    break
            file_names.append(files[file_idx])
            date_b = df[headers[2]][jdx]
            date_b = datetime.strptime(date_b, '%Y.%m.%d.')
            date_c = df[headers[3]][jdx]
            date_c = datetime.strptime(date_c, '%Y.%m.%d.')
            age_ = date_c - date_b;
            age.append(age_.days/365.25)
            if(df[headers[4]][jdx] == "female"):
                sex.append(0)
            elif(df[headers[4]][jdx] == "male"):
                sex.append(1)
            else:
                print('SexNOTFOUND')
        
        d['age'] = np.array(age)
        d['file_name'] = np.array(file_names)
        d['sex'] = np.array(sex)
        return d
        
        
    @staticmethod
    def arange_dicts(dicts):
        '''
        dicts have age and file_name attribute
        '''
        import numpy as np
        file_names = np.array([])
        ages = np.array([])
        sex = np.array([])
        for idx in range(len(dicts)):
            ages = np.append(ages,dicts[idx]['age'])
            sex = np.append(sex,dicts[idx]['sex'])
            file_names = np.append(file_names, dicts[idx]['file_name'])
        d = dict()
        d['file_name'] = file_names
        d['age'] = ages
        d['sex'] = sex
        return d
    
    @staticmethod
    def _find_ID_in_path_(path_, ID):
        ID_s = str(ID)
        from_idx = path_.find(ID_s)
        return from_idx
        