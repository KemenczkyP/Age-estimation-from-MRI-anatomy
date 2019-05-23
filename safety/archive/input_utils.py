class manip():
    @staticmethod
    def scan_1000_c():
        '''
        #scan 1000c datas -> output dict file_name and age
        '''
        import path_utils as PU
        import pandas as pd
        
        demo_files = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\1000c\\anat\\txt_s','*demog*')
        anat_files = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\1000c\\','wm*')
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
        import glob
        
        kinai = pd.read_excel('D:\\Peti\\Aging\\data\\sub_information.xlsx')
        names = []
        labels = []
        sex  = []
        for idx in range(len(kinai['Sub_ID'].values)):
            name = glob.glob("D:\\Peti\\Aging\\data\\*"+str(kinai['Sub_ID'].values[idx]) + "*.nii")
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
        
        doc = ezodf.opendoc('I:\\mnianat\\masolat_yd.ods')
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
        import glob
        from shutil import copyfile
        import path_utils as PU
        
        
        df = pd.read_excel('D:\\Peti\\Aging\\Migrene\\fMRI_anat_MTA\\labels.xlsx')
        headers = list(df.columns.values)
        df_kontrol = df[[headers[0],headers[1],headers[2],headers[3]]] #ID, file name, age, sex
        df_mig = df[[headers[7],headers[8],headers[9],headers[10]]] #ID, file name, age, sex
        
        files = PU.manip.find_in_subdirs('D:\\Peti\\Aging\\Migrene\\','anat\\wm*')
        
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
        
#all_ages = np.concatenate((labels_1000_c['age'], labels_kinai['age'], labels_MTA['age']))
#ss =plt.hist(all_ages,np.arange(np.min(all_ages),np.max(all_ages),2))

#df = pd.DataFrame(d)
'''
labels_1000_c = manip.scan_1000_c()
labels_MTA = manip.scan_MTA()
kinai = manip.scan_kinai()
files_w_labels = manip.arange_dicts([labels_1000_c, kinai])
'''
'''
def copyfiles():
    import pandas as pd
    import numpy as np
    import glob
    from shutil import copyfile

    
    mig_files = glob.glob('D:\\Peti\\Aging\\Migrene\\fMRI_anat_MTA\\migrene\\*')
    kon_files = glob.glob('D:\\Peti\\Aging\\Migrene\\fMRI_anat_MTA\\kontroll\\*')
    files = np.append(mig_files,kon_files)
    
    
    for file in files:
        spl = file.split('\\')[-1]
        nii_name = glob.glob(file+'\\*.nii')
        try:
            copyfile(nii_name[0], 'D:\\Peti\\Aging\\Migrene\\raw\\' + spl + '.nii')
        except:
            continue

#copyfiles()
'''