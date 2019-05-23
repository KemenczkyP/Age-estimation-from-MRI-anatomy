# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 09:23:52 2019

@author: latlab
"""

class statistics_from_output():
    def __init__(self, input_string):
        
        self.get_num_from_string(input_string)
        self.stat_dict = dict()
        self.differences()
        
        
    def differences(self):
        import numpy as np
        
        self.stat_dict['differences'] = np.subtract(self.predictions,self.labels)
        self.stat_dict['mean_difference'] = np.mean(self.stat_dict['differences'])
        self.stat_dict['standard_deviation'] = np.std(self.stat_dict['differences'])
        self.stat_dict['average_difference'] = np.mean(np.abs(self.stat_dict['differences']))
        
        self.stat_dict['labels'] = self.labels
        self.stat_dict['predictions'] = self.predictions
        self.stat_dict['sexs'] = self.sexs
        
        
    def get_num_from_string(self, tring_):
        import numpy as np
        
        
        parts = tring_.split('[[')
        out=[]
        predictions = []
        labels = []
        sexs = []
        
        for idx in range(0,len(parts)):
        
            if(idx % 3 == 0):
                b = ''
                a = parts[idx].split(']')
                for sdx in range(len(a[:-2])):
                    tmp = a[sdx].split('[')
                    for jdx in range(len(tmp)):
                        for kdx in range(len(tmp[jdx])):
                            try:
                                szam = int(tmp[jdx][kdx])
                                b = b + tmp[jdx][kdx] + ' '
                            except:
                                
                                continue
                out.append(b)
            else:
                out.append(parts[idx].split(']]')[0])
        
            separated_num = out[-1].split(' ')
            for jdx in range(len(separated_num)):
                try:
                    num = float(separated_num[jdx])
                except:
                    continue
                if(idx%3 == 1):
                    labels.append(num)
                elif(idx%3 == 2):
                    predictions.append(num)
                else:
                    sexs.append(num)
                    
        self.labels = np.multiply(labels,100)
        self.predictions = np.multiply(predictions,100)
        self.sexs = np.array(sexs)

        
import pickle


INstring = input("not_migrene_LABSnPREDS: ")


stats = statistics_from_output(INstring)
Anotmig = stats.stat_dict

INstring = input("migrene_LABSnPREDS: ")
stats = statistics_from_output(INstring)
Amig = stats.stat_dict

pickle.dump(Anotmig, open( "NOTMigrene0426.p", "wb" ) )
pickle.dump(Amig, open( "Migrene0426.p", "wb" ) )



import numpy as np
from scipy import stats

F_value = np.var(Amig['differences'])/np.var(Anotmig['differences'])

p_significance = 0.01
tablazatbol = 2.61
t_stat, t_p = stats.ttest_ind(Amig['differences'],Anotmig['differences'])
u_stat, u_p = stats.mannwhitneyu(Amig['differences'],Anotmig['differences'])
ks_stat, ks_p = stats.ks_2samp(Amig['differences'],Anotmig['differences'])

















'''










import matplotlib.pyplot as plt
#A = plt.hist(Amig['differences'],bins = np.arange(np.min(Amig['differences']),np.max(Amig['differences']),1))
#B = plt.hist(Anotmig['differences'],bins = np.arange(np.min(Anotmig['differences']),np.max(Anotmig['differences']),1))
'''