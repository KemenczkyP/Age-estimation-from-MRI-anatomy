import os
import sys
import glob

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import pickle

from tfrecord_def import get_dataset_ready
from net3_model import generate_model
from defs import read_data, Data_Iterator


public_train_info, public_valid_info = read_data(os.getcwd() + '\\DATA\\', 'public')

Public_train_iter  = Data_Iterator(public_train_info)
Public_valid_iter  = Data_Iterator(public_valid_info)
#labels, sexs, images = Public_iter.GetData()


model = generate_model(0.1)
model.compile(loss="mse", 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
              metrics=["mae"])

#%%

@tf.function
def training(train_labels, train_sexs, train_images):
        
    l = model.train_on_batch(x = [train_images, train_sexs],
                                 y = train_labels)
    return l

@tf.function
def validing(test_labels, test_sexs, test_images):
            
    l = model.test_on_batch(x = [test_images, test_sexs],
                            y = test_labels)
    return l

def print_process(iterator):
    sys.stdout.write('\r{:4}[{}{}]'.format(iterator.current_pointer, 
                     int((iterator.current_pointer+iterator.batch_size)*25/iterator.data_num)*'x',   
                     (25-int((iterator.current_pointer+iterator.batch_size)*25/iterator.data_num))*' '))
    sys.stdout.flush()

maxEps = 30

for ep in range(maxEps):
    
    #VALIDATION
    if(Public_train_iter.current_pointer == 0):
        
        validation_loss = []
        print("Validating...")
        
        while(True):
            
            test_labels, test_sexs, test_images = Public_valid_iter.GetData()
            test_images = test_images.reshape(-1,79,95,79,1)
            l = validing(test_labels, test_sexs, test_images)
            print_process(Public_valid_iter)
            validation_loss.append(l)
            
            if(Public_valid_iter.current_pointer == 0):
                break
            
        validation_loss = np.array(validation_loss)
        print("\n Validation MSE: {:10}; MAE: {:10}".format(np.round(validation_loss.mean(0)[0],5),
                                                            np.round(validation_loss.mean(0)[1],5)))
        
        a = []
        for idx in range(len(model.weights)):
            a.append(model.weights[idx].numpy())
        with open("sec_model_{}_{}.pickle".format(ep, np.round(validation_loss.mean(0)[1],0)), "wb") as fp:   #Pickling
            pickle.dump(a, fp)
            
        
    print("Training...")
    training_loss = []
    while(True):
        train_labels, train_sexs, train_images = Public_train_iter.GetData()
        train_images = train_images.reshape(-1,79,95,79,1)
        l = training(train_labels, train_sexs, train_images)
        
        print_process(Public_train_iter)
        training_loss.append(l)
        
        if(Public_train_iter.current_pointer == 0):
            break
    
    training_loss = np.array(training_loss)
    print("\n Training MSE: {:10}; MAE: {:10}".format(np.round(training_loss.mean(0)[0],5),
                                                        np.round(training_loss.mean(0)[1],5)))


#%%
    
ih_train_info, ih_valid_info = read_data(os.getcwd() + '\\DATA\\', 'inhouse')

IH_train_iter  = Data_Iterator(ih_train_info)
IH_valid_iter  = Data_Iterator(ih_valid_info)

maxEps = 10

for ep in range(maxEps):
    
    #VALIDATION
    if(IH_train_iter.current_pointer == 0):
        
        validation_loss = []
        print("Validating...")
        
        while(True):
            
            test_labels, test_sexs, test_images = IH_valid_iter.GetData()
            test_images = test_images.reshape(-1,79,95,79,1)
            l = validing(test_labels, test_sexs, test_images)
            print_process(IH_valid_iter)
            validation_loss.append(l)
            
            if(IH_valid_iter.current_pointer == 0):
                break
            
        validation_loss = np.array(validation_loss)
        print("\n Validation MSE: {:10}; MAE: {:10}".format(np.round(validation_loss.mean(0)[0],5),
                                                            np.round(validation_loss.mean(0)[1],5)))
        

        a = []
        for idx in range(len(model.weights)):
            a.append(model.weights[idx].numpy())
        with open("sec_RE_model_{}_{}.pickle".format(ep, np.round(validation_loss.mean(0)[1],0)), "wb") as fp:   #Pickling
            pickle.dump(a, fp)
            
        
    print("Training...")
    training_loss = []
    while(True):
        train_labels, train_sexs, train_images = IH_train_iter.GetData()
        train_images = train_images.reshape(-1,79,95,79,1)
        l = training(train_labels, train_sexs, train_images)
        
        print_process(IH_train_iter)
        training_loss.append(l)
        
        if(IH_train_iter.current_pointer == 0):
            break
    
    training_loss = np.array(training_loss)
    print("\n Training MSE: {:10}; MAE: {:10}".format(np.round(training_loss.mean(0)[0],5),
                                                        np.round(training_loss.mean(0)[1],5)))

