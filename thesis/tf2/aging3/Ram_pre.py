from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns

print(tf.__version__)

import glob
import os
import sys
import copy

import cv2 as cv


from net3_model_RAM import generate_model
from defs import read_data, Data_Iterator


model = generate_model(0.1)
model.build([(None,79,95,79,1),(None,1)])

# LOAD SAVED WEIGHTS
with open('RE_model_9_3.0.pickle', 'rb') as f:
    weights = pickle.load(f)

# copy weights ##########################

weightsidx = 0
for idx in range(len(model.layers)):
    if(weightsidx == 32):
        break
    if(model.layers[idx].trainable and len(model.layers[idx].weights)==2):
        model.layers[idx].set_weights(weights[weightsidx:weightsidx+2])
        weightsidx += 2
    elif(model.layers[idx].trainable and len(model.layers[idx].weights)==4):
        ws = copy.deepcopy(weights[weightsidx:weightsidx+2])
        ws.append(np.zeros_like(ws[0]))
        ws.append(np.ones_like(ws[0]))
        model.layers[idx].set_weights(ws)
        weightsidx += 2
        
        
for idx in range(1,len(model.layers)-2):
    model.layers[idx].trainable = False


model.compile(loss="mse", 
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
              metrics=["mae"])

del weightsidx, idx
del weights, ws

model.summary()
#%% RAM LAST LAYER TRAIN

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

ih_train_info, ih_valid_info = read_data(os.getcwd() + '\\DATA\\', 'inhouse')

IH_train_iter  = Data_Iterator(ih_train_info, batch_size=1)
IH_valid_iter  = Data_Iterator(ih_valid_info, batch_size=1)

maxEps = 10

for ep in range(maxEps):
    
    #VALIDATION
    if(IH_train_iter.current_pointer == 0 and 
       ep != 0):
        
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
        
        '''
        a = []
        for idx in range(len(model.weights)):
            a.append(model.weights[idx].numpy())
        with open("RAM_RE_model_{}_{}.pickle".format(ep, np.round(validation_loss.mean(0)[1],0)), "wb") as fp:   #Pickling
            pickle.dump(a, fp)
        '''
        
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




#%% HEATMAPS

#"""

from scipy import ndimage, misc
mig, ctr = read_data(os.getcwd() + '\\DATA\\', 'migraine')

MIG_iter  = Data_Iterator(mig, batch_size = 1)
CTR_iter  = Data_Iterator(ctr, batch_size = 1)

model2 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-5].output)

model_variables = model.trainable_variables
W = model_variables[0].numpy()

###########################################################
mig = dict()
mig['image'] = []
mig['labels'] = []
mig['sexs'] = []
mig['heatmap'] = []
while(True):
    labels, sexs, images = MIG_iter.GetData()
    images = images.reshape(-1,79,95,79,1)
    
    pred = model2.predict(x = [images,sexs])
    RAM = np.zeros((pred.shape[1],pred.shape[2],pred.shape[3]))
    for idx in range(pred.shape[-1]):
        RAM = RAM + pred[0,:,:,:,idx]*1#W[idx,0]
        
    heatmap = ndimage.zoom(RAM, (15.8, 15.833333333333334, 15.8))
    
    mig['labels'].append(labels)
    mig['heatmap'].append(heatmap)
    mig['sexs'].append(sexs)
    mig['image'].append(images)
    
    if(MIG_iter.current_pointer == 0):
        break
    
    
ctr = dict()
ctr['labels'] = []
ctr['sexs'] = []
ctr['image'] = []
ctr['heatmap'] = []
while(True):
    labels, sexs, images = CTR_iter.GetData()
    images = images.reshape(-1,79,95,79,1)
    
    pred = model2.predict(x = [images,sexs])
    RAM = np.zeros((pred.shape[1],pred.shape[2],pred.shape[3]))
    for idx in range(pred.shape[-1]):
        RAM = RAM + pred[0,:,:,:,idx]*1#W[idx,0]
        
    heatmap = ndimage.zoom(RAM, (15.8, 15.833333333333334, 15.8))
    
    ctr['labels'].append(labels)
    ctr['heatmap'].append(heatmap)
    ctr['sexs'].append(sexs)
    ctr['image'].append(images)
    
    if(CTR_iter.current_pointer == 0):
        break


###########################################################
#%%
ctr_heatmap = np.array(ctr['heatmap']).mean(0)
mig_heatmap = np.array(mig['heatmap']).mean(0)
dif_heatmap = mig_heatmap-ctr_heatmap

average_hm = np.concatenate((np.array(ctr['heatmap']),
                             np.array(mig['heatmap'])), axis = 0).mean(0)

np.save('ctr_heatmap_small.npy',ctr_heatmap)
np.save('mig_heatmap_small.npy',mig_heatmap)
np.save('dif_heatmap_small.npy',dif_heatmap)
np.save('aver_heatma_smallp.npy',average_hm)

def normalize(image, type_ = '01'):
    '''
    type_: "01" - between 0 and 1
    '''
    if(type_ == '01'):
        image = image - np.min(image)
        image = image / np.max(image)
    return image

global_normalized = normalize(np.array([ctr_heatmap,
                                        mig_heatmap,
                                        dif_heatmap]))
ctr_heatmap = global_normalized[0,:,:,:]
mig_heatmap = global_normalized[1,:,:,:]
dif_heatmap = global_normalized[2,:,:,:]  
    
f,axes = plt.subplots(2,4)
for idx in range(0, images.shape[3]):
    
    example_slice = normalize(images[0,:,:,idx,0])
    example_heatmap_ctr = ctr_heatmap[:,:,idx].astype(np.float32)
    example_heatmap_mig = mig_heatmap[:,:,idx].astype(np.float32)
    example_heatmap_dif = dif_heatmap[:,:,idx].astype(np.float32)
    
    axes[1,1].imshow(example_slice, cmap='Greys')
    axes[1,1].pcolormesh(example_heatmap_ctr, cmap='Purples', alpha = 0.5)
    axes[1,2].imshow(example_slice, cmap='Greys')
    axes[1,2].pcolormesh(example_heatmap_mig, cmap='Purples', alpha = 0.5) 
    axes[1,3].imshow(example_slice, cmap='Greys')
    axes[1,3].pcolormesh(example_heatmap_dif, cmap='Purples', alpha = 0.5) 

    axes[0,0].imshow(example_slice, cmap='Greys')
    axes[0,1].imshow(example_heatmap_ctr)
    axes[0,2].imshow(example_heatmap_mig)
    axes[0,3].imshow(example_heatmap_dif)
    
    
    plt.pause(0.5)
    axes[0,0].cla()
    axes[0,1].cla()
    axes[0,2].cla()
    axes[0,3].cla()
    axes[1,0].cla()
    axes[1,1].cla()
    axes[1,2].cla()
    axes[1,3].cla()
#"""