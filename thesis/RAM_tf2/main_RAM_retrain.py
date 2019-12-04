from __future__ import absolute_import, division, print_function, unicode_literals

import os
import copy
import pickle
import argparse

import tensorflow as tf
import numpy as np
from scipy import ndimage, misc

print(tf.__version__)

from networks.net3_big_heatmap import generate_model
from machine_learning.data_iterator import read_data, Data_Iterator
from machine_learning.utils import print_process

parser = argparse.ArgumentParser(description='TF2 train flags')
parser.add_argument('--saved_model_path', default=os.getcwd() + '\\saved_model\\' + 'model_re_10_13.0.pickle', type=str,
                    help='Path to model weights to load')
parser.add_argument('--csv_dir_path', default=os.getcwd() + '\\DATA\\', type=str,
                    help='Directory path storing label csv files')
parser.add_argument('--train_max_epoch', default=20, type=int, help='Number of train epochs')
parser.add_argument('--save_heatmap_path', default=os.getcwd() + '\\heatmaps\\', type=str, help='Heatmap save dir')
parser.add_argument('--model_save_path',    default=os.getcwd() + '\\saved_model\\', type=str, help='Dir path to save model weights')

args = parser.parse_args()


def copy_weights_into_model(model, weights):
    weightsidx = 0

    # copy layer weights
    for idx in range(len(model.layers)):
        print(model.layers[idx].name)
        # load weights of similar layers
        if (weightsidx == 32):
            break
        # load weights of convolutional layers
        if (model.layers[idx].trainable and len(model.layers[idx].weights) == 2):
            model.layers[idx].set_weights(weights[weightsidx:weightsidx + 2])
            weightsidx += 2
        # load weights of bath norm
        elif (model.layers[idx].trainable and len(model.layers[idx].weights) == 4):
            ws = copy.deepcopy(weights[weightsidx:weightsidx + 2])
            ws.append(np.zeros_like(ws[0]))
            ws.append(np.ones_like(ws[0]))
            model.layers[idx].set_weights(ws)
            weightsidx += 4

    # freeze convolutional layers
    for idx in range(1, len(model.layers) - 2):
        model.layers[idx].trainable = False
    return model


@tf.function
def tf_training(train_labels, train_sexs, train_images):
    l = model.train_on_batch(x=[train_images, train_sexs],
                             y=train_labels)
    return l


@tf.function
def tf_validating(test_labels, test_sexs, test_images):
    l = model.test_on_batch(x=[test_images, test_sexs],
                            y=test_labels)
    return l


model = generate_model(0.1)
model.build([(None, 79, 95, 79, 1), (None, 1)])

# LOAD SAVED WEIGHTS
with open(args.saved_model_path, 'rb') as f:
    weights = pickle.load(f)

# copy weights ##########################
model = copy_weights_into_model(model=model,
                                weights=weights)

model.compile(loss="mse",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=["mae"])
del weights

# print model summary
model.summary()

# RAM LAST LAYER TRAIN
ih_train_info, ih_valid_info = read_data(args.csv_dir_path, 'inhouse')

IH_train_iter = Data_Iterator(ih_train_info, batch_size=3)
IH_valid_iter = Data_Iterator(ih_valid_info, batch_size=3)

def validate(data_iter,
             current_epoch,
             weight_save_name):
    validation_loss = []
    print("Validating...")

    # iterate validation data
    while(True):

        test_labels, test_sexs, test_images = data_iter.GetData()
        test_images = test_images.reshape(-1, 79, 95, 79, 1)
        l = tf_validating(test_labels, test_sexs, test_images)
        print_process(data_iter, current_epoch)
        validation_loss.append(l)

        if (data_iter.current_pointer == 0):
            break

    validation_loss = np.array(validation_loss)
    print("\n Validation MSE: {:10}; MAE: {:10}".format(np.round(validation_loss.mean(0)[0], 5),
                                                        np.round(validation_loss.mean(0)[1], 5)))

    # save model weights into pickle file
    weight_list = []
    for idx in range(len(model.weights)):
        weight_list.append(model.weights[idx].numpy())
    with open(weight_save_name + "_{}_{}.pickle".format(current_epoch,
                                                        np.round(validation_loss.mean(0)[1], 0)), "wb") as fp:
        pickle.dump(weight_list, fp)


def train(data_iter,
         current_epoch):
    training_loss = []
    print("Training...")

    # iterate training data
    while(True):
        train_labels, train_sexs, train_images = data_iter.GetData()
        train_images = train_images.reshape(-1, 79, 95, 79, 1)
        l = tf_training(train_labels, train_sexs, train_images)

        print_process(data_iter, current_epoch)
        training_loss.append(l)

        if (data_iter.current_pointer == 0):
            break

    training_loss = np.array(training_loss)
    print("\n Training MSE: {:10}; MAE: {:10}".format(np.round(training_loss.mean(0)[0], 5),
                                                      np.round(training_loss.mean(0)[1], 5)))

for epoch in range(args.train_max_epoch):

    # VALIDATION
    if (IH_train_iter.current_pointer == 0):
        validate(data_iter=IH_valid_iter,
                 current_epoch=epoch,
                 weight_save_name=args.model_save_path + 'model_RAM')

    # TRAIN
    train(data_iter=IH_train_iter,
          current_epoch=epoch)


# generate heatmaps...

mig, ctr = read_data(args.csv_dir_path, 'migraine')

MIG_iter = Data_Iterator(mig, batch_size=1)
CTR_iter = Data_Iterator(ctr, batch_size=1)

# generate model that contains only the convolutional layers
model2 = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-5].output)

model_variables = model.trainable_variables
W = model_variables[0].numpy() # weights of the output dense layer


def generate_heatmap(data_iter, Dense_W):
    d = dict()
    d['image'] = []
    d['labels'] = []
    d['sexs'] = []
    d['heatmap'] = []

    # iterate over test data
    while (True):
        labels, sexs, images = data_iter.GetData()
        images = images.reshape(-1, 79, 95, 79, 1)

        # determine last feature map
        pred = model2.predict(x=[images, sexs])

        # compute RAM map
        RAM = np.zeros((pred.shape[1], pred.shape[2], pred.shape[3]))
        for idx in range(pred.shape[-1]):
            RAM = RAM + pred[0, :, :, :, idx] * Dense_W[idx, 0]

        heatmap = ndimage.zoom(RAM, (79/RAM.shape[0], 95/RAM.shape[1], 79/RAM.shape[2]))
                                                                       # [5*15.8 = 79,
                                                                       #  6*15.83 = 95,
                                                                       #  5*15.8 = 79]

        # save data
        d['labels'].append(labels)
        d['heatmap'].append(heatmap)
        d['sexs'].append(sexs)
        d['image'].append(images)

        if (data_iter.current_pointer == 0):
            break
    return d


###########################################################

mig = generate_heatmap(data_iter=MIG_iter,
                       Dense_W=W)
ctr = generate_heatmap(data_iter=CTR_iter,
                       Dense_W=W)

# average heatmaps
ctr_heatmap = np.array(ctr['heatmap']).mean(0)
mig_heatmap = np.array(mig['heatmap']).mean(0)

np.save(args.save_heatmap_path + 'ctr_heatmap_small.npy', ctr_heatmap)
np.save(args.save_heatmap_path + 'mig_heatmap_small.npy', mig_heatmap)
